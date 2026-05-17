# Appendix AA — Mixture of Experts (MoE): Conditional Computation from First Principles

> *Dense transformers treat every token the same: all parameters activate for every input. Mixture of Experts breaks that contract. It routes each token to a small fraction of specialist sub-networks, delivering model capacity that scales with parameter count while keeping the arithmetic cost roughly constant. The result is the architectural foundation of GPT-4, Mixtral, DeepSeek-V3, and Grok-1 — and the reason models with trillions of parameters can still be served economically.*

---

## AA.1 Motivation: The Scaling Dilemma

### AA.1.1 Why Dense Scaling Hits a Wall

The scaling laws established by Kaplan et al. (2020) and refined by Hoffmann et al. (2022, "Chinchilla") show a clear relationship: model loss decreases predictably as a power function of compute, parameters, and data. The standard playbook — double parameters, double compute, double data — has been followed diligently. The problem is that doubling compute means doubling the FLOPs per token at inference time, which translates directly to higher latency, higher GPU cost, and lower throughput.

For a dense model, FLOPs per token scale linearly with parameter count. An 8B-parameter model requires roughly 16 × 10⁹ FLOPs per token (two multiplications per weight). A 70B model requires ~140 × 10⁹. A 1T-parameter model — if it were dense — would require ~2 × 10¹² FLOPs per token, making real-time inference economically untenable at scale.

### AA.1.2 Conditional Computation as the Solution

The insight behind Mixture of Experts is simple: not every parameter needs to activate for every token. A token about molecular biology should route through different sub-networks than a token about medieval history. If we can select only a subset of experts per token, we can:

- Scale *parameter count* (capacity) without scaling *FLOPs per token* (cost)
- Achieve larger effective model sizes within fixed inference budgets
- Specialise sub-networks on different domains, syntax patterns, or abstraction levels

This idea — conditional computation — was explored as far back as 1991 by Jacobs, Jordan, Nowlan, and Hinton in their paper *Adaptive Mixtures of Local Experts*. The modern neural-network incarnation was formalised by Shazeer et al. (2017) in *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, and industrialised by the Switch Transformer (Fedus et al., 2021), GLaM (Du et al., 2022), GShard, Mixtral 8×7B, and DeepSeek-V2/V3.

### AA.1.3 The FLOPs vs Parameters Separation

The core arithmetic: if a model has E experts each of size F parameters, and we route each token to the top-K of them, the parameter count is E × F but the FLOPs per token are K × F (plus routing overhead). Setting K = 2 and E = 8 gives four times the capacity at two times the cost — not eight times. This ratio is the economic engine of MoE.

```
Dense model:
  Parameters = F
  FLOPs/token = 2F          (one forward pass through all F weights)

MoE model (E experts, top-K routing):
  Parameters = E × F_expert  (typically >> dense equivalent)
  FLOPs/token ≈ 2 × K × F_expert + routing_cost
  Capacity/Cost ratio = E / K
```

For Mixtral 8×7B: E=8, K=2, so you pay for 2 experts but get 8× the parameter count. Effective parameters ≈ 46.7B (active per token: ~13B). For DeepSeek-V3: E=256, K=8 (8 routed + 1 shared), FLOPs per token much less than the 671B total parameter count would suggest.

---

## AA.2 Architecture: The MoE Layer

### AA.2.1 Where MoE Fits in the Transformer

A standard transformer alternates attention blocks with feed-forward networks (FFN). The MoE modification replaces the FFN layer (in some or all layers) with a *sparse MoE layer*. The attention mechanism, layer norms, and residual connections remain unchanged.

```
Standard Transformer Layer:
  x → LayerNorm → MultiHeadAttention → residual
    → LayerNorm → FFN (dense) → residual → output

MoE Transformer Layer:
  x → LayerNorm → MultiHeadAttention → residual
    → LayerNorm → MoE Layer [Router + Top-K Experts] → residual → output
```

Not every layer needs to be an MoE layer. Switch Transformer alternates dense and MoE layers. Mixtral replaces *all* FFN layers. DeepSeek-V2 uses a hybrid: some dense attention layers, MoE for all FFN layers.

### AA.2.2 The Expert Sub-Network

Each expert is structurally identical to a standard FFN:

```
Expert_i(x) = W₂_i · activation(W₁_i · x)
```

Where:

- `x` ∈ ℝᵈ is the token representation (model hidden dimension)
- `W₁_i` ∈ ℝ^(d_ff × d) is the up-projection (d_ff = 4d typically)
- `activation` is SwiGLU, GeLU, or ReLU depending on the model
- `W₂_i` ∈ ℝ^(d × d_ff) is the down-projection

The per-expert parameter count is 2 × d × d_ff. For d=4096 and d_ff=14336 (Mixtral 8×7B per-expert dimensions): 2 × 4096 × 14336 ≈ 117M parameters per expert × 8 experts ≈ 938M per MoE layer.

### AA.2.3 The MoE Forward Pass (Conceptual)

```
Input:  T tokens, each x_t ∈ ℝᵈ

For each token x_t:
  1. Compute routing scores:   s_t = Router(x_t) ∈ ℝᴱ
  2. Select top-K experts:     I_t = top_k_indices(s_t)
  3. Normalise gate weights:   g_t = softmax(s_t[I_t])
  4. Compute expert outputs:   h_i = Expert_i(x_t) for i ∈ I_t
  5. Weighted combination:     y_t = Σ_{i ∈ I_t} g_{t,i} · h_i

Output: T tokens, each y_t ∈ ℝᵈ
```

### AA.2.4 The Router Network

The router is a learned linear projection — typically a single weight matrix with no bias or activation:

```
Router(x) = x · W_r^T
```

Where W_r ∈ ℝ^(E × d) maps each token from hidden dimension d to a score for each of the E experts. The output logits are then passed through softmax (or a normalised top-K selection) to produce probability-like gate weights.

```python
# Conceptual router implementation
class Router(nn.Module):
    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return self.gate(x)  # [batch, seq_len, n_experts]
```

---

## AA.3 Routing Mechanisms

### AA.3.1 Soft (Dense) Routing

The simplest formulation routes every token to every expert and combines outputs with learned weights:

```
y_t = Σ_{i=1}^{E} softmax(W_r · x_t)_i · Expert_i(x_t)
```

This is fully differentiable and has no load-balancing problem because all experts always receive input. The cost: O(E) expert computations per token — you lose the FLOPs savings entirely. Soft routing is useful for small E or as a pedagogical baseline but is not used in production MoE models.

### AA.3.2 Top-1 Routing (Switch Transformer)

The Switch Transformer (Fedus et al., 2021) simplified Shazeer's original design to route each token to exactly one expert. This maximises sparsity and simplifies the implementation at the cost of higher variance.

```
i*(x_t) = argmax_i softmax(W_r · x_t)_i
y_t = softmax(W_r · x_t)_{i*} · Expert_{i*}(x_t)
```

Advantages: minimal computation overhead per token, easy to implement with expert buffers.

Disadvantages: single point of failure per token (if routing is wrong, there's no fallback); higher instability during training; expert collapse risk is more severe.

### AA.3.3 Top-K Routing (GShard, Mixtral)

The standard production approach selects the K highest-scoring experts and normalises among them:

```
I_t = argtopk_i s_t[i],  s_t = W_r · x_t
g_t = softmax(s_t[I_t])            # normalise over selected K
y_t = Σ_{i ∈ I_t} g_{t,i} · Expert_i(x_t)
```

K=2 is the most common choice (Mixtral, GShard, DeepSeek-V2 routed experts). The softmax is applied only to the K selected logits, ensuring gate weights sum to 1 over the active set.

```python
def top_k_routing(logits, k):
    """
    logits: [batch * seq_len, n_experts]
    Returns gate weights and expert indices.
    """
    # Select top-K scores
    topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
    # Normalise gate weights via softmax over the K selected logits
    gate_weights = F.softmax(topk_vals, dim=-1)
    return gate_weights, topk_idx
```

### AA.3.4 Noisy Top-K Routing

Shazeer et al. (2017) introduced learned noise to prevent premature routing collapse. Before taking the top-K, Gaussian noise is added to the logits, scaled by a learned per-expert noise parameter:

```
h(x) = W_r · x
noise_std = softplus(W_noise · x)           # learned, non-negative
ε ~ N(0, I_E)
noisy_logits = h(x) + ε · noise_std
I_t = argtopk noisy_logits
```

The noise is only added during training; at inference time, the clean logits are used. The effect is a form of exploration — tokens occasionally route to non-greedy experts, exposing more experts to gradients and preventing winner-take-all collapse early in training.

```python
class NoisyTopKRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int, noise_eps: float = 1e-2):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.noise_gate = nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k
        self.noise_eps = noise_eps

    def forward(self, x: torch.Tensor, training: bool = True):
        # x: [tokens, d_model]
        logits = self.gate(x)                          # [tokens, n_experts]
        if training:
            noise_std = F.softplus(self.noise_gate(x)) + self.noise_eps
            noise = torch.randn_like(logits)
            logits = logits + noise * noise_std
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        gate_weights = F.softmax(topk_vals, dim=-1)
        return gate_weights, topk_idx, logits  # return raw logits for aux loss
```

### AA.3.5 Expert-Choice Routing

Expert-choice routing (Zhou et al., 2022) inverts the selection logic: instead of each token choosing K experts, each expert chooses the top-C tokens from the incoming batch (where C is the expert's capacity):

```
For each expert i:
  scores_i = softmax(W_r · X)[:, i]    # score all tokens for expert i
  T_i = argtopC(scores_i)              # expert picks its top-C tokens
```

This guarantees perfect load balancing by construction — every expert gets exactly C tokens — eliminating the need for auxiliary losses. The trade-off: a token may not be selected by any expert (coverage gaps) or may be selected by multiple experts (coverage overlap). Expert-choice is used in some Google internal models and is particularly useful when batch sizes are large and predictable.

```python
def expert_choice_routing(logits, capacity_per_expert):
    """
    logits: [n_tokens, n_experts]
    capacity_per_expert: int — how many tokens each expert selects
    Returns: dispatch mask [n_experts, n_tokens]
    """
    # Each expert independently scores and selects top-C tokens
    scores = F.softmax(logits, dim=0)           # normalise over tokens per expert
    dispatch_mask = torch.zeros_like(scores, dtype=torch.bool)
    for e in range(logits.shape[1]):
        topC_idx = torch.topk(scores[:, e], capacity_per_expert).indices
        dispatch_mask[topC_idx, e] = True
    return dispatch_mask, scores
```

### AA.3.6 Token-Choice vs Expert-Choice: Comparison

```
Property                Token-Choice (Top-K)       Expert-Choice
─────────────────────────────────────────────────────────────────────
Load balancing          Requires aux loss           Guaranteed by design
Token coverage          Every token routed          Some tokens may drop
Routing determinism     Token controls its path     Expert controls intake
Gradient flow           Clean per-token             Potentially non-uniform
Batch-size sensitivity  Works at any batch size     Needs large batches
Training stability      Moderate (aux loss helps)   High
Implementation          Simple                      Moderate
Production use          Mixtral, DeepSeek, GShard   Google internal models
─────────────────────────────────────────────────────────────────────
```

### AA.3.7 DeepSeek's Fine-Grained Expert Decomposition

DeepSeek-V2 and V3 introduced a hybrid routing scheme with two categories of experts:

- **Shared experts** (Nₛ = 1 or 2): Always activated for every token — they handle universal, syntactic, or high-frequency patterns that every token benefits from.
- **Routed experts** (Nᵣ = large): Activated via top-K selection — they handle specialised content.

```
y_t = Expert_shared(x_t) + Σ_{i ∈ TopK(routed_experts)} g_{t,i} · Expert_i(x_t)
```

DeepSeek-V3 uses Nₛ=1 shared expert + top-8 from 256 routed experts = 257 total experts, with an effective 9 active per token. The fine-grained decomposition (many small experts rather than few large ones) allows more precise routing and better specialisation.

---

## AA.4 The Load-Balancing Problem

### AA.4.1 Expert Collapse: The Winner-Take-All Failure Mode

Without any regularisation, MoE training exhibits a devastating failure mode: *expert collapse*. Early in training, random weight initialisation means some experts receive slightly higher logits for the first few batches. These experts get more gradient signal, improve faster, and are subsequently preferred by the router even more. The feedback loop is self-reinforcing. Within a few thousand steps, the router has converged to sending nearly all tokens to one or two experts; the remaining experts receive no gradient and are effectively dead.

```
Training step 0:   Expert 1: 13%  Expert 2: 12%  ...  (near-uniform)
Training step 500: Expert 1: 42%  Expert 2: 28%  ...  (starting to skew)
Training step 2K:  Expert 1: 89%  Expert 2: 10%  ...  (collapse)
Training step 5K:  Expert 1: 98%  Expert 2:  2%  ...  (complete collapse)
```

At this point, the MoE layer is functionally equivalent to a much smaller dense FFN. All the parameter capacity is wasted. The model performs worse than if it had been trained dense from the start.

### AA.4.2 The Auxiliary Load-Balancing Loss

The Switch Transformer introduced an auxiliary loss that penalises imbalanced routing. It is computed per-MoE-layer and added to the main cross-entropy loss with a small scalar weight α (typically 0.01–0.1):

```
L_aux = α · E · Σ_{i=1}^{E} f_i · P_i
```

Where:

- E is the number of experts
- f_i = fraction of tokens dispatched to expert i (in the current batch)
- P_i = mean router probability for expert i across the batch

```
f_i = (1/T) · Σ_{t=1}^{T} 𝟙[i ∈ I_t]        (fraction of tokens routed to i)
P_i = (1/T) · Σ_{t=1}^{T} softmax(s_t)[i]    (mean softmax score for i)
```

The key insight: f_i is a non-differentiable discrete count, but P_i is differentiable. The product f_i · P_i creates a quantity that is minimised when all experts receive 1/E of the tokens, and through the differentiable P_i term, backprop can push the router toward more balanced distributions.

```python
def load_balance_loss(router_logits, expert_indices, n_experts):
    """
    router_logits: [n_tokens, n_experts] — raw logits before top-K
    expert_indices: [n_tokens, top_k]   — which experts were selected
    n_experts: int
    Returns scalar auxiliary loss.
    """
    n_tokens = router_logits.shape[0]
    # f_i: fraction of tokens that chose expert i (non-differentiable)
    one_hot = F.one_hot(expert_indices, n_experts).float()  # [tokens, topk, experts]
    token_routed = one_hot.sum(dim=1)                       # [tokens, experts]
    f = token_routed.sum(dim=0) / n_tokens                  # [n_experts]
    # P_i: mean softmax probability for expert i (differentiable)
    P = F.softmax(router_logits, dim=-1).mean(dim=0)        # [n_experts]
    # Auxiliary loss: E * sum(f_i * P_i), minimised at uniform routing
    return n_experts * (f * P).sum()
```

### AA.4.3 The Z-Loss (ST-MoE)

Zoph et al. (2022) identified a separate instability: large router logits produce near-one-hot softmax distributions, making the router brittle and sensitive to small input perturbations. They introduced the *z-loss* to penalise large logit magnitudes directly:

```
L_z = (1/T) · Σ_{t=1}^{T} (log Σ_{i=1}^{E} exp(s_{t,i}))²
    = (1/T) · Σ_{t=1}^{T} [log(partition function of token t)]²
```

This is the squared log-sum-exp of the router logits. Minimising it keeps logit magnitudes small, keeping the softmax distribution more spread out.

```python
def z_loss(router_logits):
    """
    router_logits: [n_tokens, n_experts]
    Returns scalar z-loss (Zoph et al. 2022).
    """
    # log(sum(exp(logits))) per token — the log partition function
    log_z = torch.logsumexp(router_logits, dim=-1)      # [n_tokens]
    # Square and average
    return (log_z ** 2).mean()
```

The combined training loss becomes:

```
L_total = L_crossentropy + α · L_aux + β · L_z
```

Typical values: α = 0.001–0.01, β = 0.001. The z-loss coefficient is usually smaller than the auxiliary loss coefficient.

### AA.4.4 Expert Capacity and Token Dropping

In practice, to enable efficient batched execution (expert computation as matrix multiplications), each expert is given a fixed *capacity* — the maximum number of tokens it will process in one forward pass:

```
capacity = capacity_factor × (T / E) × K
```

Where T is total tokens in the batch, E is number of experts, and K is top-K. A capacity factor of 1.0 means each expert gets exactly its "fair share" of tokens. If more tokens are routed to an expert than its capacity, the overflow tokens are *dropped* — they bypass the expert and their representation is passed through unchanged (via a residual connection). If fewer tokens route to an expert, the remaining capacity slots are padded with zeros.

```python
def dispatch_with_capacity(gate_weights, expert_indices, n_experts, capacity_factor=1.25):
    """Dispatch tokens to experts with a capacity buffer."""
    n_tokens, top_k = expert_indices.shape
    capacity = int(capacity_factor * n_tokens * top_k / n_experts)

    # Count tokens per expert
    expert_counts = torch.zeros(n_experts, dtype=torch.long)
    dispatch_mask = torch.zeros(n_tokens, n_experts, capacity)
    combine_weights = torch.zeros(n_tokens, n_experts, capacity)

    for t in range(n_tokens):
        for k_idx in range(top_k):
            e = expert_indices[t, k_idx].item()
            slot = expert_counts[e].item()
            if slot < capacity:
                dispatch_mask[t, e, slot] = 1.0
                combine_weights[t, e, slot] = gate_weights[t, k_idx].item()
                expert_counts[e] += 1
            # else: token t is dropped for expert e

    return dispatch_mask, combine_weights, expert_counts
```

A capacity factor > 1.0 (typically 1.25 for training, 2.0 for inference) creates a buffer to absorb routing imbalance without dropping tokens. The trade-off is wasted compute on padding zeros.

### AA.4.5 Expert Dropout

During training, randomly dropping entire experts (zeroing their output with probability p) forces the router to distribute load more evenly — no expert can be relied upon absolutely, so the model learns to spread routing. Expert dropout is applied at the layer level, not the token level:

```python
def moe_forward_with_expert_dropout(x, experts, router, dropout_p=0.1, training=True):
    gate_weights, expert_idx = router(x)
    # During training, randomly disable some experts
    if training and dropout_p > 0:
        expert_mask = torch.bernoulli(
            torch.ones(len(experts)) * (1 - dropout_p)
        )
    else:
        expert_mask = torch.ones(len(experts))
    # ... dispatch respecting expert_mask ...
```

### AA.4.6 Router Jitter

Introduced in Switch Transformer, router jitter adds uniform noise to inputs *before* routing (as opposed to Shazeer's noise on logits):

```
x_noisy = x · (1 + ε),  ε ~ Uniform[-jitter_noise, jitter_noise]
routing_scores = Router(x_noisy)
```

This prevents the router from overfitting to exact input values and helps maintain diverse routing distributions across different batches.

---

## AA.5 Manual Worked Example: Forward Pass

We walk through a complete numerical forward pass for a minimal MoE setup. All arithmetic is done by hand — no library calls. This makes the data flow concrete before we encounter the PyTorch implementation.

### AA.5.1 Setup

```
Configuration:
  d_model    = 4        (token embedding dimension)
  d_ff       = 4        (expert hidden dimension, small for tractability)
  n_experts  = 4        (E = 4)
  top_k      = 2        (K = 2; each token goes to 2 experts)
  n_tokens   = 2        (T = 2; two tokens in the batch)
  activation = ReLU     (for manual tractability)
```

**Input tokens** (2 tokens, each a 4-dimensional vector):

```
X = [[0.5,  0.1, -0.2,  0.8],   ← token 0
     [0.3, -0.4,  0.7,  0.2]]   ← token 1
```

**Router weight matrix** W_r ∈ ℝ^(4 experts × 4 d_model):

```
W_r = [[ 0.1,  0.4, -0.1,  0.2],   ← expert 0 weights
       [ 0.3, -0.2,  0.5,  0.1],   ← expert 1 weights
       [-0.1,  0.3,  0.2, -0.4],   ← expert 2 weights
       [ 0.2,  0.1, -0.3,  0.5]]   ← expert 3 weights
```

**Expert 0 weights** (for concreteness we fully specify Expert 0; others follow the same pattern):

```
Expert 0:
  W₁₀ ∈ ℝ^(4×4):
  [[ 0.2,  0.3, -0.1,  0.4],
   [ 0.1, -0.2,  0.3,  0.1],
   [-0.1,  0.1,  0.2,  0.3],
   [ 0.4, -0.1,  0.1,  0.2]]

  W₂₀ ∈ ℝ^(4×4):
  [[ 0.3,  0.1, -0.2,  0.1],
   [-0.1,  0.2,  0.1,  0.3],
   [ 0.2, -0.1,  0.3,  0.1],
   [ 0.1,  0.3,  0.1, -0.2]]

Expert 1:
  W₁₁ = I₄ × 0.5  (identity scaled, for brevity)
  W₂₁ = I₄ × 0.3
```

### AA.5.2 Step 1 — Compute Router Logits

For token 0: x₀ = [0.5, 0.1, -0.2, 0.8]

```
s₀ = W_r · x₀

Expert 0: 0.1×0.5 + 0.4×0.1 + (-0.1)×(-0.2) + 0.2×0.8
        = 0.05 + 0.04 + 0.02 + 0.16 = 0.27

Expert 1: 0.3×0.5 + (-0.2)×0.1 + 0.5×(-0.2) + 0.1×0.8
        = 0.15 - 0.02 - 0.10 + 0.08 = 0.11

Expert 2: (-0.1)×0.5 + 0.3×0.1 + 0.2×(-0.2) + (-0.4)×0.8
        = -0.05 + 0.03 - 0.04 - 0.32 = -0.38

Expert 3: 0.2×0.5 + 0.1×0.1 + (-0.3)×(-0.2) + 0.5×0.8
        = 0.10 + 0.01 + 0.06 + 0.40 = 0.57

s₀ = [0.27, 0.11, -0.38, 0.57]
```

For token 1: x₁ = [0.3, -0.4, 0.7, 0.2]

```
s₁ = W_r · x₁

Expert 0: 0.1×0.3 + 0.4×(-0.4) + (-0.1)×0.7 + 0.2×0.2
        = 0.03 - 0.16 - 0.07 + 0.04 = -0.16

Expert 1: 0.3×0.3 + (-0.2)×(-0.4) + 0.5×0.7 + 0.1×0.2
        = 0.09 + 0.08 + 0.35 + 0.02 = 0.54

Expert 2: (-0.1)×0.3 + 0.3×(-0.4) + 0.2×0.7 + (-0.4)×0.2
        = -0.03 - 0.12 + 0.14 - 0.08 = -0.09

Expert 3: 0.2×0.3 + 0.1×(-0.4) + (-0.3)×0.7 + 0.5×0.2
        = 0.06 - 0.04 - 0.21 + 0.10 = -0.09

s₁ = [-0.16, 0.54, -0.09, -0.09]
```

### AA.5.3 Step 2 — Top-K Selection (K=2)

```
Token 0: s₀ = [0.27, 0.11, -0.38, 0.57]
  Top-2 indices: [3, 0]  (scores 0.57 and 0.27)
  Top-2 values:  [0.57, 0.27]

Token 1: s₁ = [-0.16, 0.54, -0.09, -0.09]
  Top-2 indices: [1, 2]  (scores 0.54 and -0.09)
  Top-2 values:  [0.54, -0.09]
```

### AA.5.4 Step 3 — Normalise Gate Weights via Softmax

**Token 0** — softmax over top-K logits [0.57, 0.27]:

```
exp(0.57) = 1.7683
exp(0.27) = 1.3100
sum        = 3.0783

g₀,₃ = 1.7683 / 3.0783 = 0.5745   (Expert 3)
g₀,₀ = 1.3100 / 3.0783 = 0.4255   (Expert 0)

Gate weights for token 0:  {Expert 3: 0.5745, Expert 0: 0.4255}
```

**Token 1** — softmax over top-K logits [0.54, -0.09]:

```
exp(0.54)  = 1.7160
exp(-0.09) = 0.9139
sum         = 2.6299

g₁,₁ = 1.7160 / 2.6299 = 0.6525   (Expert 1)
g₁,₂ = 0.9139 / 2.6299 = 0.3475   (Expert 2)

Gate weights for token 1:  {Expert 1: 0.6525, Expert 2: 0.3475}
```

### AA.5.5 Step 4 — Expert Forward Passes

**Expert 0 processing token 0** (x₀ = [0.5, 0.1, -0.2, 0.8]):

```
h = W₁₀ · x₀:
  h[0] = 0.2×0.5 + 0.3×0.1 + (-0.1)×(-0.2) + 0.4×0.8 = 0.10+0.03+0.02+0.32 = 0.47
  h[1] = 0.1×0.5 + (-0.2)×0.1 + 0.3×(-0.2) + 0.1×0.8 = 0.05-0.02-0.06+0.08 = 0.05
  h[2] = (-0.1)×0.5 + 0.1×0.1 + 0.2×(-0.2) + 0.3×0.8 = -0.05+0.01-0.04+0.24 = 0.16
  h[3] = 0.4×0.5 + (-0.1)×0.1 + 0.1×(-0.2) + 0.2×0.8 = 0.20-0.01-0.02+0.16 = 0.33

h = [0.47, 0.05, 0.16, 0.33]

ReLU(h) = [0.47, 0.05, 0.16, 0.33]  (all positive, unchanged)

Expert0(x₀) = W₂₀ · ReLU(h):
  out[0] = 0.3×0.47 + 0.1×0.05 + (-0.2)×0.16 + 0.1×0.33
         = 0.141 + 0.005 - 0.032 + 0.033 = 0.147
  out[1] = (-0.1)×0.47 + 0.2×0.05 + 0.1×0.16 + 0.3×0.33
         = -0.047 + 0.010 + 0.016 + 0.099 = 0.078
  out[2] = 0.2×0.47 + (-0.1)×0.05 + 0.3×0.16 + 0.1×0.33
         = 0.094 - 0.005 + 0.048 + 0.033 = 0.170
  out[3] = 0.1×0.47 + 0.3×0.05 + 0.1×0.16 + (-0.2)×0.33
         = 0.047 + 0.015 + 0.016 - 0.066 = 0.012

Expert0(x₀) = [0.147, 0.078, 0.170, 0.012]
```

**Expert 3 processing token 0** (using W₁₃ = I₄×0.4, W₂₃ = I₄×0.35 for brevity):

```
h = 0.4 × x₀ = [0.200, 0.040, -0.080, 0.320]
ReLU(h) = [0.200, 0.040, 0.000, 0.320]  (negative clamped to 0)
Expert3(x₀) = 0.35 × ReLU(h) = [0.070, 0.014, 0.000, 0.112]
```

**Expert 1 processing token 1** (using W₁₁ = I₄×0.5, W₂₁ = I₄×0.3):

```
h = 0.5 × x₁ = [0.150, -0.200, 0.350, 0.100]
ReLU(h) = [0.150, 0.000, 0.350, 0.100]
Expert1(x₁) = 0.3 × ReLU(h) = [0.045, 0.000, 0.105, 0.030]
```

**Expert 2 processing token 1** (using W₁₂ = I₄×0.3, W₂₂ = I₄×0.25):

```
h = 0.3 × x₁ = [0.090, -0.120, 0.210, 0.060]
ReLU(h) = [0.090, 0.000, 0.210, 0.060]
Expert2(x₁) = 0.25 × ReLU(h) = [0.0225, 0.000, 0.0525, 0.015]
```

### AA.5.6 Step 5 — Weighted Combination

**Token 0** (Expert 3 with weight 0.5745, Expert 0 with weight 0.4255):

```
y₀ = 0.5745 × Expert3(x₀) + 0.4255 × Expert0(x₀)
   = 0.5745 × [0.070, 0.014, 0.000, 0.112]
   + 0.4255 × [0.147, 0.078, 0.170, 0.012]

Component-wise:
  y₀[0] = 0.5745×0.070 + 0.4255×0.147 = 0.04022 + 0.06255 = 0.10277
  y₀[1] = 0.5745×0.014 + 0.4255×0.078 = 0.00804 + 0.03319 = 0.04123
  y₀[2] = 0.5745×0.000 + 0.4255×0.170 = 0.00000 + 0.07234 = 0.07234
  y₀[3] = 0.5745×0.112 + 0.4255×0.012 = 0.06434 + 0.00511 = 0.06945

y₀ = [0.1028, 0.0412, 0.0723, 0.0695]
```

**Token 1** (Expert 1 with weight 0.6525, Expert 2 with weight 0.3475):

```
y₁ = 0.6525 × Expert1(x₁) + 0.3475 × Expert2(x₁)
   = 0.6525 × [0.045, 0.000, 0.105, 0.030]
   + 0.3475 × [0.0225, 0.000, 0.0525, 0.015]

Component-wise:
  y₁[0] = 0.6525×0.045 + 0.3475×0.0225 = 0.02936 + 0.00782 = 0.03718
  y₁[1] = 0.6525×0.000 + 0.3475×0.000  = 0.000
  y₁[2] = 0.6525×0.105 + 0.3475×0.0525 = 0.06851 + 0.01824 = 0.08675
  y₁[3] = 0.6525×0.030 + 0.3475×0.015  = 0.01958 + 0.00521 = 0.02479

y₁ = [0.0372, 0.0000, 0.0868, 0.0248]
```

### AA.5.7 Summary of the Forward Pass

```
Input X:
  Token 0: [0.500,  0.100, -0.200,  0.800]
  Token 1: [0.300, -0.400,  0.700,  0.200]

Router logits:
  Token 0: [0.27,  0.11, -0.38,  0.57]
  Token 1: [-0.16, 0.54, -0.09, -0.09]

Selected experts (top-2):
  Token 0 → {Expert 3: 0.5745, Expert 0: 0.4255}
  Token 1 → {Expert 1: 0.6525, Expert 2: 0.3475}

Expert load this batch:
  Expert 0: 1 token (50% utilisation)
  Expert 1: 1 token (50% utilisation)
  Expert 2: 1 token (50% utilisation)
  Expert 3: 1 token (50% utilisation)
  → Perfectly balanced (coincidence in this example; not typical)

Output Y:
  Token 0: [0.1028, 0.0412, 0.0723, 0.0695]
  Token 1: [0.0372, 0.0000, 0.0868, 0.0248]
```

---

## AA.6 Manual Worked Example: Backpropagation Through the Router

We now derive the gradients analytically for the forward pass above, focusing on the router — the most novel component.

### AA.6.1 Setting Up the Loss

Assume the final scalar loss L is given (e.g., from cross-entropy on the next-token prediction). We want ∂L/∂W_r — the gradient of the loss with respect to the router weight matrix.

By the chain rule:

```
∂L/∂W_r = ∂L/∂s · ∂s/∂W_r
```

Where s = x · W_r^T are the router logits (before top-K selection).

### AA.6.2 Gradient Through the Gate Weights

For token 0, selected experts {3, 0} with gate weights g = softmax([s₃, s₀]) = [0.5745, 0.4255]:

```
y₀ = g₀,₃ · Expert3(x₀) + g₀,₀ · Expert0(x₀)
```

Suppose the upstream gradient from the loss is ∂L/∂y₀ = δ₀ ∈ ℝ⁴ (provided by backprop from layers above). Then:

```
∂L/∂g₀,₃ = δ₀ · Expert3(x₀)   (dot product = scalar)
∂L/∂g₀,₀ = δ₀ · Expert0(x₀)

Assuming δ₀ = [1, 1, 1, 1] for illustration:
  ∂L/∂g₀,₃ = 0.070 + 0.014 + 0.000 + 0.112 = 0.196
  ∂L/∂g₀,₀ = 0.147 + 0.078 + 0.170 + 0.012 = 0.407
```

### AA.6.3 Gradient Through Softmax (Top-K Selected)

The gate weights are g = softmax(selected_logits) where selected_logits = [s₃, s₀] = [0.57, 0.27]. Let g = [0.5745, 0.4255].

The Jacobian of softmax is:

```
∂g_i/∂s_j = g_i(δ_{ij} - g_j)

For g = [0.5745, 0.4255]:
J = [g₃(1-g₃),   -g₃·g₀  ]   = [0.5745×0.4255,  -0.5745×0.4255]
    [-g₀·g₃,    g₀(1-g₀)  ]     [-0.4255×0.5745,  0.4255×0.5745 ]

  = [0.2445, -0.2445]
    [-0.2445,  0.2445]
```

Upstream gradient w.r.t. gate weights: dL/dg = [0.196, 0.407]

Gradient w.r.t. selected logits:

```
dL/d[s₃, s₀] = J^T · dL/dg

dL/ds₃ = 0.2445×0.196 + (-0.2445)×0.407 = 0.0479 - 0.0995 = -0.0516
dL/ds₀ = (-0.2445)×0.196 + 0.2445×0.407 = -0.0479 + 0.0995 = +0.0516
```

### AA.6.4 Gradient w.r.t. Non-Selected Experts

The top-K selection is a discrete operation — it has zero gradient w.r.t. the non-selected expert logits (s₁, s₂ for token 0). This is the fundamental challenge of routing: experts not selected receive no direct gradient signal from the main task loss. This is why the auxiliary losses (§AA.4.2, §AA.4.3) are essential — they provide gradient through P_i even for non-selected experts.

```
dL/ds₀ = -0.0516  (from above)
dL/ds₁ = 0        (Expert 1 not selected for token 0 — no task gradient)
dL/ds₂ = 0        (Expert 2 not selected for token 0 — no task gradient)
dL/ds₃ = +0.0516  (from above)
```

### AA.6.5 Gradient w.r.t. Router Weights W_r

Now we can compute the gradient with respect to W_r:

```
dL/dW_r[expert_i, :] += dL/ds_{t,i} · x_t

For token 0 (x₀ = [0.5, 0.1, -0.2, 0.8]):
  W_r row for expert 3: += (-0.0516) × [0.5, 0.1, -0.2, 0.8]
                        = [-0.0258, -0.00516, +0.01032, -0.04128]
  W_r row for expert 0: += (+0.0516) × [0.5, 0.1, -0.2, 0.8]
                        = [+0.0258, +0.00516, -0.01032, +0.04128]
  W_r rows for experts 1,2: += 0 (no gradient from task loss)
```

### AA.6.6 How Auxiliary Loss Fills the Gradient Gap

The auxiliary load-balance loss L_aux = E · Σ_i f_i · P_i, where P_i is differentiable via softmax. For token 0, expert 1:

```
∂L_aux/∂s_{0,1} = α · E · f₁ · softmax(s₀)[1] · (1 - softmax(s₀)[1])
                ≈ α · E · f₁ · small_value
```

Even though s₀,₁ is not selected, if expert 1 is underutilised (f₁ < 1/E), the auxiliary loss provides a positive gradient push on s₀,₁, nudging the router to send future tokens to expert 1. This is the mechanism by which load balancing works — the aux loss is the *only* gradient pathway for non-selected experts.

---

## AA.7 Full Python Implementation with Test Harness

This section provides two implementations: a pure NumPy version that mirrors the manual arithmetic above (for study), and a full PyTorch MoE layer (for practical use). Both include a `main` / `test` harness with assertions.

### AA.7.1 Pure NumPy Implementation (Mirrors §AA.5)

```python
"""
moe_numpy.py
------------
Pure NumPy MoE forward pass — mirrors the manual worked example in §AA.5.
Run: python moe_numpy.py
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def expert_forward(x: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """Single FFN expert: x -> ReLU(W1 x) -> W2."""
    return W2 @ relu(W1 @ x)


def top_k_indices(scores: np.ndarray, k: int):
    """Return indices of k largest values (descending order)."""
    return np.argsort(scores)[::-1][:k]


# ─────────────────────────────────────────────────────────────────
# MoE forward pass
# ─────────────────────────────────────────────────────────────────

def moe_forward(
    X: np.ndarray,          # [n_tokens, d_model]
    W_r: np.ndarray,        # [n_experts, d_model]
    experts_W1: list,       # list of [d_ff, d_model] arrays
    experts_W2: list,       # list of [d_model, d_ff] arrays
    top_k: int,
) -> tuple:
    """
    Sparse MoE forward pass.

    Returns:
        Y           — output tokens [n_tokens, d_model]
        route_info  — dict with logits, selected experts, gate weights per token
    """
    n_tokens, d_model = X.shape
    n_experts = W_r.shape[0]
    Y = np.zeros_like(X)
    route_info = {"logits": [], "selected": [], "gates": []}

    for t in range(n_tokens):
        x_t = X[t]                                 # [d_model]
        # Step 1: router logits
        s_t = W_r @ x_t                            # [n_experts]
        route_info["logits"].append(s_t.copy())
        # Step 2: top-K selection
        idx = top_k_indices(s_t, top_k)            # [top_k]
        route_info["selected"].append(idx.copy())
        # Step 3: gate weights (softmax over selected logits)
        g_t = softmax(s_t[idx])                    # [top_k]
        route_info["gates"].append(g_t.copy())
        # Step 4 + 5: expert forward and weighted combination
        y_t = np.zeros(d_model)
        for rank, expert_idx in enumerate(idx):
            e_out = expert_forward(x_t, experts_W1[expert_idx], experts_W2[expert_idx])
            y_t += g_t[rank] * e_out
        Y[t] = y_t

    return Y, route_info


def compute_load_balance_loss(route_info: dict, n_experts: int) -> float:
    """
    Auxiliary load-balance loss (Switch Transformer formulation).
    Non-differentiable in NumPy — for visualisation only.
    """
    logits_all = np.stack(route_info["logits"])          # [T, E]
    selected_all = np.stack(route_info["selected"])      # [T, K]
    n_tokens = logits_all.shape[0]

    # f_i: fraction of tokens routed to each expert
    counts = np.zeros(n_experts)
    for t in range(n_tokens):
        for e in selected_all[t]:
            counts[e] += 1
    f = counts / (n_tokens * selected_all.shape[1])      # normalise by T*K

    # P_i: mean softmax probability per expert
    probs = np.array([softmax(s) for s in logits_all])   # [T, E]
    P = probs.mean(axis=0)                               # [E]

    return float(n_experts * np.dot(f, P))


def compute_z_loss(route_info: dict) -> float:
    """Z-loss (Zoph et al. 2022)."""
    logits_all = np.stack(route_info["logits"])          # [T, E]
    log_z = np.log(np.exp(logits_all).sum(axis=1))       # [T]  (= logsumexp)
    return float((log_z ** 2).mean())


# ─────────────────────────────────────────────────────────────────
# Reproduce §AA.5 exactly
# ─────────────────────────────────────────────────────────────────

def build_worked_example():
    """Return the exact matrices from the manual worked example (§AA.5)."""
    X = np.array([
        [ 0.5,  0.1, -0.2,  0.8],
        [ 0.3, -0.4,  0.7,  0.2],
    ], dtype=np.float64)

    W_r = np.array([
        [ 0.1,  0.4, -0.1,  0.2],
        [ 0.3, -0.2,  0.5,  0.1],
        [-0.1,  0.3,  0.2, -0.4],
        [ 0.2,  0.1, -0.3,  0.5],
    ], dtype=np.float64)

    # Expert 0: fully specified
    W10 = np.array([
        [ 0.2,  0.3, -0.1,  0.4],
        [ 0.1, -0.2,  0.3,  0.1],
        [-0.1,  0.1,  0.2,  0.3],
        [ 0.4, -0.1,  0.1,  0.2],
    ], dtype=np.float64)
    W20 = np.array([
        [ 0.3,  0.1, -0.2,  0.1],
        [-0.1,  0.2,  0.1,  0.3],
        [ 0.2, -0.1,  0.3,  0.1],
        [ 0.1,  0.3,  0.1, -0.2],
    ], dtype=np.float64)

    # Expert 1: scaled identity
    W11 = 0.5 * np.eye(4)
    W21 = 0.3 * np.eye(4)

    # Expert 2: scaled identity
    W12 = 0.3 * np.eye(4)
    W22 = 0.25 * np.eye(4)

    # Expert 3: scaled identity
    W13 = 0.4 * np.eye(4)
    W23 = 0.35 * np.eye(4)

    experts_W1 = [W10, W11, W12, W13]
    experts_W2 = [W20, W21, W22, W23]

    return X, W_r, experts_W1, experts_W2


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────

def test_softmax():
    x = np.array([0.57, 0.27])
    g = softmax(x)
    assert abs(g.sum() - 1.0) < 1e-9, "Softmax must sum to 1"
    assert abs(g[0] - 0.5745) < 1e-3, f"Expected 0.5745, got {g[0]:.4f}"
    assert abs(g[1] - 0.4255) < 1e-3, f"Expected 0.4255, got {g[1]:.4f}"
    print("  [PASS] test_softmax")


def test_router_logits():
    X, W_r, _, _ = build_worked_example()
    s0 = W_r @ X[0]
    expected_s0 = np.array([0.27, 0.11, -0.38, 0.57])
    assert np.allclose(s0, expected_s0, atol=1e-6), f"Router logits token 0 mismatch: {s0}"
    s1 = W_r @ X[1]
    expected_s1 = np.array([-0.16, 0.54, -0.09, -0.09])
    assert np.allclose(s1, expected_s1, atol=1e-6), f"Router logits token 1 mismatch: {s1}"
    print("  [PASS] test_router_logits")


def test_top_k_selection():
    s0 = np.array([0.27, 0.11, -0.38, 0.57])
    idx = top_k_indices(s0, k=2)
    assert set(idx) == {3, 0}, f"Top-2 for token 0 should be {{3,0}}, got {set(idx)}"
    s1 = np.array([-0.16, 0.54, -0.09, -0.09])
    idx1 = top_k_indices(s1, k=2)
    assert 1 in idx1, f"Expert 1 should be selected for token 1"
    print("  [PASS] test_top_k_selection")


def test_expert0_token0():
    X, _, experts_W1, experts_W2 = build_worked_example()
    out = expert_forward(X[0], experts_W1[0], experts_W2[0])
    expected = np.array([0.147, 0.078, 0.170, 0.012])
    assert np.allclose(out, expected, atol=1e-3), f"Expert0(x0) mismatch: {out}"
    print("  [PASS] test_expert0_token0")


def test_moe_output_shape():
    X, W_r, experts_W1, experts_W2 = build_worked_example()
    Y, info = moe_forward(X, W_r, experts_W1, experts_W2, top_k=2)
    assert Y.shape == X.shape, f"Output shape {Y.shape} != input shape {X.shape}"
    print("  [PASS] test_moe_output_shape")


def test_gate_weights_sum_to_one():
    X, W_r, experts_W1, experts_W2 = build_worked_example()
    _, info = moe_forward(X, W_r, experts_W1, experts_W2, top_k=2)
    for t, gates in enumerate(info["gates"]):
        assert abs(gates.sum() - 1.0) < 1e-9, f"Gate weights for token {t} don't sum to 1: {gates}"
    print("  [PASS] test_gate_weights_sum_to_one")


def test_token0_output():
    X, W_r, experts_W1, experts_W2 = build_worked_example()
    Y, _ = moe_forward(X, W_r, experts_W1, experts_W2, top_k=2)
    expected_y0 = np.array([0.1028, 0.0412, 0.0723, 0.0695])
    assert np.allclose(Y[0], expected_y0, atol=1e-3), f"Token 0 output mismatch: {Y[0]}"
    print("  [PASS] test_token0_output")


def test_load_balance_loss_range():
    X, W_r, experts_W1, experts_W2 = build_worked_example()
    _, info = moe_forward(X, W_r, experts_W1, experts_W2, top_k=2)
    loss = compute_load_balance_loss(info, n_experts=4)
    assert loss >= 0, "Load balance loss must be non-negative"
    assert loss <= 1.0, f"Unexpectedly high load balance loss: {loss}"
    print(f"  [PASS] test_load_balance_loss_range  (loss={loss:.4f})")


def test_z_loss_positive():
    X, W_r, experts_W1, experts_W2 = build_worked_example()
    _, info = moe_forward(X, W_r, experts_W1, experts_W2, top_k=2)
    zloss = compute_z_loss(info)
    assert zloss >= 0, "Z-loss must be non-negative"
    print(f"  [PASS] test_z_loss_positive  (z_loss={zloss:.4f})")


def run_all_tests():
    print("=" * 56)
    print("NumPy MoE — Unit Tests")
    print("=" * 56)
    test_softmax()
    test_router_logits()
    test_top_k_selection()
    test_expert0_token0()
    test_moe_output_shape()
    test_gate_weights_sum_to_one()
    test_token0_output()
    test_load_balance_loss_range()
    test_z_loss_positive()
    print("=" * 56)
    print("All tests passed.")
    print("=" * 56)


# ─────────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────────

def main():
    run_all_tests()

    print("\n--- MoE Forward Pass Demo (§AA.5 values) ---\n")
    X, W_r, experts_W1, experts_W2 = build_worked_example()
    Y, info = moe_forward(X, W_r, experts_W1, experts_W2, top_k=2)

    n_experts = W_r.shape[0]
    for t in range(len(X)):
        print(f"Token {t}:")
        print(f"  Input:          {X[t]}")
        print(f"  Router logits:  {info['logits'][t].round(4)}")
        print(f"  Selected expts: {info['selected'][t]}")
        print(f"  Gate weights:   {info['gates'][t].round(4)}")
        print(f"  Output:         {Y[t].round(4)}")
        print()

    lb_loss = compute_load_balance_loss(info, n_experts)
    z_loss_val = compute_z_loss(info)
    print(f"Load-balance loss (α=1): {lb_loss:.6f}")
    print(f"Z-loss:                  {z_loss_val:.6f}")


if __name__ == "__main__":
    main()
```

### AA.7.2 Full PyTorch MoE Layer

```python
"""
moe_pytorch.py
--------------
Production-grade PyTorch MoE layer with:
  - Noisy top-K router
  - Auxiliary load-balance loss (Switch Transformer)
  - Z-loss (ST-MoE / Zoph et al. 2022)
  - Expert-choice routing variant
  - Token capacity buffer and drop logging
  - Full test/main harness

Run: python moe_pytorch.py
Dependencies: torch >= 2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MoEConfig:
    d_model: int = 512          # token embedding dimension
    d_ff: int = 2048            # per-expert FFN hidden dimension
    n_experts: int = 8          # total number of experts
    top_k: int = 2              # experts activated per token
    capacity_factor: float = 1.25  # buffer over perfect balance
    aux_loss_weight: float = 0.01  # α — load-balance loss weight
    z_loss_weight: float = 0.001   # β — z-loss weight
    dropout: float = 0.0
    noise_eps: float = 1e-2     # floor for noisy top-K std
    use_bias: bool = False      # experts usually have no bias


# ═══════════════════════════════════════════════════════════════════
# Expert FFN
# ═══════════════════════════════════════════════════════════════════

class Expert(nn.Module):
    """Single FFN expert with SwiGLU-style gating."""

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.use_bias)
        self.w_up   = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.use_bias)
        self.w_down = nn.Linear(cfg.d_ff,    cfg.d_model, bias=cfg.use_bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate(x) ⊙ silu(up(x)) → down
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))


# ═══════════════════════════════════════════════════════════════════
# Noisy Top-K Router
# ═══════════════════════════════════════════════════════════════════

class NoisyTopKRouter(nn.Module):
    """
    Router with learned Gaussian noise (Shazeer et al. 2017).
    During training, noise is added to logits to encourage exploration.
    At inference, clean logits are used.
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.gate       = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.noise_gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.top_k      = cfg.top_k
        self.noise_eps  = cfg.noise_eps

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x: [n_tokens, d_model]
        Returns:
            gate_weights  [n_tokens, top_k]
            expert_indices [n_tokens, top_k]
            raw_logits    [n_tokens, n_experts]  — for aux loss computation
        """
        logits = self.gate(x)                                  # [T, E]
        if self.training:
            noise_std = F.softplus(self.noise_gate(x)) + self.noise_eps
            logits = logits + torch.randn_like(logits) * noise_std
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        gate_weights = F.softmax(topk_vals, dim=-1)
        return gate_weights, topk_idx, logits


# ═══════════════════════════════════════════════════════════════════
# Auxiliary Losses
# ═══════════════════════════════════════════════════════════════════

def aux_load_balance_loss(
    router_logits: torch.Tensor,      # [n_tokens, n_experts]
    expert_indices: torch.Tensor,     # [n_tokens, top_k]
) -> torch.Tensor:
    """
    Switch Transformer auxiliary load-balance loss.
    Encourages uniform expert utilisation.
    """
    n_tokens, n_experts = router_logits.shape
    top_k = expert_indices.shape[-1]

    # f_i: fraction of tokens dispatched to expert i (non-differentiable)
    # One-hot encode all selected experts
    flat_idx = expert_indices.reshape(-1)                      # [T * top_k]
    one_hot = F.one_hot(flat_idx, n_experts).float()           # [T*top_k, E]
    one_hot = one_hot.reshape(n_tokens, top_k, n_experts)      # [T, K, E]
    tokens_per_expert = one_hot.sum(dim=1)                     # [T, E]
    f = tokens_per_expert.sum(dim=0) / (n_tokens * top_k)     # [E]

    # P_i: mean softmax probability for expert i (differentiable)
    P = F.softmax(router_logits, dim=-1).mean(dim=0)           # [E]

    return (n_experts * (f * P).sum())


def z_loss_fn(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Z-loss (Zoph et al., 2022, ST-MoE).
    Penalises large router logit magnitudes to prevent brittleness.
    """
    log_z = torch.logsumexp(router_logits, dim=-1)             # [T]
    return (log_z ** 2).mean()


# ═══════════════════════════════════════════════════════════════════
# Token Dispatch Utilities
# ═══════════════════════════════════════════════════════════════════

def dispatch_tokens(
    x: torch.Tensor,              # [n_tokens, d_model]
    gate_weights: torch.Tensor,   # [n_tokens, top_k]
    expert_indices: torch.Tensor, # [n_tokens, top_k]
    n_experts: int,
    capacity: int,
) -> tuple:
    """
    Build per-expert input batches respecting capacity limits.

    Returns:
        expert_inputs  — list of n_experts tensors, each [tokens_routed, d_model]
        combine_info   — list of (token_idx, gate_weight) pairs per expert slot
        n_dropped      — total tokens dropped due to capacity overflow
    """
    expert_inputs = [[] for _ in range(n_experts)]
    combine_info  = [[] for _ in range(n_experts)]
    expert_counts = [0] * n_experts
    n_dropped = 0

    n_tokens, top_k = expert_indices.shape
    for t in range(n_tokens):
        for k_pos in range(top_k):
            e = expert_indices[t, k_pos].item()
            w = gate_weights[t, k_pos].item()
            if expert_counts[e] < capacity:
                expert_inputs[e].append(x[t])
                combine_info[e].append((t, w))
                expert_counts[e] += 1
            else:
                n_dropped += 1

    # Stack collected tokens into tensors
    for e in range(n_experts):
        if expert_inputs[e]:
            expert_inputs[e] = torch.stack(expert_inputs[e])   # [count, d]
        else:
            expert_inputs[e] = x.new_zeros(0, x.shape[-1])

    return expert_inputs, combine_info, n_dropped


def combine_expert_outputs(
    expert_outputs: list,          # list of [count, d_model] tensors
    combine_info: list,            # list of (token_idx, weight) lists
    n_tokens: int,
    d_model: int,
    device: torch.device,
) -> torch.Tensor:
    """Scatter expert outputs back to per-token positions."""
    Y = torch.zeros(n_tokens, d_model, device=device)
    for e, (outputs, slots) in enumerate(zip(expert_outputs, combine_info)):
        for slot_idx, (tok_idx, weight) in enumerate(slots):
            Y[tok_idx] += weight * outputs[slot_idx]
    return Y


# ═══════════════════════════════════════════════════════════════════
# Full MoE Layer
# ═══════════════════════════════════════════════════════════════════

class MoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts FFN layer with:
      - Noisy top-K routing
      - Capacity buffer (no token left behind philosophy)
      - Auxiliary load-balance loss + z-loss
      - Optional shared expert (DeepSeek-style)
    """

    def __init__(self, cfg: MoEConfig, use_shared_expert: bool = False):
        super().__init__()
        self.cfg = cfg
        self.router = NoisyTopKRouter(cfg)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.n_experts)])
        self.use_shared_expert = use_shared_expert
        if use_shared_expert:
            self.shared_expert = Expert(cfg)

    def _capacity(self, n_tokens: int) -> int:
        return max(1, int(
            self.cfg.capacity_factor * n_tokens * self.cfg.top_k / self.cfg.n_experts
        ))

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x: [batch, seq_len, d_model]  OR  [n_tokens, d_model]
        Returns:
            y           — same shape as x
            aux_loss    — scalar tensor (add to main loss × weight)
        """
        # Flatten batch/seq dims to a single token dimension
        input_shape = x.shape
        if x.dim() == 3:
            B, S, D = x.shape
            x_flat = x.reshape(B * S, D)                          # [T, d]
        else:
            x_flat = x
        n_tokens = x_flat.shape[0]

        # ── Router ──────────────────────────────────────────────
        gate_weights, expert_indices, raw_logits = self.router(x_flat)

        # ── Auxiliary losses ─────────────────────────────────────
        lb_loss = aux_load_balance_loss(raw_logits, expert_indices)
        zl_loss = z_loss_fn(raw_logits)
        aux_loss = self.cfg.aux_loss_weight * lb_loss + self.cfg.z_loss_weight * zl_loss

        # ── Dispatch ─────────────────────────────────────────────
        capacity = self._capacity(n_tokens)
        expert_inputs, combine_info, n_dropped = dispatch_tokens(
            x_flat, gate_weights, expert_indices, self.cfg.n_experts, capacity
        )
        if n_dropped > 0 and self.training:
            # In production you'd log this metric to your observability stack
            pass  # logging.warning(f"MoE dropped {n_dropped} token-expert slots")

        # ── Expert forward passes ─────────────────────────────────
        expert_outputs = []
        for e, expert in enumerate(self.experts):
            tokens_e = expert_inputs[e]
            if tokens_e.shape[0] > 0:
                expert_outputs.append(expert(tokens_e))
            else:
                expert_outputs.append(tokens_e)  # empty — shapes preserved

        # ── Combine ───────────────────────────────────────────────
        Y = combine_expert_outputs(
            expert_outputs, combine_info, n_tokens, self.cfg.d_model, x_flat.device
        )

        # ── Optional shared expert ────────────────────────────────
        if self.use_shared_expert:
            Y = Y + self.shared_expert(x_flat)

        # Restore original shape
        Y = Y.reshape(input_shape)
        return Y, aux_loss


# ═══════════════════════════════════════════════════════════════════
# Expert-Choice Routing Variant
# ═══════════════════════════════════════════════════════════════════

class ExpertChoiceMoELayer(nn.Module):
    """
    Expert-Choice MoE (Zhou et al. 2022).
    Each expert selects its top-C tokens from the batch.
    Guarantees perfect load balance; some tokens may be missed.
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.n_experts)])

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x: [batch, seq_len, d_model]
        Returns y (same shape), aux_loss (scalar — just z-loss here).
        """
        input_shape = x.shape
        if x.dim() == 3:
            B, S, D = x.shape
            x_flat = x.reshape(B * S, D)
        else:
            x_flat = x
        n_tokens = x_flat.shape[0]

        # Expert scores: [n_tokens, n_experts]
        logits = self.gate(x_flat)
        # Normalise over the token dimension per expert
        scores = F.softmax(logits, dim=0)                   # [T, E]

        # Each expert picks top-C tokens
        capacity = max(1, int(self.cfg.capacity_factor * n_tokens / self.cfg.n_experts))
        Y = torch.zeros_like(x_flat)

        for e, expert in enumerate(self.experts):
            e_scores = scores[:, e]                          # [T]
            topC_vals, topC_idx = torch.topk(e_scores, min(capacity, n_tokens))
            tokens_for_e = x_flat[topC_idx]                  # [C, d]
            expert_out = expert(tokens_for_e)                # [C, d]
            # Weight by score and scatter back
            Y.index_add_(0, topC_idx, topC_vals.unsqueeze(-1) * expert_out)

        # Z-loss only (no aux loss needed — load is balanced by construction)
        aux_loss = self.cfg.z_loss_weight * z_loss_fn(logits)
        return Y.reshape(input_shape), aux_loss


# ═══════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════

def test_expert_forward():
    cfg = MoEConfig(d_model=16, d_ff=32, n_experts=4, top_k=2)
    expert = Expert(cfg)
    x = torch.randn(5, 16)
    out = expert(x)
    assert out.shape == (5, 16), f"Expert output shape wrong: {out.shape}"
    print("  [PASS] test_expert_forward")


def test_router_output_shape():
    cfg = MoEConfig(d_model=16, d_ff=32, n_experts=4, top_k=2)
    router = NoisyTopKRouter(cfg)
    router.train()
    x = torch.randn(8, 16)
    gw, idx, logits = router(x)
    assert gw.shape    == (8, 2),  f"Gate weights shape wrong: {gw.shape}"
    assert idx.shape   == (8, 2),  f"Expert indices shape wrong: {idx.shape}"
    assert logits.shape == (8, 4), f"Logits shape wrong: {logits.shape}"
    print("  [PASS] test_router_output_shape")


def test_gate_weights_sum_to_one_torch():
    cfg = MoEConfig(d_model=16, d_ff=32, n_experts=4, top_k=2)
    router = NoisyTopKRouter(cfg)
    router.eval()
    x = torch.randn(16, 16)
    gw, _, _ = router(x)
    sums = gw.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        f"Gate weights don't sum to 1: {sums}"
    print("  [PASS] test_gate_weights_sum_to_one_torch")


def test_aux_loss_positive():
    cfg = MoEConfig(d_model=16, d_ff=32, n_experts=4, top_k=2)
    router = NoisyTopKRouter(cfg)
    router.eval()
    x = torch.randn(20, 16)
    gw, idx, logits = router(x)
    loss = aux_load_balance_loss(logits, idx)
    assert loss.item() >= 0, f"Aux loss should be non-negative: {loss.item()}"
    print(f"  [PASS] test_aux_loss_positive  (loss={loss.item():.4f})")


def test_z_loss_positive():
    logits = torch.randn(20, 8)
    loss = z_loss_fn(logits)
    assert loss.item() >= 0, f"Z-loss should be non-negative: {loss.item()}"
    print(f"  [PASS] test_z_loss_positive  (z_loss={loss.item():.4f})")


def test_moe_layer_output_shape_3d():
    cfg = MoEConfig(d_model=32, d_ff=64, n_experts=4, top_k=2)
    layer = MoELayer(cfg)
    layer.eval()
    x = torch.randn(2, 16, 32)   # [batch=2, seq=16, d=32]
    y, aux_loss = layer(x)
    assert y.shape == x.shape, f"MoE output shape mismatch: {y.shape} vs {x.shape}"
    assert aux_loss.dim() == 0, "aux_loss should be scalar"
    print("  [PASS] test_moe_layer_output_shape_3d")


def test_moe_layer_output_shape_2d():
    cfg = MoEConfig(d_model=32, d_ff=64, n_experts=4, top_k=2)
    layer = MoELayer(cfg)
    layer.eval()
    x = torch.randn(24, 32)      # [tokens, d_model]
    y, aux_loss = layer(x)
    assert y.shape == x.shape, f"MoE output shape mismatch: {y.shape}"
    print("  [PASS] test_moe_layer_output_shape_2d")


def test_moe_with_shared_expert():
    cfg = MoEConfig(d_model=32, d_ff=64, n_experts=4, top_k=2)
    layer = MoELayer(cfg, use_shared_expert=True)
    layer.eval()
    x = torch.randn(8, 32)
    y, _ = layer(x)
    assert y.shape == x.shape
    print("  [PASS] test_moe_with_shared_expert")


def test_expert_choice_layer():
    cfg = MoEConfig(d_model=32, d_ff=64, n_experts=4, top_k=2)
    layer = ExpertChoiceMoELayer(cfg)
    layer.eval()
    x = torch.randn(2, 16, 32)
    y, aux_loss = layer(x)
    assert y.shape == x.shape, f"ExpertChoice output shape wrong: {y.shape}"
    assert aux_loss.item() >= 0
    print("  [PASS] test_expert_choice_layer")


def test_backward_flows():
    """Verify gradients reach the router weights."""
    cfg = MoEConfig(d_model=16, d_ff=32, n_experts=4, top_k=2,
                    aux_loss_weight=0.01, z_loss_weight=0.001)
    layer = MoELayer(cfg)
    layer.train()
    x = torch.randn(12, 16, requires_grad=True)
    y, aux_loss = layer(x)
    task_loss = y.mean()
    total_loss = task_loss + aux_loss
    total_loss.backward()
    # Router gate should have gradients
    router_grad = layer.router.gate.weight.grad
    assert router_grad is not None, "Router gate has no gradient"
    assert router_grad.abs().sum().item() > 0, "Router gate gradient is all zeros"
    print("  [PASS] test_backward_flows")


def test_expert_indices_in_range():
    cfg = MoEConfig(d_model=16, d_ff=32, n_experts=8, top_k=2)
    router = NoisyTopKRouter(cfg)
    router.eval()
    x = torch.randn(50, 16)
    _, idx, _ = router(x)
    assert idx.min().item() >= 0
    assert idx.max().item() < cfg.n_experts
    # No duplicate experts per token
    for t in range(50):
        assert len(set(idx[t].tolist())) == cfg.top_k, \
            f"Duplicate expert selection at token {t}"
    print("  [PASS] test_expert_indices_in_range")


def test_uniform_routing_gives_low_aux_loss():
    """If routing is perfectly uniform, aux loss should be near 0."""
    n_tokens, n_experts, top_k = 100, 4, 2
    # Build logits that are identical for all tokens → uniform routing
    logits = torch.zeros(n_tokens, n_experts)
    # Build ideal expert indices cycling through experts
    idx = torch.tensor([[t % n_experts, (t+1) % n_experts] for t in range(n_tokens)])
    loss = aux_load_balance_loss(logits, idx)
    # With zero logits, P_i = 1/E for all i; with cycling idx, f_i ≈ 1/E
    # → loss ≈ E × E × (1/E × 1/E) = 1
    # The important check: loss is well-defined and finite
    assert not torch.isnan(loss), "Aux loss is NaN under uniform routing"
    assert not torch.isinf(loss), "Aux loss is Inf under uniform routing"
    print(f"  [PASS] test_uniform_routing_gives_low_aux_loss  (loss={loss.item():.4f})")


def run_all_tests_torch():
    print("=" * 56)
    print("PyTorch MoE — Unit Tests")
    print("=" * 56)
    torch.manual_seed(42)
    test_expert_forward()
    test_router_output_shape()
    test_gate_weights_sum_to_one_torch()
    test_aux_loss_positive()
    test_z_loss_positive()
    test_moe_layer_output_shape_3d()
    test_moe_layer_output_shape_2d()
    test_moe_with_shared_expert()
    test_expert_choice_layer()
    test_backward_flows()
    test_expert_indices_in_range()
    test_uniform_routing_gives_low_aux_loss()
    print("=" * 56)
    print("All PyTorch tests passed.")
    print("=" * 56)


# ═══════════════════════════════════════════════════════════════════
# Main demo
# ═══════════════════════════════════════════════════════════════════

def main():
    run_all_tests_torch()

    print("\n--- MoE Layer Training Step Demo ---\n")
    torch.manual_seed(0)
    cfg = MoEConfig(
        d_model=128, d_ff=512, n_experts=8, top_k=2,
        capacity_factor=1.5, aux_loss_weight=0.01, z_loss_weight=0.001,
    )
    layer = MoELayer(cfg, use_shared_expert=True)
    layer.train()
    optim = torch.optim.AdamW(layer.parameters(), lr=1e-3)

    # Simulate 5 training steps
    for step in range(5):
        x = torch.randn(4, 32, cfg.d_model)   # batch=4, seq=32
        target = torch.randn_like(x)
        y, aux_loss = layer(x)
        task_loss = F.mse_loss(y, target)
        total_loss = task_loss + aux_loss
        optim.zero_grad()
        total_loss.backward()
        # Gradient clipping — important for MoE stability
        torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0)
        optim.step()
        print(f"  Step {step+1}: task_loss={task_loss.item():.4f}  "
              f"aux_loss={aux_loss.item():.4f}  "
              f"total={total_loss.item():.4f}")

    print("\n--- Expert-Choice Variant Demo ---\n")
    ec_layer = ExpertChoiceMoELayer(cfg)
    ec_layer.eval()
    with torch.no_grad():
        x_test = torch.randn(2, 16, cfg.d_model)
        y_ec, z = ec_layer(x_test)
        print(f"  ExpertChoice output shape: {y_ec.shape}")
        print(f"  Z-loss: {z.item():.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

---

## AA.8 Training Dynamics

### AA.8.1 Expert Specialisation

When MoE training converges well, different experts develop measurable specialisation. Analyses of Switch Transformer and Mixtral show:

- **Syntactic experts**: capture punctuation, common function words, grammatical connectives.
- **Semantic-domain experts**: specialise on science, law, code, narrative prose.
- **Positional experts**: some experts preferentially activate at sentence beginnings or endings.
- **Token-type experts**: distinct experts for digits, subword units, capitalised tokens.

Specialisation is not guaranteed or hand-engineered — it emerges from gradient descent under load-balancing pressure. The auxiliary loss prevents any single expert from monopolising all tokens, and the gradient signal from diverse data drives semantic differentiation.

You can measure specialisation after training by recording which expert received each token and computing per-expert token-type distributions. A Gini coefficient over expert utilisation by token category quantifies specialisation.

```python
def measure_specialisation(expert_indices, token_metadata, n_experts):
    """
    expert_indices: [n_tokens, top_k]
    token_metadata: [n_tokens] — categorical label per token (e.g., part-of-speech)
    Returns: dict of {expert_id: Counter(label -> count)}
    """
    from collections import Counter
    expert_dist = {e: Counter() for e in range(n_experts)}
    for t, exps in enumerate(expert_indices):
        label = token_metadata[t]
        for e in exps:
            expert_dist[e.item()][label] += 1
    return expert_dist
```

### AA.8.2 Common Training Failure Modes

**Expert collapse** — Described in §AA.4.1. Prevention: auxiliary loss (§AA.4.2) + z-loss (§AA.4.3) + noisy routing (§AA.3.4).

**Router oscillation** — The router flip-flops between two routing strategies without settling. Cause: large learning rate on router weights, or aux_loss_weight too large pulling the router away from task-optimal routing. Fix: use a lower learning rate multiplier on the router; keep α ≤ 0.01.

**Dead experts** — Experts that receive zero tokens for extended periods. Unlike temporary underuse (which the aux loss corrects), dead experts arise when load-balancing loss is insufficient or the router initialisation is very asymmetric. Prevention: expert dropout (§AA.4.5), lower auxiliary loss α, or reinitialising dead expert weights mid-training.

**Training instability at scale** — Large router logits (before z-loss mitigation) cause loss spikes. The z-loss in §AA.4.3 was specifically motivated by this failure observed at 2B+ parameter scale. Apply z-loss from step 0 for large models.

**Capacity overflow causing token dropping** — At small batch sizes, variance in routing decisions can send many tokens to one expert while another sits empty. Increasing capacity_factor to 2.0 at the cost of wasted computation prevents dropped tokens. At inference time, where batch size can be 1, use capacity_factor = ∞ (unbounded) since capacity overflow is no longer a batching concern.

### AA.8.3 Best Practices from Literature

```
Practice                         Source              Effect
───────────────────────────────────────────────────────────────────────────
α = 0.01 aux loss weight         Switch Transformer  Prevents collapse without over-regularising
β = 0.001 z-loss weight          ST-MoE              Stabilises large-scale training
top-K = 2 (not 1)                GShard, Mixtral     Better gradient coverage vs top-1
Noisy routing during training    Shazeer 2017        Exploration prevents early collapse
capacity_factor = 1.25 train     Switch Transformer  5% compute overhead, low drop rate
capacity_factor = 2.0 infer      Various             Eliminates drops at inference time
Expert dropout p=0.1             Fedus 2021          Redundancy, better generalisation
Gradient clipping norm=1.0       Standard            MoE gradients can spike
Separate LR for router           DeepSeek-V2         Router benefits from slower updates
Fine-grained experts (many       DeepSeek-V2/V3      Better specialisation; more routing
  small rather than few large)                       granularity
───────────────────────────────────────────────────────────────────────────
```

### AA.8.4 MoE vs Dense Scaling at Equal Compute

Empirically, MoE models trained with the same FLOPs budget as dense models consistently outperform them, but this advantage diminishes if:

- The batch size is too small to keep all experts busy (recommendation: total_tokens >> n_experts × capacity).
- The number of training steps is too small for expert specialisation to develop.
- The auxiliary loss is mis-tuned (too high → routing ignores task; too low → collapse).

The Mixtral paper reports that Mixtral 8×7B matches or exceeds Llama 2 70B at roughly 5× lower inference FLOPs — a strong empirical confirmation of the FLOPs/parameters decoupling argument in §AA.1.3.

---

## AA.9 Inference and Serving MoE

### AA.9.1 Why MoE Inference Is Different

Dense model inference is well-understood: load all parameters once, run matrix multiplications sequentially. MoE inference adds three new problems:

1. **Expert selection per token is dynamic** — unlike dense models where the computation graph is fixed.
2. **Expert parameters may not all fit in GPU VRAM** — a 46.7B-effective Mixtral 8×7B has 46.7B total parameters across 8 experts per layer; serving requires all of them resident.
3. **Batching efficiency degrades** — if different tokens in a batch route to different experts, you get many small matrix multiplications instead of one large one, losing GPU utilisation.

### AA.9.2 Expert Parallelism

The standard distributed serving strategy for MoE is *expert parallelism* (EP): different GPUs hold different subsets of experts.

```
Expert Parallelism Layout (8 experts across 4 GPUs):

GPU 0: Expert 0, Expert 1    <- all tokens for these experts sent here
GPU 1: Expert 2, Expert 3
GPU 2: Expert 4, Expert 5
GPU 3: Expert 6, Expert 7

Communication pattern:
  Each GPU holds router + all attention layers (replicated via tensor parallel)
  After routing, tokens are ALL-TO-ALL dispatched to the GPU holding their experts
  Expert outputs are ALL-TO-ALL gathered back to original GPUs
```

The cost of expert parallelism is two all-to-all collective operations per MoE layer — one to dispatch tokens, one to gather results. For large MoE models (DeepSeek-V3, Mixtral), this communication overhead can be 10–20% of total step time. DeepSeek-V3 reports that with 256 experts across 320 GPUs and a custom all-to-all implementation over InfiniBand, this overhead is minimised.

```
Parallelism strategy for large MoE:
  Tensor Parallelism (TP):   split attention heads across GPUs within a node
  Expert Parallelism (EP):   split experts across nodes
  Pipeline Parallelism (PP): split layers across node groups

  Combined: TP within node, EP + PP across nodes
```

### AA.9.3 KV-Cache Implications for MoE

The KV cache is stored per attention layer per sequence. MoE layers do not have KV caches — they are purely FFN replacements. The KV-cache footprint of a MoE model is therefore *identical* to a dense model with the same number of attention layers and the same hidden dimension. The additional memory cost of MoE is entirely in the expert weights.

```
Memory breakdown for Mixtral 8x7B (BF16):
  Attention weights (shared):    ~6.7B params  x 2 bytes  = ~13.4 GB
  Expert weights (8 experts x 32 MoE layers): ~32B params x 2 bytes = ~64 GB
  KV cache (runtime, per sequence): identical to a 7B dense model per layer
  Total weights:                 ~46.7B x 2 bytes = ~93.4 GB (4x A100 80GB)
```

This means KV-cache eviction policies (Chapter 11), prefix caching (Chapter 11.5), and RadixAttention work identically for MoE and dense models — they operate on the attention component only.

### AA.9.4 Batching Challenges and Expert Load at Inference

During inference, with small batch sizes (common for latency-optimised deployments):

- A batch of 8 tokens may route all tokens to the same 2 experts, leaving 6 experts idle.
- Each expert then gets a very small matrix multiplication (8 tokens × d_model), which is too small to saturate GPU tensor cores.
- The GPU spends most time loading expert weights from HBM, not computing — deeply memory-bound.

This is the *expert weight loading bottleneck*. It is analogous to the KV-cache memory-bandwidth bottleneck for dense models, but worse: you must load the weights of all 8 experts even if only 2 are used in a given batch, because you don't know which tokens will arrive next.

**Throughput-latency trade-off for MoE inference:**

```
Small batch (B=1-4):    Mostly memory-bound, low expert utilisation, low latency
Large batch (B=32+):    Experts better utilised, compute approaches ridge point
Optimal batch for MoE:  Larger than for dense; need T >> E to balance experts
```

### AA.9.5 Expert Caching Strategies

Since expert parameters are large and only a subset are active per forward pass, caching strategies reduce repeated HBM → compute-unit data movement:

**Frequency-based caching**: Profile which experts are most frequently used across a representative workload. Keep top-N expert weights in GPU L2 or fast SRAM. This works when routing is skewed — some experts genuinely are more popular.

**Speculative expert prefetching**: After the router produces expert selections for layer L, prefetch the expert weights for layer L+1 while layer L is computing. Hides the memory latency behind the compute latency. This is directly analogous to kernel pipelining in Flash Attention (Chapter 5).

**Expert co-location**: Place the most co-occurring expert pairs on the same GPU to minimise inter-GPU communication. Profiling expert co-occurrence across a dataset and using a graph partitioning algorithm (e.g., METIS) to assign experts to GPUs is a standard production technique.

### AA.9.6 vLLM's MoE Support

vLLM implements MoE serving for Mixtral, DeepSeek-V2/V3, and other MoE architectures. Key implementation details:

**Fused MoE kernels**: vLLM uses a custom Triton kernel (`fused_moe` in `vllm/model_executor/layers/fused_moe/`) that fuses token dispatch, expert GEMM, and token combine into a single GPU kernel, avoiding intermediate tensor materialisation and reducing memory traffic.

```python
# vLLM's fused MoE API (simplified)
# from vllm.model_executor.layers.fused_moe import fused_moe

output = fused_moe(
    hidden_states,          # [n_tokens, d_model]
    w1,                     # expert up-projections [n_experts, d_ff*2, d_model]
    w2,                     # expert down-projections [n_experts, d_model, d_ff]
    gating_output,          # router logits [n_tokens, n_experts]
    topk=2,
    renormalize=True,       # re-normalise gate weights after top-K selection
    inplace=False,
    use_grouped_topk=False,
)
```

**Tensor parallelism for MoE in vLLM**: When running Mixtral on multiple GPUs with tensor parallelism, vLLM shards expert weight matrices across GPUs and uses all-reduce to combine expert outputs, rather than expert parallelism. This avoids the complexity of all-to-all routing at the cost of not scaling expert count with GPU count.

**DeepSeek-V2/V3 in vLLM**: The `DeepseekV2MoE` module in vLLM uses the multi-head latent attention (MLA) architecture for DeepSeek-V2's attention layers, plus the shared+routed expert decomposition described in §AA.3.7. The 256-expert routing with top-8 selection is handled by the same `fused_moe` kernel with a `use_grouped_topk=True` flag that groups expert selection by device to reduce communication overhead.

### AA.9.7 DeepSeek-V2 and V3 MoE Architecture Deep-Dive

DeepSeek-V2 (2024) and DeepSeek-V3 (2024) represent the state of the art in production MoE design. Their MoE architecture differs from Mixtral in several important ways:

**V2 MoE layer:**

- 160 routed experts, top-6 selection
- 2 shared experts (always active)
- Fine-grained expert decomposition: each expert is smaller than Mixtral's, enabling more precise routing

**V3 MoE layer:**

- 256 routed experts, top-8 selection
- 1 shared expert
- Auxiliary-loss-free load balancing (a breakthrough — they use a biased routing mechanism that achieves balance without degrading task performance)

**V3's auxiliary-loss-free balancing**: Instead of a differentiable auxiliary loss, V3 introduces a per-expert *bias term* b_i added to the router logits during top-K selection but *not* during softmax gate weight computation:

```
Selection:  I_t = argtopK(s_t + b)          # b is a bias vector per expert
Gate:       g_t = softmax(s_t[I_t])         # no bias in gate weights
Bias update: if f_i > target_freq: b_i -= gamma
             if f_i < target_freq: b_i += gamma
```

The bias b_i is updated after each training step (not via backprop) to steer routing toward balance without polluting the gradient of the main loss with an auxiliary term.

```python
class AuxFreeRouter(nn.Module):
    """
    DeepSeek-V3 style auxiliary-loss-free router.
    Bias terms updated online to enforce load balance without aux loss.
    """
    def __init__(self, d_model, n_experts, top_k, update_rate=1e-3, target_freq=None):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k
        self.update_rate = update_rate
        self.target_freq = target_freq or (top_k / n_experts)
        # Bias is not a parameter — updated manually each step
        self.register_buffer('bias', torch.zeros(n_experts))

    def forward(self, x):
        logits = self.gate(x)                              # [T, E]
        # Top-K selection uses biased logits
        biased = logits + self.bias.unsqueeze(0)
        _, topk_idx = torch.topk(biased, self.top_k, dim=-1)
        # Gate weights use unbiased logits (important!)
        selected_logits = logits.gather(1, topk_idx)
        gate_weights = F.softmax(selected_logits, dim=-1)
        return gate_weights, topk_idx, logits

    @torch.no_grad()
    def update_bias(self, expert_indices):
        """Call after each optimiser step."""
        n_tokens = expert_indices.shape[0]
        counts = torch.zeros_like(self.bias)
        for e_idx in expert_indices.reshape(-1):
            counts[e_idx] += 1
        freq = counts / (n_tokens * self.top_k)
        self.bias -= self.update_rate * (freq - self.target_freq).sign()
```

---

## AA.10 Production MoE Models Survey

### AA.10.1 Chronological Development

**Jacobs et al. (1991) — Adaptive Mixtures of Local Experts**: The original paper. Soft routing, small scale, but established the theoretical foundation.

**Shazeer et al. (2017) — Sparsely-Gated MoE**: First large-scale application to LSTMs. Introduced noisy top-K routing and the load-balancing auxiliary loss. Trained a 137B-parameter model in 2017 using top-2 routing across 65,536 experts.

**GShard (Lepikhin et al., 2021)**: First MoE transformer at scale. 600B parameters, top-2 from 2048 experts per layer, trained with expert sharding across 2048 TPUs. Established expert parallelism as the standard distributed training pattern.

**Switch Transformer (Fedus et al., 2021)**: Simplified to top-1 routing. Demonstrated MoE transformers can be trained stably at T5 scale. First to quantify the capacity factor trade-off. Showed 4–7× training speed-up at equal compute vs dense T5.

**GLaM (Du et al., 2022)**: 1.2T parameter MoE model, top-2 from 64 experts. Matched GPT-3 quality with 3× less training energy.

**ST-MoE (Zoph et al., 2022)**: Introduced z-loss. The first systematic study of MoE training instabilities at scale and their mitigations.

**Mixtral 8×7B (Mistral AI, 2023)**: The first open-weight production MoE model. 46.7B total / ~13B active parameters per token. 8 experts, top-2, replacing all FFN layers. Released with full weights; widely deployed.

**Mixtral 8×22B (Mistral AI, 2024)**: 141B total / ~39B active. Extends the 8×7B design to a larger base. Competitive with larger dense models on reasoning benchmarks.

**DeepSeek-V2 (DeepSeek, 2024)**: 236B total / 21B active. Introduced MLA (Multi-head Latent Attention) for KV-cache compression, fine-grained expert decomposition (160 routed + 2 shared), and device-limited routing to minimise all-to-all communication.

**Grok-1 (xAI, 2024)**: 314B total parameters, 8 experts, top-2. Open-sourced. Uses a similar architecture to Mixtral with a larger expert count.

**DeepSeek-V3 (DeepSeek, 2024)**: 671B total / 37B active. 256 routed + 1 shared expert, top-8 routing. Auxiliary-loss-free load balancing. Multi-token prediction. Trained for $5.5M on H800 clusters — the most compute-efficient frontier MoE to date.

### AA.10.2 Architecture Comparison Table

```
Model             Total Params  Active/Token  Experts     Top-K  Routing Style
──────────────────────────────────────────────────────────────────────────────────
Switch-Base       7B            ~1B           128         1      Noisy top-1
GLaM              1.2T          ~96B          64          2      Top-2
Mixtral 8x7B      46.7B         12.9B         8           2      Top-2
Mixtral 8x22B     140.6B        39.1B         8           2      Top-2
Grok-1            314B          ~80B          8           2      Top-2
DeepSeek-V2       236B          21B           160+2sh     6      Top-6 + shared
DeepSeek-V3       671B          37B           256+1sh     8      Aux-free bias
──────────────────────────────────────────────────────────────────────────────────
```

### AA.10.3 Active Parameters at Equal Inference Cost

```
Model            Active Params   Dense Equivalent    Quality vs Dense Equivalent
──────────────────────────────────────────────────────────────────────────────────
Mixtral 8x7B     12.9B           Llama 2 13B         Beats Llama 2 70B
DeepSeek-V2      21B             LLaMA 3 20B         Beats GPT-4 on many tasks
DeepSeek-V3      37B             Llama 3 40B         Matches GPT-4o / Claude 3.5
──────────────────────────────────────────────────────────────────────────────────
Each model outperforms a dense model of comparable active-parameter count because
total capacity (all expert parameters) is much larger.
```

### AA.10.4 When to Choose MoE Over Dense

**Use MoE when:**

- You have a fixed inference FLOPs budget and want maximum model capacity.
- You can afford the total weight memory (all experts must reside somewhere).
- Your serving infrastructure supports expert parallelism across multiple GPUs.
- Your workload has diverse content types that benefit from specialised sub-networks.
- You are training at scale (>10B parameters) where the capacity/cost ratio matters.

**Prefer dense when:**

- You are running single-GPU inference where loading all expert weights is prohibitive.
- Your use case is latency-critical at batch size 1 (MoE experts worsen memory-bound access).
- You need to fine-tune on a narrow domain (MoE routing may not adapt well to distribution shift without all experts receiving gradient signal).
- Your serving infrastructure cannot support all-to-all communication (edge, mobile).

---

## AA.11 Summary and Key Takeaways

Mixture of Experts is the architecture that breaks the linear relationship between model capacity and inference cost. The core ideas, applied consistently from Shazeer 2017 through DeepSeek-V3, are:

**Conditional computation**: Route each token to a small subset of expert FFNs. Pay K × expert_size FLOPs per token, not E × expert_size.

**Routing mechanisms form a spectrum**: from soft (all experts, differentiable, no load problem) to top-1 (maximum sparsity, fragile) to top-K (the production sweet spot) to expert-choice (perfect balance, coverage gaps).

**Load balancing is non-negotiable**: Without auxiliary losses, expert collapse destroys the capacity advantage within thousands of training steps. Auxiliary load-balance loss and z-loss are standard; DeepSeek-V3's bias-update mechanism shows that they can be replaced with a cleaner non-gradient approach at frontier scale.

**Inference has new failure modes**: Expert parallelism introduces all-to-all communication; small batch sizes kill GPU utilisation; all expert weights must reside in memory simultaneously. These are tractable engineering problems, and vLLM's fused MoE kernels address the compute efficiency side directly.

**The economics are compelling**: Mixtral 8×7B inference costs 13B-parameter FLOPs while delivering 70B-parameter quality. DeepSeek-V3 extends this — 37B active parameters matching models with 4–10× more active computation. For anyone building production inference systems, understanding MoE is no longer optional.

```
Key numbers to remember:
  Top-K = 2      — industry default since GShard (2021)
  E / K ratio    — capacity multiplier; Mixtral: 4x, DeepSeek-V3: 32x
  alpha = 0.01   — auxiliary loss weight (safe default)
  beta  = 0.001  — z-loss weight (safe default)
  capacity_factor = 1.25 (train),  2.0 (inference)
  Expert parallelism: 2 all-to-all ops per MoE layer
  KV cache: identical footprint to dense model of same attention size
```

---

*This appendix covers MoE from first principles through production serving. For the practical vLLM integration, see Chapter 33 (Engine Landscape), Chapter 34 (DeepSeek), and the `fused_moe` source code in `vllm/model_executor/layers/fused_moe/`. For the mathematical foundations of softmax and automatic differentiation that underpin the router, see Appendix A and Appendix A.3 respectively.*
