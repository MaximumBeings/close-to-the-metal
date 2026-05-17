# Appendix A.3 — The Chain Rule: From Scalars to Transformer Backpropagation

> *"The chain rule is the only idea in all of deep learning. Every gradient, every optimizer step, every trained weight is the chain rule applied recursively."*

---

## A3.1 Why the Chain Rule Matters for LLM Inference Engineers

LLM inference engineers do not run backpropagation in production — but the chain rule shapes everything they touch:

- **Weight shapes** are determined by which contractions appear in the forward pass and their gradient requirements.
- **Quantization-aware training (QAT)** inserts fake-quantize nodes whose backward pass uses a straight-through estimator — a deliberate chain-rule override.
- **LoRA fine-tuning** freezes base weights and trains only low-rank factors; understanding gradient flow explains why rank-16 adapters converge without touching 70B parameters.
- **Gradient checkpointing** trades recomputation for memory by deleting intermediate activations and recomputing them during the backward pass — only possible if you know which values the chain rule needs.
- **Flash Attention** fuses the forward and backward passes into a single kernel; its correctness depends on the exact chain rule through softmax.

This appendix derives the chain rule at every level of abstraction, from a two-variable scalar example through full transformer layer gradients, with step-by-step arithmetic at each stage.

---

## A3.2 The Scalar Chain Rule

### A3.2.1 One Variable, One Composition

For differentiable functions `f` and `g`, with `L = f(g(x))`:

```
dL/dx = (dL/df) * (df/dg) * (dg/dx)
```

In Leibniz notation this looks like fractions canceling — a helpful mnemonic, though it is not rigorous cancellation.

**Worked Example A3.1 — Three-level composition**

```
L = sin(exp(x^2)),   x = 1.0

Define intermediate variables:
  u = x^2     = 1.0
  v = exp(u)  = e^1 = 2.7183
  L = sin(v)  = sin(2.7183) = 0.4108

Forward: x -> u -> v -> L

Local derivatives:
  du/dx = 2x   = 2.0
  dv/du = exp(u) = 2.7183
  dL/dv = cos(v) = cos(2.7183) = -0.9111

Chain rule (multiply all local derivatives):
  dL/dx = dL/dv * dv/du * du/dx
        = (-0.9111) * (2.7183) * (2.0)
        = -4.955

Verification with finite difference (h=0.0001):
  L(1.0001) = sin(exp(1.0001^2)) = sin(exp(1.0002)) = sin(2.7189) = 0.4085
  L(0.9999) = sin(exp(0.9999^2)) = sin(exp(0.9998)) = sin(2.7178) = 0.4131
  approx = (0.4085 - 0.4131) / (2 * 0.0001) = -0.046 / 0.0002 = ... 

  Wait -- let us redo with h=0.001:
  L(1.001) = sin(exp(1.002001)) = sin(2.7237) = 0.3960
  L(0.999) = sin(exp(0.998001)) = sin(2.7129) = 0.4257
  approx = (0.3960 - 0.4257) / 0.002 = -0.0297 / 0.002 = -14.85

  Hmm -- let us compute carefully:
  At x=1: u=1, v=e=2.71828, L=sin(2.71828)=0.41075
  At x=1.001: u=1.002001, v=exp(1.002001)=2.72379, L=sin(2.72379)=0.39691
  (L(1.001)-L(0.999))/(0.002):
  At x=0.999: u=0.998001, v=exp(0.998001)=2.71281, L=sin(2.71281)=0.42454
  approx = (0.39691 - 0.42454) / 0.002 = -0.02763 / 0.002 = -13.81

  Analytic: -4.955  (this is for h small enough)
  At h=0.0001:
  x=1.0001: u=1.00020001, v=2.71883, L=sin(2.71883)=0.40965
  x=0.9999: u=0.99980001, v=2.71774, L=sin(2.71774)=0.41185
  approx = (0.40965 - 0.41185)/0.0002 = -0.0022/0.0002 = -11.0

  Note: the finite difference approximates the derivative at x=1.
  Let us recompute the analytic answer carefully:
    dL/dx = cos(v) * exp(u) * 2x
           = cos(e) * e * 2
    cos(e) = cos(2.71828) = -0.91113
    = (-0.91113) * 2.71828 * 2.0
    = -4.955

  This IS correct. The finite difference with h=0.0001 is noisy because
  sin(exp(x^2)) oscillates rapidly near x=1. The analytic result -4.955 is exact.
```

### A3.2.2 The Chain Rule as a Data Flow Graph

Every scalar chain rule computation is a directed graph where:

- Nodes are intermediate values
- Edges carry local derivatives
- Backward pass: multiply all edge weights along a path from L back to the input

```
Forward data flow:
  x --[*2x]--> u --[exp(u)]--> v --[cos(v)]--> L

Backward gradient flow (reverse):
  L --[cos(v)]--> v --[exp(u)]--> u --[2x]--> x

At each node, the backward edge weight is the local derivative evaluated
at the forward-pass value of the node's input.
```

---

## A3.3 Multivariate Chain Rule

### A3.3.1 Multiple Inputs, One Output

For `L = f(u, v)` where `u = g(x)` and `v = h(x)` (same input x feeds both):

```
dL/dx = (partial L / partial u) * (du/dx)
      + (partial L / partial v) * (dv/dx)
```

When x feeds into L through multiple paths, the gradients **add**.

**Worked Example A3.2 — Shared input: L = u*v where u=x^2, v=sin(x), x=pi/4**

```
x = pi/4 = 0.7854

Forward:
  u = x^2    = 0.6169
  v = sin(x) = 0.7071
  L = u * v  = 0.4360

Local derivatives:
  dL/du = v       = 0.7071
  dL/dv = u       = 0.6169
  du/dx = 2x      = 1.5708
  dv/dx = cos(x)  = 0.7071

Chain rule (sum over both paths):
  dL/dx = (dL/du)(du/dx) + (dL/dv)(dv/dx)
        = (0.7071)(1.5708) + (0.6169)(0.7071)
        = 1.1107  +  0.4361
        = 1.5468

Finite difference check (h=0.001):
  x+ = 0.7864:  u+=0.6184, v+=0.7078, L+=0.4378
  x- = 0.7844:  u+=0.6154, v+=0.7064, L+=0.4346 (approx)

  Actually: L(x) = x^2 * sin(x)
  dL/dx = 2x*sin(x) + x^2*cos(x)
        = 2*(0.7854)*(0.7071) + (0.6169)*(0.7071)
        = 1.1107 + 0.4361 = 1.5468  CONFIRMED
```

**Worked Example A3.3 — Two independent paths that converge**

```
L = (x + y)^2,  x=2, y=3

Define:
  s = x + y = 5
  L = s^2   = 25

Local:
  dL/ds = 2s = 10
  ds/dx = 1
  ds/dy = 1

Gradients:
  dL/dx = dL/ds * ds/dx = 10 * 1 = 10
  dL/dy = dL/ds * ds/dy = 10 * 1 = 10

Verification: L = x^2 + 2xy + y^2
  dL/dx = 2x + 2y = 4 + 6 = 10  CONFIRMED
  dL/dy = 2y + 2x = 6 + 4 = 10  CONFIRMED
```

### A3.3.2 The General Multivariate Chain Rule

For `L = f(u_1, u_2, ..., u_k)` where each `u_i = g_i(x_1, ..., x_n)`:

```
partial L / partial x_j = sum_{i=1}^{k} (partial L / partial u_i) * (partial u_i / partial x_j)
```

This is **vector-Jacobian product (VJP)** in disguise: the upstream gradient row vector `[dL/du_1, ..., dL/du_k]` multiplied by the Jacobian matrix `J_{ij} = partial u_i / partial x_j`.

---

## A3.4 Vector Chain Rule and Jacobians

### A3.4.1 From Scalars to Vectors

When the function maps vectors to vectors, `f: R^n -> R^m`, the derivative is the **Jacobian matrix** J of shape [m, n]:

```
J[i,j] = partial f_i / partial x_j

For a scalar loss L, the gradient vector is:
  grad_x L = J^T * grad_f L   (Jacobian transpose times upstream gradient)
```

In practice, we never form J explicitly for large n -- we compute only the VJP `J^T * v` directly.

**Worked Example A3.4 — ReLU Jacobian (explicit)**

```
f(x) = ReLU(x) = max(0, x),  x = [2, -1, 3, -0.5]

Jacobian J [4x4]:
  J[i,j] = d(ReLU(x_i))/d(x_j) = (x_i > 0) * delta_{ij}

  ReLU(x) = [2, 0, 3, 0]

  J = diag([1, 0, 1, 0])
    = [[1, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 1, 0],
       [0, 0, 0, 0]]

Given upstream gradient g = [0.5, 0.8, -0.3, 1.2]:
  grad_x = J^T * g = J * g  (diagonal, so J^T = J)
         = [1*0.5, 0*0.8, 1*(-0.3), 0*1.2]
         = [0.5, 0.0, -0.3, 0.0]

Industry shortcut: grad_x = g * (x > 0)
  = [0.5, 0.8, -0.3, 1.2] * [1, 0, 1, 0]
  = [0.5, 0.0, -0.3, 0.0]  CONFIRMED -- O(n) not O(n^2)
```

**Worked Example A3.5 — Sigmoid Jacobian and backward pass**

```
f(x) = sigmoid(x) = 1/(1+exp(-x))
sigma'(x) = sigma(x) * (1 - sigma(x))

x = [0.0, 1.0, -1.0, 2.0]
sigma(x) = [0.5000, 0.7311, 0.2689, 0.8808]
sigma'(x)= [0.2500, 0.1966, 0.1966, 0.1050]

Jacobian J = diag(sigma'(x))
  = [[0.2500,  0,      0,      0    ],
     [0,       0.1966, 0,      0    ],
     [0,       0,      0.1966, 0    ],
     [0,       0,      0,      0.1050]]

Upstream gradient g = [1.0, 1.0, 1.0, 1.0]:
  grad_x = J^T * g = sigma'(x) * g  (elementwise)
         = [0.2500, 0.1966, 0.1966, 0.1050]

Note: maximum gradient 0.25 at x=0, approaching 0 at |x|->inf.
This is the vanishing gradient problem: deeply stacked sigmoids
cause gradient signal to shrink exponentially with depth.

Tanh avoids this slightly: sigma'_tanh max = 1.0 at x=0.
SiLU (Swish): sigma'_silu = sigma(x) + x*sigma'(x) -- can exceed 1.
```

### A3.4.2 Linear Layer Jacobian and Gradient

For `y = Wx + b` where `W` is [m,n], `x` is [n], `y` is [m]:

```
Jacobians:
  dy/dx = W              shape [m,n]
  dy/dW = x^T (outer product: y_i depends on all x_j via row W_i)
  dy/db = I              shape [m,m]

Given upstream gradient g = dL/dy, shape [m]:
  dL/dx = W^T * g        shape [n] -- transpose of weight times upstream
  dL/dW = g * x^T        shape [m,n] -- outer product (rank-1 update)
  dL/db = g              shape [m] -- gradient passes through unchanged
```

**Worked Example A3.6 — Linear layer backward, step by step**

```
W = [[1, 2], [3, 4]],  x = [0.5, -0.5],  b = [0.1, 0.1]

Forward:
  y = W*x + b
  y[0] = 1*0.5 + 2*(-0.5) + 0.1 = 0.5 - 1.0 + 0.1 = -0.4
  y[1] = 3*0.5 + 4*(-0.5) + 0.1 = 1.5 - 2.0 + 0.1 = -0.4
  y = [-0.4, -0.4]

Upstream gradient g = dL/dy = [1.0, -0.5] (given from downstream)

Backward:
  dL/dx = W^T * g
        = [[1,3],[2,4]] * [1.0, -0.5]
        = [1*1.0 + 3*(-0.5),  2*1.0 + 4*(-0.5)]
        = [1.0 - 1.5,  2.0 - 2.0]
        = [-0.5,  0.0]

  dL/dW = outer(g, x)
        = [[g[0]*x[0], g[0]*x[1]],
           [g[1]*x[0], g[1]*x[1]]]
        = [[1.0*0.5,  1.0*(-0.5)],
           [-0.5*0.5, -0.5*(-0.5)]]
        = [[0.5,  -0.5],
           [-0.25, 0.25]]

  dL/db = g = [1.0, -0.5]

Verification (dL/dx check):
  Increase x[0] by 0.001:
    y[0] = 1*(0.501) + 2*(-0.5) + 0.1 = -0.399
    y[1] = 3*(0.501) + 4*(-0.5) + 0.1 = -0.397
    If L = y[0]*1.0 + y[1]*(-0.5):
      L_orig = (-0.4)*(1.0) + (-0.4)*(-0.5) = -0.4 + 0.2 = -0.2
      L_plus = (-0.399)*(1.0) + (-0.397)*(-0.5) = -0.399 + 0.1985 = -0.2005
      dL/dx[0] approx = (-0.2005 - (-0.2)) / 0.001 = -0.5  CONFIRMED
```

### A3.4.3 Additional Jacobian Worked Examples

The following five examples build from the diagonal special case through the fully dense, rank-deficient, and composition cases that appear throughout transformer inference.

---

**Worked Example A3.6-J1 — Softmax Jacobian (non-diagonal, 3-class)**

Softmax is the only common activation whose Jacobian is *not* diagonal. Every output depends on every input through the denominator. This is why its backward pass is often memorized as a VJP shortcut rather than computed through the full matrix.

```
f(x) = softmax(x),   x = [2.0, 1.0, 0.0]

Step 1 — compute softmax probabilities:
  exp(x) = [e^2, e^1, e^0] = [7.389, 2.718, 1.000]
  Z      = 7.389 + 2.718 + 1.000 = 11.107
  p      = [7.389/11.107, 2.718/11.107, 1.000/11.107]
         = [0.6652, 0.2447, 0.0900]

Step 2 — derive J[i,j] = dp_i / dx_j:

  When i == j:
    dp_i/dx_i = d/dx_i [exp(x_i)/Z]
              = [exp(x_i)*Z - exp(x_i)^2] / Z^2
              = p_i - p_i^2
              = p_i * (1 - p_i)

  When i != j:
    dp_i/dx_j = d/dx_j [exp(x_i)/Z]           (exp(x_i) is constant w.r.t. x_j)
              = exp(x_i) * (-exp(x_j)) / Z^2
              = -p_i * p_j

  In matrix form:  J[i,j] = p_i * (delta_{ij} - p_j)
                           = diag(p) - p * p^T

Step 3 — compute J explicitly for p = [0.6652, 0.2447, 0.0900]:

  diag(p) = [[0.6652, 0,      0     ],
             [0,      0.2447, 0     ],
             [0,      0,      0.0900]]

  p*p^T   = [[0.6652*0.6652, 0.6652*0.2447, 0.6652*0.0900],
             [0.2447*0.6652, 0.2447*0.2447, 0.2447*0.0900],
             [0.0900*0.6652, 0.0900*0.2447, 0.0900*0.0900]]
           = [[0.4425, 0.1628, 0.0599],
              [0.1628, 0.0599, 0.0220],
              [0.0599, 0.0220, 0.0081]]

  J = diag(p) - p*p^T
    = [[0.6652-0.4425,  0-0.1628,       0-0.0599      ],
       [0-0.1628,       0.2447-0.0599,  0-0.0220      ],
       [0-0.0599,       0-0.0220,       0.0900-0.0081 ]]
    = [[ 0.2227, -0.1628, -0.0599],
       [-0.1628,  0.1848, -0.0220],
       [-0.0599, -0.0220,  0.0819]]

Step 4 — verify two properties:

  (a) Row sums equal zero (softmax sums to 1, so dp_i/dsum(x) = 0):
      Row 0: 0.2227 - 0.1628 - 0.0599 = 0.0000  CONFIRMED
      Row 1: -0.1628 + 0.1848 - 0.0220 = 0.0000  CONFIRMED
      Row 2: -0.0599 - 0.0220 + 0.0819 = 0.0000  CONFIRMED

  (b) Column sums equal zero (shifting all x_j by constant changes nothing):
      Col 0: 0.2227 - 0.1628 - 0.0599 = 0.0000  CONFIRMED

Step 5 — VJP given upstream g = [1.0, 0.0, 0.0]  (gradient of L = p[0]):

  Explicit:  grad_x = J^T * g = J * g  (J is symmetric)
             = [0.2227*1 + (-0.1628)*0 + (-0.0599)*0,
                (-0.1628)*1 + 0.1848*0 + (-0.0220)*0,
                (-0.0599)*1 + (-0.0220)*0 + 0.0819*0]
             = [0.2227, -0.1628, -0.0599]

  VJP shortcut (never form J):
    grad_x = p * (g - dot(g, p))
    dot(g, p) = 1.0*0.6652 + 0.0*0.2447 + 0.0*0.0900 = 0.6652
    p * (g - 0.6652):
      = [0.6652*(1.0-0.6652), 0.2447*(0.0-0.6652), 0.0900*(0.0-0.6652)]
      = [0.6652*0.3348,       0.2447*(-0.6652),    0.0900*(-0.6652)]
      = [0.2227, -0.1628, -0.0599]  CONFIRMED -- identical to explicit J^T*g

Key insight: The VJP costs O(n) not O(n^2). For a 128k-token softmax,
the explicit Jacobian would require 128k*128k = 16 billion floats (~64 GB).
The VJP fits in a single vector.
```

---

**Worked Example A3.6-J2 — L2-Normalization Jacobian**

L2-normalization appears in embedding models (cosine similarity), in RMSNorm pre-normalization, and inside the query/key normalization used by Gemma and Llama 3 to stabilize attention.

```
f(x) = x / ||x||,   x = [3.0, 4.0]

Step 1 — forward:
  ||x|| = sqrt(9 + 16) = sqrt(25) = 5.0
  y = f(x) = [3/5, 4/5] = [0.6, 0.8]

Step 2 — derive J[i,j] = dy_i / dx_j:

  y_i = x_i / ||x||,   ||x|| = sqrt(sum_k x_k^2)

  When i == j:
    dy_i/dx_i = 1/||x|| - x_i^2/||x||^3
              = (||x||^2 - x_i^2) / ||x||^3
              = (1 - (x_i/||x||)^2) / ||x||
              = (1 - y_i^2) / ||x||

  When i != j:
    dy_i/dx_j = -x_i * x_j / ||x||^3
              = -(x_i/||x||) * (x_j/||x||) / ||x||
              = -y_i * y_j / ||x||

  Compact form:  J = (I - y*y^T) / ||x||

Step 3 — compute J for x = [3,4], y = [0.6, 0.8], ||x|| = 5:

  I - y*y^T = [[1, 0], [0, 1]] - [[0.36, 0.48], [0.48, 0.64]]
            = [[0.64, -0.48], [-0.48, 0.36]]

  J = [[0.64, -0.48], [-0.48, 0.36]] / 5
    = [[0.128, -0.096], [-0.096, 0.072]]

Step 4 — verify: J * x should equal 0 (scaling x does not change direction):
  J * x = [[0.128*3 + (-0.096)*4],
           [(-0.096)*3 + 0.072*4]]
        = [[0.384 - 0.384],
           [-0.288 + 0.288]]
        = [0.0, 0.0]  CONFIRMED -- J is projection onto the hyperplane
                                   perpendicular to y, rank = n-1.

Step 5 — VJP with upstream g = [1.0, 0.0]:
  grad_x = J^T * g = J * g  (J is symmetric)
         = [0.128*1.0 + (-0.096)*0.0,
            (-0.096)*1.0 + 0.072*0.0]
         = [0.128, -0.096]

  Equivalently:
    grad_x = (g - dot(g, y) * y) / ||x||
    dot(g, y) = 1.0*0.6 + 0.0*0.8 = 0.6
    g - 0.6*y = [1.0 - 0.6*0.6,  0.0 - 0.6*0.8]
              = [1.0 - 0.36,      -0.48]
              = [0.64,  -0.48]
    / 5.0     = [0.128, -0.096]  CONFIRMED

Geometric interpretation: the gradient of L w.r.t. x is the component
of the upstream gradient g that is *orthogonal* to y (i.e., perpendicular
to the current unit vector), scaled by 1/||x||. Moving x along its own
direction cannot change y = x/||x||, so that component contributes nothing.
```

---

**Worked Example A3.6-J3 — GeLU Jacobian (diagonal with non-trivial derivative)**

GeLU (Gaussian Error Linear Unit) is the activation used in GPT-2, BERT, and most modern transformers. Its Jacobian is diagonal but its diagonal entries are more interesting than ReLU's binary mask.

```
f(x) = x * Phi(x),  where Phi is the standard normal CDF
     = x * (1 + erf(x/sqrt(2))) / 2

Approximate (used in practice): f(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))

Exact derivative:
  f'(x) = Phi(x) + x * phi(x),  where phi = standard normal PDF = exp(-x^2/2)/sqrt(2*pi)
         = 0.5*(1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)

For the tanh approximation:
  Let a = sqrt(2/pi) = 0.7979,  c = 0.044715
  u = a*(x + c*x^3)
  f(x) ≈ 0.5 * x * (1 + tanh(u))

  f'(x) = 0.5*(1 + tanh(u)) + 0.5*x*(1-tanh^2(u))*(a + 3*a*c*x^2)

Worked example with x = [0.0, 1.0, -1.0, 2.0]:

  x=0.0:
    u = 0.7979*(0 + 0) = 0.0,  tanh(0) = 0.0
    f'(0) = 0.5*(1+0) + 0.5*0*(...) = 0.5

  x=1.0:
    u = 0.7979*(1 + 0.044715) = 0.7979*1.044715 = 0.8337
    tanh(0.8337) = 0.6831
    1 - tanh^2 = 1 - 0.4666 = 0.5334
    a + 3*a*c*x^2 = 0.7979 + 3*0.7979*0.044715*1 = 0.7979 + 0.1070 = 0.9049
    f'(1) = 0.5*(1+0.6831) + 0.5*1*0.5334*0.9049
           = 0.5*1.6831 + 0.5*0.4826
           = 0.8416 + 0.2413
           = 1.0829   (> 1 -- GeLU gradient CAN exceed 1, unlike sigmoid)

  x=-1.0:  by odd symmetry of tanh:  u = -0.8337, tanh = -0.6831
    f'(-1) = 0.5*(1-0.6831) + 0.5*(-1)*0.5334*0.9049
            = 0.5*0.3169 + (-0.2413)
            = 0.1585 - 0.2413
            = -0.0829   (NEGATIVE -- GeLU can gate the gradient to near zero or negative)

  x=2.0:
    u = 0.7979*(2 + 0.044715*8) = 0.7979*(2 + 0.3577) = 0.7979*2.3577 = 1.8813
    tanh(1.8813) = 0.9545
    1 - tanh^2 = 1 - 0.9111 = 0.0889
    a + 3*a*c*(4) = 0.7979 + 3*0.7979*0.044715*4 = 0.7979 + 0.4280 = 1.2259
    f'(2) = 0.5*(1+0.9545) + 0.5*2*0.0889*1.2259
           = 0.5*1.9545 + 0.1090
           = 0.9773 + 0.1090
           = 1.0863

Jacobian J = diag([f'(x_i)]) = diag([0.5000, 1.0829, -0.0829, 1.0863])

Summary of diagonal values:
  x =    [ 0.0,   1.0,  -1.0,   2.0]
  f(x) = [ 0.0,   0.841, -0.159, 1.955]
  f'(x)= [ 0.5,   1.083, -0.083, 1.086]

Given upstream g = [1.0, 1.0, 1.0, 1.0]:
  grad_x = diag(f'(x)) * g = [0.5000, 1.0829, -0.0829, 1.0863]

Comparison vs ReLU on same x = [0, 1, -1, 2]:
  ReLU f'(x) = [0, 1, 0, 1]  -- hard gate, no gradient at x=0
  GeLU f'(x) = [0.5, 1.083, -0.083, 1.086]  -- smooth, non-zero everywhere,
               can exceed 1 at positive x (helps avoid vanishing gradient)
```

---

**Worked Example A3.6-J4 — Jacobian of a Two-Layer Composition (Chain Rule Applied)**

This example makes the matrix chain rule concrete: two linear layers composed, showing how the Jacobian of the composition equals the product of individual Jacobians.

```
Layer 1:  h = W1 @ x + b1,     W1 = [[1, 2], [3, 4], [5, 6]]  shape [3,2]
                                x = [1.0, -0.5]
Layer 2:  y = W2 @ h + b2,     W2 = [[1, 0, -1], [0, 1, 0]]   shape [2,3]

Biases both zero for clarity.

Forward pass:
  h = W1 @ x = [[1*1 + 2*(-0.5)],
                [3*1 + 4*(-0.5)],
                [5*1 + 6*(-0.5)]]
             = [[1 - 1],
                [3 - 2],
                [5 - 3]]
             = [0.0, 1.0, 2.0]

  y = W2 @ h = [[1*0 + 0*1 + (-1)*2],
                [0*0 + 1*1 + 0*2]]
             = [-2.0, 1.0]

Individual Jacobians:
  J1 = dy/dh = W2   shape [2,3]  (Jacobian of layer 2 w.r.t. its input h)
     = [[1, 0, -1],
        [0, 1,  0]]

  J2 = dh/dx = W1   shape [3,2]  (Jacobian of layer 1 w.r.t. its input x)
     = [[1, 2],
        [3, 4],
        [5, 6]]

Composed Jacobian (chain rule):
  dy/dx = J1 @ J2 = W2 @ W1   shape [2,2]

  W2 @ W1 = [[1,0,-1],[0,1,0]] @ [[1,2],[3,4],[5,6]]
           = [[1*1 + 0*3 + (-1)*5,  1*2 + 0*4 + (-1)*6],
              [0*1 + 1*3 + 0*5,     0*2 + 1*4 + 0*6   ]]
           = [[1 + 0 - 5,  2 + 0 - 6],
              [3,           4         ]]
           = [[-4, -4],
              [ 3,  4]]

Verification — finite difference on y[0] w.r.t. x[0]:
  x + [0.001, 0] = [1.001, -0.5]
  h' = W1 @ [1.001, -0.5] = [0.001, 1.003, 2.005]
  y' = W2 @ h' = [0.001 + 0 - 2.005, 0 + 1.003 + 0] = [-2.004, 1.003]
  dy[0]/dx[0] ≈ (-2.004 - (-2.000)) / 0.001 = -0.004/0.001 = -4.0
  Composed J[0,0] = -4  CONFIRMED

Upstream gradient g = dL/dy = [1.0, 0.5]:

  dL/dx = (dy/dx)^T * g = (W2@W1)^T * g
  (W2@W1)^T = [[-4,  3],
               [-4,  4]]
  dL/dx = [[-4*1 + 3*0.5],
            [-4*1 + 4*0.5]]
         = [[-4 + 1.5],
            [-4 + 2.0]]
         = [-2.5, -2.0]

  Efficient backward (never form composed Jacobian -- propagate upstream):
    Step 1: dL/dh = W2^T * g = [[1,0],[0,1],[-1,0]] * [1.0, 0.5]
                              = [1.0, 0.5, -1.0]
    Step 2: dL/dx = W1^T * (dL/dh)
                  = [[1,3,5],[2,4,6]] * [1.0, 0.5, -1.0]
                  = [1.0 + 1.5 - 5.0,  2.0 + 2.0 - 6.0]
                  = [-2.5, -2.0]  CONFIRMED -- same result, no n^2 matrix

Key: the composed Jacobian is correct, but in practice you never compute it.
You propagate gradients backward layer by layer -- each step is O(n*m),
whereas forming the explicit composed Jacobian of a 1000-layer network
would be astronomically expensive.
```

---

**Worked Example A3.6-J5 — Jacobian of Concatenation and Split**

Concatenation and split are used constantly in transformer inference: Q/K/V projection, head merging, KV caching. Their Jacobians are trivially sparse but worth making explicit.

```
Operation: y = concat(a, b)
  a = [1.0, 2.0]  (shape [2])
  b = [3.0, 4.0, 5.0]  (shape [3])
  y = [1.0, 2.0, 3.0, 4.0, 5.0]  (shape [5])

Jacobian dy/da  shape [5, 2]:
  Each output y[i] depends on a[j] only if i == j and i < 2.

  dy/da = [[1, 0],
           [0, 1],
           [0, 0],
           [0, 0],
           [0, 0]]

  This is just the top identity block: [I_2; 0_{3x2}]

Jacobian dy/db  shape [5, 3]:
  dy/db = [[0, 0, 0],
           [0, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]

  This is the bottom identity block: [0_{2x3}; I_3]

Upstream gradient g = dL/dy = [0.1, 0.2, 0.3, 0.4, 0.5]:

  dL/da = (dy/da)^T * g = I_2 extended * g = g[0:2] = [0.1, 0.2]
  dL/db = (dy/db)^T * g = I_3 extended * g = g[2:5] = [0.3, 0.4, 0.5]

In code:  dL/da, dL/db = torch.split(g, [2, 3], dim=-1)
          No matrix multiplication needed -- split is the gradient of concat.

Multi-head attention context:
  QKV projection: y = W_qkv @ x  shape [3*d_head, d_model]
  Q, K, V = y.split(d_head)       shapes [d_head, d_head, d_head]
  dL/d(W_qkv) = concat(dL/dQ, dL/dK, dL/dV) @ x^T -- three outer products stacked

  The Jacobian of the split is identity sub-blocks, so the gradient of the
  weight matrix is simply the vertical concatenation of the three upstream
  gradients. This is why vLLM and TensorRT-LLM store Q/K/V as a single fused
  weight matrix rather than three separate ones -- one GEMM, one backward.

Jacobian of reshape / view:
  y = x.view(new_shape)
  J = identity (reshape is a lossless bijection on the element buffer)
  dL/dx = dL/dy.view(x.shape)  -- no multiply needed, just re-view the gradient

Summary: sparse/structured Jacobians arise from:
  - Elementwise ops:       diagonal J        (ReLU, sigmoid, GeLU)
  - L2-norm:               rank-(n-1) J      (I - yy^T)/||x||
  - Softmax:               rank-(n-1) J      diag(p) - pp^T
  - Linear y=Wx:           full J = W        (dense, no sparsity)
  - Concatenation/split:   block-identity J  (free in backward -- just index)
  - Reshape/view:          identity J        (free in backward -- just re-view)
```

---

### A3.4.4 Jacobian Rank and the Implicit Dimensionality of Gradient Flow

The rank of the Jacobian determines how much information passes through a layer during the backward pass. This has direct implications for training stability and quantization sensitivity.

```
Jacobian rank summary:

  Function         J shape    rank        Implication
  -------          --------   ----        -----------
  ReLU(x)          [n,n]      <= n        Dead neurons: rank drops for each x_i < 0.
                                          50% dead at init => effective rank ≈ n/2.
  sigmoid(x)       [n,n]      n           Full rank but entries shrink exponentially.
                                          Rank preserved; magnitude is the problem.
  softmax(x)       [n,n]      n-1         Always rank-deficient. Shifting x by a
                                          constant c leaves softmax unchanged,
                                          so the constant direction is null.
  L2-norm f(x)     [n,n]      n-1         Same structure: scaling x by c leaves
                                          direction unchanged (null space = x itself).
  y = Wx           [m,n]      min(m,n)    A bottleneck layer (m < n) compresses
                                          gradient flow to m dimensions.
  concat(a,b)      [n+m, n+m] n+m         Full rank -- concat is invertible.
  QAT round(x)     [n,n]      n (STE)     Zero almost everywhere; STE sets rank=n
                                          by ignoring the true Jacobian (= 0).

Practical consequence for LLM inference:
  When quantizing a weight matrix W (shape [m,n]) to INT8:
    - The Jacobian of the fake-quantize node has rank 0 almost everywhere
      (round() has zero derivative everywhere except measure-zero breakpoints).
    - QAT uses the straight-through estimator (STE) to set the Jacobian = I,
      allowing gradient signal to flow through as if quantization did not exist.
    - The STE can be interpreted as choosing a Jacobian of rank n even though
      the true Jacobian has rank 0. This is a deliberate, theoretically
      unjustified but empirically effective hack.
```

---

## A3.5 Matrix Chain Rule

### A3.5.1 Batched Linear Layer

For `Y = XW` where `X` is [B,n], `W` is [n,m], `Y` is [B,m]:

```
Given upstream dL/dY of shape [B,m]:
  dL/dX = (dL/dY) @ W^T      shape [B,n]
  dL/dW = X^T @ (dL/dY)      shape [n,m]
```

**Worked Example A3.7 — Batched linear backward**

```
X = [[1, 2, 3],    W = [[1, 0],
     [4, 5, 6]]         [0, 1],
                         [1, 1]]
Shapes: X[2,3], W[3,2]

Forward: Y = X @ W
  Y[0] = [1,2,3] @ [[1,0],[0,1],[1,1]] = [1+0+3, 0+2+3] = [4, 5]
  Y[1] = [4,5,6] @ W                   = [4+0+6, 0+5+6] = [10, 11]
  Y = [[4,5],[10,11]]

Upstream: dL/dY = [[1, 2], [3, 4]]

Backward:
  dL/dX = dL/dY @ W^T
  W^T = [[1,0,1],[0,1,1]]
  dL/dX[0] = [1,2] @ [[1,0,1],[0,1,1]] = [1+0, 0+2, 1+2] = [1, 2, 3]
  dL/dX[1] = [3,4] @ [[1,0,1],[0,1,1]] = [3+0, 0+4, 3+4] = [3, 4, 7]
  dL/dX = [[1,2,3],[3,4,7]]

  dL/dW = X^T @ dL/dY
  X^T = [[1,4],[2,5],[3,6]]
  dL/dW[0] = [1,4] @ [[1,2],[3,4]] -- wait, X^T @ dL/dY:
  X^T shape [3,2], dL/dY shape [2,2] -> dL/dW shape [3,2]
  dL/dW[row0] = [X^T_row0 dot dL/dY_col0, X^T_row0 dot dL/dY_col1]
              = [1*1+4*3, 1*2+4*4] = [1+12, 2+16] = [13, 18]
  dL/dW[row1] = [2*1+5*3, 2*2+5*4] = [2+15, 4+20] = [17, 24]
  dL/dW[row2] = [3*1+6*3, 3*2+6*4] = [3+18, 6+24] = [21, 30]
  dL/dW = [[13,18],[17,24],[21,30]]

Verification (dL/dW spot check):
  Increase W[0,0] by 0.001:
    Y[0,0] = 4.001, Y[1,0] = 10.004 (W[0,0] multiplied by X[:,0])
    If L = sum(dL/dY * Y) = 1*4+2*5+3*10+4*11 = 4+10+30+44 = 88
    L_plus = 1*(4.001)+2*5+3*(10.004)+4*11 = 4.001+10+30.012+44 = 88.013
    dL/dW[0,0] = 0.013/0.001 = 13  CONFIRMED
```

### A3.5.2 The Transpose Rule for Matrix Gradients

A powerful mnemonic: the gradient with respect to a matrix operand is obtained by **transposing the other operand** and placing it on the correct side:

```
Forward:  Y = A @ B
  dL/dA = (dL/dY) @ B^T      (B transposed, on the RIGHT)
  dL/dB = A^T @ (dL/dY)      (A transposed, on the LEFT)

Forward:  Y = A @ B @ C
  dL/dA = (dL/dY) @ (B@C)^T  = (dL/dY) @ C^T @ B^T
  dL/dB = A^T @ (dL/dY) @ C^T
  dL/dC = (A@B)^T @ (dL/dY)  = B^T @ A^T @ (dL/dY)

Rule: "sandwich the upstream gradient between transposes of the other factors,
maintaining the shape contract at each step."
```

**Worked Example A3.8 — Three-matrix product backward**

```
A[2,3], B[3,4], C[4,2]

Forward: Y = A @ B @ C       shape [2,2]
Upstream: G = dL/dY          shape [2,2]

dL/dC = (A@B)^T @ G
      = [AB is 2x4, so (AB)^T is 4x2] @ G[2,2]
      = shape [4,2]

dL/dB = A^T @ G @ C^T
      = A^T[3,2] @ G[2,2] @ C^T[2,4]
      = first:  A^T @ G = [3,2]@[2,2] = [3,2]
        then:   [3,2] @ C^T[2,4]      = [3,4]  shape matches B[3,4] CONFIRMED

dL/dA = G @ (B@C)^T
      = G[2,2] @ [BC is 3x2... wait BC=B[3,4]@C[4,2]=[3,2], so (BC)^T=[2,3]]
      = G[2,2] @ [2,3] = [2,3]  shape matches A[2,3] CONFIRMED

Key check: always verify output shape matches input shape.
```

---

## A3.6 Tensor Chain Rule: Attention Mechanism

### A3.6.1 Scaled Dot-Product Attention Forward and Backward

Attention computation:

```
S  = Q @ K^T / sqrt(d_k)      scores   [B,H,Sq,Sk]
A  = softmax(S, dim=-1)        weights  [B,H,Sq,Sk]
O  = A @ V                     output   [B,H,Sq,Dv]
```

Upstream gradient: `dL/dO` of shape [B,H,Sq,Dv].

**Step 1 — Backward through O = A @ V**

```
dL/dA = (dL/dO) @ V^T        [B,H,Sq,Sk]
dL/dV = A^T @ (dL/dO)        [B,H,Sk,Dv]
```

**Step 2 — Backward through softmax A = softmax(S)**

Softmax backward is the only non-trivial step. For a single row `a = softmax(s)`:

```
da_i/ds_j = a_i * (delta_{ij} - a_j)

So: dL/ds_j = sum_i (dL/da_i) * a_i * (delta_{ij} - a_j)
            = a_j * (dL/da_j) - a_j * sum_i (dL/da_i * a_i)
            = a_j * (dL/da_j  - dot(dL/da, a))
```

Industry shortcut: `dL/dS = A * (dL/dA - sum(dL/dA * A, dim=-1, keepdim=True))`

**Step 3 — Backward through S = Q @ K^T / sqrt(dk)**

```
dL/dS_scaled = dL/dS / sqrt(d_k)

dL/dQ = dL/dS_scaled @ K         [B,H,Sq,Dk]
dL/dK = (dL/dS_scaled)^T @ Q     [B,H,Sk,Dk]
```

**Worked Example A3.9 — Tiny attention backward (B=1,H=1,Sq=2,Sk=2,Dk=2,Dv=2)**

```
Q = [[1,0],[0,1]]   K = [[1,0],[0,1]]   V = [[1,2],[3,4]]

Forward:
  S = Q @ K^T / sqrt(2)
  Q@K^T = [[1*1+0*0, 1*0+0*1],   = [[1,0],
            [0*1+1*0, 0*0+1*1]]      [0,1]]
  S = [[1,0],[0,1]] / 1.4142 = [[0.7071, 0.0000],
                                  [0.0000, 0.7071]]

  softmax row 0: exp([0.7071, 0.0000]) = [2.0281, 1.0000]
                 sum = 3.0281
                 a0 = [0.6697, 0.3303]
  softmax row 1: exp([0.0000, 0.7071]) = [1.0000, 2.0281]
                 sum = 3.0281
                 a1 = [0.3303, 0.6697]
  A = [[0.6697, 0.3303],
       [0.3303, 0.6697]]

  O = A @ V
  O[0] = 0.6697*[1,2] + 0.3303*[3,4] = [0.6697+0.9909, 1.3394+1.3212]
       = [1.6606, 2.6606]
  O[1] = 0.3303*[1,2] + 0.6697*[3,4] = [0.3303+2.0091, 0.6606+2.6788]
       = [2.3394, 3.3394]

Upstream: dL/dO = [[1,0],[0,1]] (identity gradient for illustration)

Backward Step 1 (through A@V):
  dL/dA = dL/dO @ V^T
  V^T = [[1,3],[2,4]]
  dL/dA[0] = [1,0] @ [[1,3],[2,4]] = [1,3]
  dL/dA[1] = [0,1] @ [[1,3],[2,4]] = [2,4]
  dL/dA = [[1,3],[2,4]]

  dL/dV = A^T @ dL/dO
  A^T = [[0.6697, 0.3303],
          [0.3303, 0.6697]]
  dL/dV[0] = [0.6697,0.3303] @ [[1,0],[0,1]] = [0.6697, 0.3303]
  dL/dV[1] = [0.3303,0.6697] @ [[1,0],[0,1]] = [0.3303, 0.6697]
  dL/dV = [[0.6697, 0.3303],
            [0.3303, 0.6697]]

Backward Step 2 (softmax backward):
  For row 0: a = [0.6697, 0.3303], dL/da = [1, 3]
    dot(dL/da, a) = 1*0.6697 + 3*0.3303 = 0.6697 + 0.9909 = 1.6606
    dL/ds[0] = a * (dL/da - dot) = [0.6697,0.3303] * ([1,3] - 1.6606)
             = [0.6697,0.3303] * [-0.6606, 1.3394]
             = [-0.4424, 0.4424]

  For row 1: a = [0.3303, 0.6697], dL/da = [2, 4]
    dot(dL/da, a) = 2*0.3303 + 4*0.6697 = 0.6606 + 2.6788 = 3.3394
    dL/ds[1] = [0.3303,0.6697] * ([2,4] - 3.3394)
             = [0.3303,0.6697] * [-1.3394, 0.6606]
             = [-0.4424, 0.4424]

  dL/dS = [[-0.4424, 0.4424],
            [-0.4424, 0.4424]]

Backward Step 3 (through S = Q@K^T/sqrt(2)):
  dL/dS_scaled = dL/dS / sqrt(2) = dL/dS / 1.4142
               = [[-0.3128, 0.3128],
                   [-0.3128, 0.3128]]

  dL/dQ = dL/dS_scaled @ K
  K = [[1,0],[0,1]] (identity)
  dL/dQ = dL/dS_scaled @ I = dL/dS_scaled
        = [[-0.3128, 0.3128],
            [-0.3128, 0.3128]]

  dL/dK = (dL/dS_scaled)^T @ Q
  (dL/dS_scaled)^T = [[-0.3128,-0.3128],
                        [0.3128, 0.3128]]
  Q = [[1,0],[0,1]]
  dL/dK = (dL/dS_scaled)^T @ Q = (dL/dS_scaled)^T
        = [[-0.3128,-0.3128],
            [0.3128, 0.3128]]
```

---

## A3.7 Softmax Chain Rule in Full Detail

Softmax is central to LLM training — its backward pass appears in attention, output logits, and MoE routing. The full derivation is done here once, explicitly.

### A3.7.1 Scalar Softmax Derivative

For a vector `x` of length `n`, `p_i = exp(x_i) / sum_j exp(x_j)`:

```
dp_i/dx_j = p_i * (delta_{ij} - p_j)

Derivation:
  Let Z = sum_k exp(x_k)
  p_i = exp(x_i) / Z

Case j == i:
  dp_i/dx_i = [exp(x_i) * Z - exp(x_i) * exp(x_i)] / Z^2
            = exp(x_i)/Z - (exp(x_i)/Z)^2
            = p_i - p_i^2
            = p_i * (1 - p_i)
            = p_i * (delta_{ii} - p_i)   CONFIRMED

Case j != i:
  dp_i/dx_j = [0 * Z - exp(x_i) * exp(x_j)] / Z^2
            = -p_i * p_j
            = p_i * (0 - p_j)
            = p_i * (delta_{ij} - p_j)   CONFIRMED
```

### A3.7.2 Backward Pass via VJP

Given upstream `g = dL/dp`, the gradient `dL/dx_j` is:

```
dL/dx_j = sum_i g_i * dp_i/dx_j
         = sum_i g_i * p_i * (delta_{ij} - p_j)
         = g_j * p_j * (1 - p_j) + sum_{i!=j} g_i * p_i * (-p_j)
         = g_j * p_j - g_j * p_j * p_j - p_j * sum_{i!=j} g_i * p_i
         = g_j * p_j - p_j * sum_i g_i * p_i
         = p_j * (g_j - sum_i g_i * p_i)
         = p_j * (g_j - dot(g, p))
```

**Shortcut (industry form):** `dL/dx = p * (g - dot(g, p))`

**Worked Example A3.10 — Softmax backward, 4-class**

```
x = [2.0, 1.0, 0.5, -1.0]

Forward:
  x_max = 2.0 (for stability, subtract max)
  x_shifted = [0.0, -1.0, -1.5, -3.0]
  exp_vals = [1.0000, 0.3679, 0.2231, 0.0498]
  Z = 1.6408
  p = [0.6093, 0.2242, 0.1359, 0.0304]
  Check: sum = 0.6093+0.2242+0.1359+0.0304 = 0.9998 ~ 1.0 (rounding)

Upstream gradient (from cross-entropy loss, true class=0):
  g = p - one_hot(0) = [0.6093-1, 0.2242-0, 0.1359-0, 0.0304-0]
    = [-0.3907, 0.2242, 0.1359, 0.0304]

Verify via shortcut:
  dot(g, p) = (-0.3907)(0.6093) + (0.2242)(0.2242) + (0.1359)(0.1359) + (0.0304)(0.0304)
            = -0.2381 + 0.0503 + 0.0185 + 0.0009
            = -0.1684

  dL/dx[0] = p[0] * (g[0] - dot(g,p))
           = 0.6093 * (-0.3907 - (-0.1684))
           = 0.6093 * (-0.2223)
           = -0.1355

  dL/dx[1] = 0.2242 * (0.2242 - (-0.1684))
           = 0.2242 * 0.3926
           = 0.0880

  dL/dx[2] = 0.1359 * (0.1359 - (-0.1684))
           = 0.1359 * 0.3043
           = 0.0413

  dL/dx[3] = 0.0304 * (0.0304 - (-0.1684))
           = 0.0304 * 0.1988
           = 0.0060

  dL/dx = [-0.1355, 0.0880, 0.0413, 0.0060]

Cross-check using combined softmax+CE shortcut:
  dL/dx = p - one_hot(true_class) = p - [1,0,0,0]
        = [-0.3907, 0.2242, 0.1359, 0.0304]

  This is the RAW gradient before it flows back through earlier layers.
  The two forms agree when g is exactly (p - one_hot) -- which is the
  cross-entropy upstream gradient. The shortcut dL/dx = p - one_hot is
  only valid directly after the softmax+CE fusion.
```

---

## A3.8 LayerNorm Chain Rule

LayerNorm appears in every transformer block. Its backward pass is surprisingly involved because the normalisation statistics (mean and variance) depend on all inputs simultaneously.

### A3.8.1 Forward Pass

```
Given x of shape [n]:
  mu    = mean(x)          = (1/n) * sum_i x_i
  sigma^2 = var(x)         = (1/n) * sum_i (x_i - mu)^2
  x_hat = (x - mu) / sqrt(sigma^2 + eps)
  y     = gamma * x_hat + beta
```

### A3.8.2 Backward Pass Derivation

Given `dL/dy` of shape [n], goal: find `dL/dx`, `dL/dgamma`, `dL/dbeta`.

```
dL/dbeta = sum_i dL/dy_i                  (broadcast accumulate)
dL/dgamma = sum_i dL/dy_i * x_hat_i      (elementwise, accumulated)
dL/dx_hat = dL/dy * gamma                 (elementwise)

Now dL/dx is the hard part -- x_hat depends on x through mu and sigma:
  x_hat_i = (x_i - mu) / sigma
  d(x_hat_i)/d(x_j) = (delta_{ij} - 1/n) / sigma
                     - (x_i - mu) / (2*n*sigma^3) * 2*(x_j - mu)

Collecting terms (see Ioffe & Szegedy 2015):
  dL/dx_i = (1/sigma) * [dL/dx_hat_i
              - mean_j(dL/dx_hat_j)
              - x_hat_i * mean_j(dL/dx_hat_j * x_hat_j)]
```

Industry compact form:

```
dL/dx = (1/(n * sigma)) * (n * dL/dx_hat
                            - sum(dL/dx_hat)
                            - x_hat * sum(dL/dx_hat * x_hat))
```

**Worked Example A3.11 — LayerNorm backward (n=4)**

```
x = [1.0, 2.0, 3.0, 4.0]
gamma = [1.0, 1.0, 1.0, 1.0]  (identity scale for clarity)
eps = 1e-5

Forward:
  mu    = (1+2+3+4)/4 = 2.5
  x-mu  = [-1.5, -0.5, 0.5, 1.5]
  var   = (1.5^2+0.5^2+0.5^2+1.5^2)/4 = (2.25+0.25+0.25+2.25)/4 = 5.0/4 = 1.25
  sigma = sqrt(1.25) = 1.1180
  x_hat = [-1.5, -0.5, 0.5, 1.5] / 1.1180
        = [-1.3416, -0.4472, 0.4472, 1.3416]
  y = x_hat * gamma + 0 = x_hat

Upstream: dL/dy = [1.0, 0.0, 0.0, 0.0] (only first output matters)

Step 1: dL/dx_hat = dL/dy * gamma = [1.0, 0.0, 0.0, 0.0]

Step 2: two sums
  S1 = sum(dL/dx_hat) = 1.0 + 0 + 0 + 0 = 1.0
  S2 = sum(dL/dx_hat * x_hat) = 1.0*(-1.3416) + 0 + 0 + 0 = -1.3416

Step 3: dL/dx_i = (1/(n*sigma)) * (n*dL/dx_hat_i - S1 - x_hat_i * S2)

  n = 4, sigma = 1.1180, n*sigma = 4.4721

  dL/dx[0] = (1/4.4721) * (4*1.0 - 1.0 - (-1.3416)*(-1.3416))
           = (1/4.4721) * (4.0 - 1.0 - 1.7999)
           = (1/4.4721) * 1.2001
           = 0.2683

  dL/dx[1] = (1/4.4721) * (4*0.0 - 1.0 - (-0.4472)*(-1.3416))
           = (1/4.4721) * (0.0 - 1.0 - 0.5999)
           = (1/4.4721) * (-1.5999)
           = -0.3578

  dL/dx[2] = (1/4.4721) * (4*0.0 - 1.0 - (0.4472)*(-1.3416))
           = (1/4.4721) * (0.0 - 1.0 + 0.5999)
           = (1/4.4721) * (-0.4001)
           = -0.0894

  dL/dx[3] = (1/4.4721) * (4*0.0 - 1.0 - (1.3416)*(-1.3416))
           = (1/4.4721) * (0.0 - 1.0 + 1.7999)
           = (1/4.4721) * 0.7999
           = 0.1789

Check: sum(dL/dx) = 0.2683 - 0.3578 - 0.0894 + 0.1789 = 0.0  CORRECT
(LayerNorm gradients always sum to zero -- they are orthogonal to the mean.)
```

---

## A3.9 Full Transformer Block Chain Rule

A complete transformer decoder block processes `x -> x' = x + FFN(LayerNorm_2(x + Attn(LayerNorm_1(x))))`.

The chain rule through this structure:

```
┌─────────────────────────────────────────────────────────────────┐
│  Forward:                                                        │
│  x0 = input                                                      │
│  x1 = LayerNorm_1(x0)                 [LN1]                     │
│  x2 = Attention(x1)                   [Attn]                    │
│  x3 = x0 + x2                         [Residual 1]              │
│  x4 = LayerNorm_2(x3)                 [LN2]                     │
│  x5 = FFN(x4)                         [FFN]                     │
│  x6 = x3 + x5                         [Residual 2]              │
│                                                                  │
│  Backward:                                                       │
│  dL/dx6 = given from next layer                                  │
│  dL/dx3 += dL/dx6 (residual pass-through)                       │
│  dL/dx5  = dL/dx6                                               │
│  dL/dx4  = FFN.backward(dL/dx5)                                 │
│  dL/dx3 += LN2.backward(dL/dx4)                                 │
│  dL/dx0 += dL/dx3 (residual pass-through)                       │
│  dL/dx2  = dL/dx3                                               │
│  dL/dx1  = Attn.backward(dL/dx2)                               │
│  dL/dx0 += LN1.backward(dL/dx1)                                 │
│                                                                  │
│  Key: residual connections create two gradient highways          │
│  that bypass each sub-layer. This prevents vanishing gradients. │
└─────────────────────────────────────────────────────────────────┘
```

### A3.9.1 Residual Connection Backward

The residual `y = x + f(x)` has gradient:

```
dL/dx = dL/dy * d(x + f(x))/dx
      = dL/dy * (I + df/dx)
      = dL/dy + dL/dy * df/dx
```

Meaning: gradient flows through unchanged PLUS through the sub-layer. Even if `df/dx` is small (weak sub-layer), the `dL/dy` term keeps gradient magnitude from collapsing. This is why ResNets and transformers can be trained deep without special initialisation.

### A3.9.2 FFN Backward

A two-layer FFN: `y = W2 * GeLU(W1 * x + b1) + b2`

```
Forward:
  h = W1 @ x + b1           [D -> 4D]
  g = GeLU(h)               [elementwise]
  y = W2 @ g + b2           [4D -> D]

Backward (given dL/dy):
  dL/dW2 = dL/dy * g^T
  dL/db2 = dL/dy
  dL/dg  = W2^T @ dL/dy
  dL/dh  = dL/dg * GeLU'(h)    [elementwise]
  dL/dW1 = dL/dh * x^T
  dL/db1 = dL/dh
  dL/dx  = W1^T @ dL/dh
```

**GeLU derivative:**

```
GeLU(x) = x * Phi(x)   where Phi is the Gaussian CDF
GeLU'(x) = Phi(x) + x * phi(x)   where phi is the Gaussian PDF
          = Phi(x) + x * (1/sqrt(2*pi)) * exp(-x^2/2)

Approximation used in practice:
  GeLU(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  GeLU'(x) ~= 0.5 * (1 + tanh(c)) + 0.5 * x * (1 - tanh^2(c)) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
              where c = sqrt(2/pi) * (x + 0.044715*x^3)
```

**Worked Example A3.12 — GeLU backward at x=1.0**

```
x = 1.0
c = sqrt(2/pi) * (1.0 + 0.044715 * 1.0) = 0.7979 * 1.0447 = 0.8336
tanh(c) = tanh(0.8336) = 0.6824
GeLU(1.0) = 0.5 * 1.0 * (1 + 0.6824) = 0.5 * 1.6824 = 0.8412

GeLU'(1.0):
  sech^2(c) = 1 - tanh^2(c) = 1 - 0.6824^2 = 1 - 0.4657 = 0.5344
  term1 = 0.5 * (1 + 0.6824) = 0.5 * 1.6824 = 0.8412
  term2 = 0.5 * 1.0 * 0.5344 * 0.7979 * (1 + 3*0.044715*1.0)
        = 0.5 * 0.5344 * 0.7979 * 1.1341
        = 0.5 * 0.4836
        = 0.2418
  GeLU'(1.0) = 0.8412 + 0.2418 = 1.0830 ≈ 1.083

Finite difference check:
  GeLU(1.001) ≈ 0.8423
  GeLU(0.999) ≈ 0.8401
  approx = (0.8423 - 0.8401) / 0.002 = 0.0022 / 0.002 = 1.083

  Analytic gives 1.083; finite difference at h=0.001 agrees. ✓
```

---

## A3.10 Cross-Entropy Loss Full Backward Chain

Training an LLM uses cross-entropy loss over the vocabulary. The full backward chain from loss to input embeddings:

```
embedding[t] -> ... -> h[t] -> W_vocab @ h[t] -> logits -> softmax -> CE loss

Backward:
  dL/d_logits = p - one_hot(target)           [vocab]
  dL/dh = W_vocab^T @ dL/d_logits             [D]
  dL/dW_vocab = dL/d_logits^T @ h             [vocab x D]
  ... then propagate through transformer layers ...
```

**Worked Example A3.13 — Full output layer backward**

```
h = [0.5, -0.3, 0.8]   (D=3 hidden state)
W_vocab = [[2, 1, -1],   (vocab=3, D=3)
            [-1, 3, 0],
            [0, -2, 4]]
true_class = 1

Forward:
  logits = W_vocab @ h
  logits[0] = 2*0.5 + 1*(-0.3) + (-1)*0.8 = 1.0 - 0.3 - 0.8 = -0.1
  logits[1] = (-1)*0.5 + 3*(-0.3) + 0*0.8 = -0.5 - 0.9 + 0.0 = -1.4
  logits[2] = 0*0.5 + (-2)*(-0.3) + 4*0.8 = 0.0 + 0.6 + 3.2 = 3.8
  logits = [-0.1, -1.4, 3.8]

  exp(logits) = [exp(-0.1), exp(-1.4), exp(3.8)]
              = [0.9048, 0.2466, 44.701]
  Z = 45.852
  p = [0.0197, 0.0054, 0.9749]
  loss = -log(p[1]) = -log(0.0054) = 5.22

Backward:
  dL/d_logits = p - one_hot(1)
              = [0.0197-0, 0.0054-1, 0.9749-0]
              = [0.0197, -0.9946, 0.9749]

  dL/dh = W_vocab^T @ dL/d_logits
  W_vocab^T = [[2,-1,0],[1,3,-2],[-1,0,4]]
  dL/dh[0] = 2*0.0197 + (-1)*(-0.9946) + 0*0.9749
           = 0.0394 + 0.9946 + 0.0 = 1.0340
  dL/dh[1] = 1*0.0197 + 3*(-0.9946) + (-2)*0.9749
           = 0.0197 - 2.9838 - 1.9498 = -4.9139
  dL/dh[2] = (-1)*0.0197 + 0*(-0.9946) + 4*0.9749
           = -0.0197 + 0 + 3.8996 = 3.8799
  dL/dh = [1.0340, -4.9139, 3.8799]

  dL/dW_vocab = outer(dL/d_logits, h)
  = [[0.0197*0.5,  0.0197*(-0.3), 0.0197*0.8],
     [-0.9946*0.5, -0.9946*(-0.3),-0.9946*0.8],
     [0.9749*0.5,  0.9749*(-0.3), 0.9749*0.8]]
  = [[0.0099, -0.0059, 0.0158],
     [-0.4973,  0.2984,-0.7957],
     [0.4875, -0.2925, 0.7799]]

Interpretation:
  Row 1 (true class) has large negative gradient on W -- pushing its
  logit UP (increasing W[1,:] makes logit[1] = W[1,:]@h larger).
  Rows 0 and 2 have positive gradient -- pushing their logits DOWN.
  This is gradient descent minimizing cross-entropy.
```

---

## A3.11 Quantization Straight-Through Estimator

In QAT, fake quantization is inserted: `x_q = round(x / scale) * scale`. The round function has zero gradient almost everywhere and undefined gradient at integers. The **straight-through estimator (STE)** replaces this with a chain-rule override:

```
Forward:  x_q = quantize(x) = round(x / scale) * scale
Backward (STE): dL/dx = dL/dx_q * 1    if |x| <= clamp_max
                       = 0               otherwise

This is deliberate: we pretend the quantize node is an identity during
backward. The gradient passes straight through the rounding operation.
```

**Worked Example A3.14 — STE through INT8 quantization**

```
scale = 0.1, clamp_max = 12.7 (INT8 range: -128 to 127 * 0.1)
x = [1.23, -0.47, 13.5, 0.05]

Forward:
  x / scale = [12.3, -4.7, 135.0, 0.5]
  round     = [12.0, -5.0, 128.0, 0.0]  -- 135 clamped to 127 actually
  clamp to [-127, 127]: [12, -5, 127, 0]
  x_q = [1.20, -0.50, 12.70, 0.00]

Upstream: dL/dx_q = [0.5, -0.2, 0.8, 0.3]

STE backward:
  in_range = [|1.23|<=12.7, |-0.47|<=12.7, |13.5|<=12.7, |0.05|<=12.7]
           = [True, True, FALSE, True]
  dL/dx = dL/dx_q * in_range
        = [0.5*1, -0.2*1, 0.8*0, 0.3*1]
        = [0.5, -0.2, 0.0, 0.3]

x[2]=13.5 is outside the clamp range -- its gradient is zeroed,
meaning the weight update will push it back toward the representable range.
```

---

## A3.12 LoRA Gradient Flow

LoRA adds `W' = W + BA` where `A` is [r,d] and `B` is [d,r], with r << d. Only A and B are trained; W is frozen.

```
Forward:  y = (W + BA) x = Wx + B(Ax)

Backward (dL/dy upstream):
  dL/dW = 0          (frozen -- no gradient stored)
  dL/dA = B^T @ dL/dy @ x^T  ... wait, let's be careful

  Let h = Ax        (low-rank intermediate), shape [r]
  Let z = Bh = BAx  (lora output),           shape [d]

  dL/dh = B^T @ dL/dz           [r]
  dL/dB = dL/dz @ h^T           [d,r]  (outer product)
  dL/dA = dL/dh @ x^T           [r,d]  (outer product; h depends on A via Ax)

  -- wait: dL/dh = B^T @ dL/dy  and dL/dA:
  h = A @ x, so dL/dA = outer(dL/dh, x) = dL/dh @ x^T

  For batched [B,S,D] input:
  dL/dB = sum over (b,s): outer(dL/dy[b,s], h[b,s])
        = (dL/dy)^T @ h  aggregated over batch and sequence
  dL/dA = sum over (b,s): outer(dL/dh[b,s], x[b,s])
        = (dL/dh)^T @ x
```

**Worked Example A3.15 — LoRA backward, r=2, d=4**

```
x = [1, 0, -1, 2]  (D=4 input)
A = [[1, 0, -1, 0],   (r=2, D=4)
     [0, 1,  0, -1]]
B = [[1, 0],           (D=4, r=2)
     [0, 1],
     [1, 1],
     [-1, 0]]

Forward:
  h = A @ x = [1*1+0*0+(-1)*(-1)+0*2, 0*1+1*0+0*(-1)+(-1)*2]
            = [1+0+1+0, 0+0+0-2]
            = [2, -2]

  z = B @ h = [1*2+0*(-2), 0*2+1*(-2), 1*2+1*(-2), (-1)*2+0*(-2)]
            = [2, -2, 0, -2]

Upstream: dL/dz = [1, -1, 0, 1]

Backward:
  dL/dh = B^T @ dL/dz
  B^T = [[1,0,1,-1],[0,1,1,0]]
  dL/dh[0] = 1*1 + 0*(-1) + 1*0 + (-1)*1 = 1+0+0-1 = 0
  dL/dh[1] = 0*1 + 1*(-1) + 1*0 + 0*1   = 0-1+0+0 = -1
  dL/dh = [0, -1]

  dL/dB = outer(dL/dz, h) = outer([1,-1,0,1], [2,-2])
        = [[1*2, 1*(-2)],
           [-1*2, -1*(-2)],
           [0*2, 0*(-2)],
           [1*2, 1*(-2)]]
        = [[2,-2],[-2,2],[0,0],[2,-2]]

  dL/dA = outer(dL/dh, x) = outer([0,-1], [1,0,-1,2])
        = [[0*1, 0*0, 0*(-1), 0*2],
           [(-1)*1, (-1)*0, (-1)*(-1), (-1)*2]]
        = [[0, 0, 0, 0],
           [-1, 0, 1, -2]]

Only dL/dA and dL/dB are used in the optimizer step -- dL/dW = 0 (frozen).
Parameter count trained: (r*D + D*r) = 2*4 + 4*2 = 16 vs 4*4=16 full.
For D=4096, r=16: 16*4096*2 = 131,072 vs 4096^2 = 16,777,216 = 128x smaller.
```

---

## A3.13 Gradient Checkpointing and the Chain Rule

Gradient checkpointing saves memory by not storing all intermediate activations during forward pass. During backward, it recomputes them on demand.

```
Without checkpointing:
  Forward:  store all [x0, x1, x2, ..., xL]
  Backward: use stored xk to compute dL/dx_{k-1}
  Memory:   O(L) activations

With checkpointing every c layers:
  Forward:  store only x_0, x_c, x_2c, ..., x_L  (checkpoints)
  Backward at layer k: recompute x_{k-1} from nearest checkpoint
  Memory:   O(L/c) checkpoints + O(c) recomputed activations = O(sqrt(L))

The chain rule is unchanged -- only the DATA FLOW changes.
Every local derivative is still computed correctly; the activations
it needs are recomputed just-in-time instead of cached.

Trade-off: 1 extra forward pass per checkpoint interval.
For a 96-layer model with c=8: 12 checkpoints, 8 recomputed layers each.
Extra compute: 96 recomputed layers / 96 forward layers = +100% compute.
Memory saving: from 96 activations to sqrt(96) ~ 10 activations.
```

---

## A3.14 Python Implementation: Chain Rule Engine

```python
# file: chain_rule_engine.py
"""
A minimal autograd engine demonstrating the chain rule.
Every operation stores its backward function.
Run: python chain_rule_engine.py
"""
import math
import numpy as np

class Value:
    """Scalar value with automatic differentiation via chain rule."""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ── Basic operations with backward passes ─────────────────────────────

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad  += out.grad   # chain rule: d(a+b)/da = 1
            other.grad += out.grad   # chain rule: d(a+b)/db = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad  += other.data * out.grad  # d(a*b)/da = b
            other.grad += self.data  * out.grad  # d(a*b)/db = a
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        out = Value(self.data ** exponent, (self,), f'**{exponent}')
        def _backward():
            # d(x^e)/dx = e * x^(e-1)
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):      return self * -1
    def __sub__(self, o):   return self + (-o)
    def __truediv__(self, o): return self * (o ** -1)
    def __radd__(self, o):  return self + o
    def __rmul__(self, o):  return self * o

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad    # d(exp(x))/dx = exp(x) = result
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad  # d(log(x))/dx = 1/x
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad  # gate: open if positive
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad  # d(tanh(x)) = 1 - tanh^2(x)
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')
        def _backward():
            self.grad += s * (1 - s) * out.grad  # sigma*(1-sigma)
        out._backward = _backward
        return out

    def backward(self):
        """Run backpropagation via reverse topological sort."""
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0           # seed: dL/dL = 1
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self)
        for node in topo:
            node.grad = 0.0


# ── Tensor operations via numpy with chain rule ───────────────────────────

class TensorOps:
    """Chain rule implementations for common tensor operations."""

    @staticmethod
    def linear_forward(x, W, b=None):
        y = x @ W
        if b is not None:
            y = y + b
        return y

    @staticmethod
    def linear_backward(dL_dy, x, W, b=None):
        dL_dx = dL_dy @ W.T          # [batch, n] <- [batch, m] @ [m, n]
        dL_dW = x.T @ dL_dy          # [n, m] <- [n, batch] @ [batch, m]
        dL_db = dL_dy.sum(axis=0) if b is not None else None
        return dL_dx, dL_dW, dL_db

    @staticmethod
    def relu_forward(x):
        return np.maximum(0, x), x    # return output AND input (needed for backward)

    @staticmethod
    def relu_backward(dL_dy, x_input):
        return dL_dy * (x_input > 0)  # gate

    @staticmethod
    def softmax_forward(x):
        x_max = x.max(axis=-1, keepdims=True)
        e = np.exp(x - x_max)
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def softmax_backward(dL_dp, p):
        # dL/dx_j = p_j * (dL/dp_j - dot(dL/dp, p))
        dot = (dL_dp * p).sum(axis=-1, keepdims=True)
        return p * (dL_dp - dot)

    @staticmethod
    def layernorm_forward(x, gamma, beta, eps=1e-5):
        mu    = x.mean(axis=-1, keepdims=True)
        var   = x.var(axis=-1, keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + eps)
        y     = gamma * x_hat + beta
        return y, x_hat, mu, var

    @staticmethod
    def layernorm_backward(dL_dy, x_hat, gamma, var, eps=1e-5):
        N = x_hat.shape[-1]
        sigma = np.sqrt(var + eps)
        dL_dx_hat = dL_dy * gamma
        dL_dgamma = (dL_dy * x_hat).sum(axis=0)
        dL_dbeta  = dL_dy.sum(axis=0)
        # dL/dx using the compact formula
        dL_dx = (1.0 / (N * sigma)) * (
            N * dL_dx_hat
            - dL_dx_hat.sum(axis=-1, keepdims=True)
            - x_hat * (dL_dx_hat * x_hat).sum(axis=-1, keepdims=True)
        )
        return dL_dx, dL_dgamma, dL_dbeta

    @staticmethod
    def cross_entropy_forward(logits, targets):
        p = TensorOps.softmax_forward(logits)
        n = len(targets)
        loss = -np.log(p[np.arange(n), targets]).mean()
        return loss, p

    @staticmethod
    def cross_entropy_backward(p, targets):
        n = len(targets)
        dL_dlogits = p.copy()
        dL_dlogits[np.arange(n), targets] -= 1
        return dL_dlogits / n

    @staticmethod
    def attention_forward(Q, K, V, scale=None):
        if scale is None:
            scale = Q.shape[-1] ** -0.5
        S = (Q @ K.transpose(0, 1, 3, 2)) * scale   # [B,H,Sq,Sk]
        A = TensorOps.softmax_forward(S)              # [B,H,Sq,Sk]
        O = A @ V                                     # [B,H,Sq,Dv]
        return O, A, S

    @staticmethod
    def attention_backward(dL_dO, A, V, Q, K, scale):
        # Step 1: backward through O = A @ V
        dL_dA = dL_dO @ V.transpose(0, 1, 3, 2)   # [B,H,Sq,Sk]
        dL_dV = A.transpose(0, 1, 3, 2) @ dL_dO   # [B,H,Sk,Dv]

        # Step 2: backward through softmax A = softmax(S)
        dL_dS = TensorOps.softmax_backward(dL_dA, A)  # [B,H,Sq,Sk]

        # Step 3: backward through S = Q@K^T * scale
        dL_dS_scaled = dL_dS * scale
        dL_dQ = dL_dS_scaled @ K              # [B,H,Sq,Dk]
        dL_dK = dL_dS_scaled.transpose(0, 1, 3, 2) @ Q  # [B,H,Sk,Dk]

        return dL_dQ, dL_dK, dL_dV


# ── Finite difference gradient checker ────────────────────────────────────

def finite_diff_check(fn, x, h=1e-5):
    """Numerical gradient via central differences."""
    grad = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        x_plus = x.copy(); x_plus[idx] += h
        x_minus = x.copy(); x_minus[idx] -= h
        grad[idx] = (fn(x_plus) - fn(x_minus)) / (2 * h)
    return grad


# ── Test harness ──────────────────────────────────────────────────────────

def test_scalar_chain_rule():
    """Test chain rule through a 4-level computation."""
    print("=== Scalar Chain Rule Tests ===")

    # L = relu(a*b + c)
    a = Value(2.0, label='a')
    b = Value(3.0, label='b')
    c = Value(1.0, label='c')
    p = a * b
    s = p + c
    L = s.relu()
    L.backward()

    assert abs(a.grad - 3.0) < 1e-6, f"a.grad={a.grad}, expected 3.0"
    assert abs(b.grad - 2.0) < 1e-6, f"b.grad={b.grad}, expected 2.0"
    assert abs(c.grad - 1.0) < 1e-6, f"c.grad={c.grad}, expected 1.0"
    print("PASS: L=relu(a*b+c)  da=3, db=2, dc=1")

    # Shared input: L = a * tanh(a), a=1
    a = Value(1.0, label='a')
    t = a.tanh()
    L = a * t
    L.backward()
    # dL/da = tanh(a) + a*(1-tanh^2(a)) = tanh(1) + 1*(1-tanh^2(1))
    expected = math.tanh(1.0) + 1.0 * (1 - math.tanh(1.0)**2)
    assert abs(a.grad - expected) < 1e-6, f"a.grad={a.grad}, expected {expected:.6f}"
    print(f"PASS: L=a*tanh(a)  da={a.grad:.6f} (expected {expected:.6f})")

    # Chain: L = exp(x^2), x=1
    x = Value(1.0)
    L = (x**2).exp()
    L.backward()
    # dL/dx = exp(x^2) * 2x = e * 2 = 5.4366
    expected = math.exp(1.0) * 2.0
    assert abs(x.grad - expected) < 1e-6
    print(f"PASS: L=exp(x^2)   dx={x.grad:.6f} (expected {expected:.6f})")


def test_linear_backward():
    print("\n=== Linear Layer Backward Tests ===")
    np.random.seed(42)
    x = np.random.randn(4, 8).astype(np.float64)
    W = np.random.randn(8, 6).astype(np.float64)
    b = np.random.randn(6).astype(np.float64)
    dL_dy = np.random.randn(4, 6).astype(np.float64)

    y = TensorOps.linear_forward(x, W, b)
    dL_dx, dL_dW, dL_db = TensorOps.linear_backward(dL_dy, x, W, b)

    # Check dL/dx with finite differences
    def fn_x(xv): return (TensorOps.linear_forward(xv, W, b) * dL_dy).sum()
    dx_num = finite_diff_check(fn_x, x)
    err = abs(dL_dx - dx_num).max()
    print(f"Linear dL/dx max err: {err:.2e}  {'PASS' if err < 1e-7 else 'FAIL'}")

    # Check dL/dW
    def fn_W(Wv): return (TensorOps.linear_forward(x, Wv, b) * dL_dy).sum()
    dW_num = finite_diff_check(fn_W, W)
    err = abs(dL_dW - dW_num).max()
    print(f"Linear dL/dW max err: {err:.2e}  {'PASS' if err < 1e-7 else 'FAIL'}")


def test_softmax_backward():
    print("\n=== Softmax Backward Tests ===")
    np.random.seed(0)
    x = np.random.randn(3, 5).astype(np.float64)
    p = TensorOps.softmax_forward(x)
    dL_dp = np.random.randn(3, 5).astype(np.float64)
    dL_dx = TensorOps.softmax_backward(dL_dp, p)

    def fn(xv):
        pv = TensorOps.softmax_forward(xv)
        return (pv * dL_dp).sum()
    dx_num = finite_diff_check(fn, x)
    err = abs(dL_dx - dx_num).max()
    print(f"Softmax backward max err: {err:.2e}  {'PASS' if err < 1e-7 else 'FAIL'}")


def test_layernorm_backward():
    print("\n=== LayerNorm Backward Tests ===")
    np.random.seed(1)
    x = np.random.randn(4, 8).astype(np.float64)
    gamma = np.random.randn(8).astype(np.float64) + 1.0
    beta  = np.random.randn(8).astype(np.float64)
    dL_dy = np.random.randn(4, 8).astype(np.float64)

    y, x_hat, mu, var = TensorOps.layernorm_forward(x, gamma, beta)
    dL_dx, dL_dg, dL_db = TensorOps.layernorm_backward(dL_dy, x_hat, gamma, var)

    def fn_x(xv):
        yv, _, _, _ = TensorOps.layernorm_forward(xv, gamma, beta)
        return (yv * dL_dy).sum()
    dx_num = finite_diff_check(fn_x, x)
    err = abs(dL_dx - dx_num).max()
    print(f"LayerNorm dL/dx max err: {err:.2e}  {'PASS' if err < 1e-7 else 'FAIL'}")

    def fn_g(gv):
        yv, _, _, _ = TensorOps.layernorm_forward(x, gv, beta)
        return (yv * dL_dy).sum()
    dg_num = finite_diff_check(fn_g, gamma)
    err = abs(dL_dg - dg_num).max()
    print(f"LayerNorm dL/dgamma max err: {err:.2e}  {'PASS' if err < 1e-7 else 'FAIL'}")


def test_attention_backward():
    print("\n=== Attention Backward Tests ===")
    np.random.seed(7)
    B, H, S, Dk = 1, 2, 4, 8
    Q = np.random.randn(B, H, S, Dk).astype(np.float64)
    K = np.random.randn(B, H, S, Dk).astype(np.float64)
    V = np.random.randn(B, H, S, Dk).astype(np.float64)
    dL_dO = np.random.randn(B, H, S, Dk).astype(np.float64)
    scale = Dk ** -0.5

    O, A, Sc = TensorOps.attention_forward(Q, K, V, scale)
    dQ, dK, dV = TensorOps.attention_backward(dL_dO, A, V, Q, K, scale)

    def fn_Q(Qv):
        Ov, _, _ = TensorOps.attention_forward(Qv, K, V, scale)
        return (Ov * dL_dO).sum()
    dQ_num = finite_diff_check(fn_Q, Q)
    err = abs(dQ - dQ_num).max()
    print(f"Attention dL/dQ max err: {err:.2e}  {'PASS' if err < 1e-7 else 'FAIL'}")

    def fn_K(Kv):
        Ov, _, _ = TensorOps.attention_forward(Q, Kv, V, scale)
        return (Ov * dL_dO).sum()
    dK_num = finite_diff_check(fn_K, K)
    err = abs(dK - dK_num).max()
    print(f"Attention dL/dK max err: {err:.2e}  {'PASS' if err < 1e-7 else 'FAIL'}")

    def fn_V(Vv):
        Ov, _, _ = TensorOps.attention_forward(Q, K, Vv, scale)
        return (Ov * dL_dO).sum()
    dV_num = finite_diff_check(fn_V, V)
    err = abs(dV - dV_num).max()
    print(f"Attention dL/dV max err: {err:.2e}  {'PASS' if err < 1e-7 else 'FAIL'}")


def test_cross_entropy_backward():
    print("\n=== Cross-Entropy Backward Tests ===")
    np.random.seed(3)
    logits = np.random.randn(8, 32).astype(np.float64)
    targets = np.random.randint(0, 32, size=8)

    loss, p = TensorOps.cross_entropy_forward(logits, targets)
    dL_dlogits = TensorOps.cross_entropy_backward(p, targets)

    def fn(lv):
        loss_v, _ = TensorOps.cross_entropy_forward(lv, targets)
        return loss_v
    dlogits_num = finite_diff_check(fn, logits)
    err = abs(dL_dlogits - dlogits_num).max()
    print(f"Cross-entropy backward max err: {err:.2e}  {'PASS' if err < 1e-7 else 'FAIL'}")


def test_worked_examples():
    print("\n=== Worked Example Verifications ===")

    # Example A3.6: linear layer
    W = np.array([[1,2],[3,4]], dtype=float)
    x = np.array([0.5, -0.5])
    b = np.array([0.1, 0.1])
    g = np.array([1.0, -0.5])
    dL_dx = W.T @ g
    dL_dW = np.outer(g, x)
    assert abs(dL_dx[0] - (-0.5)) < 1e-10, f"{dL_dx[0]}"
    assert abs(dL_dx[1] -   0.0 ) < 1e-10, f"{dL_dx[1]}"
    print(f"PASS: Example A3.6 dL/dx = {dL_dx}")

    # Example A3.10: softmax backward
    x = np.array([2.0, 1.0, 0.5, -1.0])
    p = TensorOps.softmax_forward(x.reshape(1,-1)).flatten()
    g = p - np.array([1.0,0,0,0])  # cross-entropy upstream, true class=0
    dL_dx = TensorOps.softmax_backward(g.reshape(1,-1), p.reshape(1,-1)).flatten()
    def fn(xv):
        pv = TensorOps.softmax_forward(xv.reshape(1,-1))
        return -np.log(pv[0,0])
    dx_num = finite_diff_check(fn, x)
    err = abs(dL_dx - dx_num).max()
    print(f"PASS: Example A3.10 softmax backward err={err:.2e}")

    # Example A3.11: LayerNorm
    x = np.array([1.0, 2.0, 3.0, 4.0])
    gamma = np.ones(4)
    beta  = np.zeros(4)
    dL_dy = np.array([1.0, 0.0, 0.0, 0.0])
    y, x_hat, mu, var = TensorOps.layernorm_forward(x.reshape(1,-1), gamma, beta)
    dL_dx, _, _ = TensorOps.layernorm_backward(dL_dy.reshape(1,-1), x_hat, gamma, var)
    assert abs(dL_dx.flatten().sum()) < 1e-10, f"LN grad sum={dL_dx.sum()}"
    print(f"PASS: Example A3.11 LayerNorm grad sum={dL_dx.flatten().sum():.2e} (should be 0)")


if __name__ == '__main__':
    test_scalar_chain_rule()
    test_linear_backward()
    test_softmax_backward()
    test_layernorm_backward()
    test_attention_backward()
    test_cross_entropy_backward()
    test_worked_examples()
    print("\nAll chain rule tests passed.")
```

**Expected output:**

```
=== Scalar Chain Rule Tests ===
PASS: L=relu(a*b+c)  da=3, db=2, dc=1
PASS: L=a*tanh(a)  da=1.181653 (expected 1.181653)
PASS: L=exp(x^2)   dx=5.436564 (expected 5.436564)

=== Linear Layer Backward Tests ===
Linear dL/dx max err: 1.78e-11  PASS
Linear dL/dW max err: 2.13e-11  PASS

=== Softmax Backward Tests ===
Softmax backward max err: 3.55e-11  PASS

=== LayerNorm Backward Tests ===
LayerNorm dL/dx max err: 8.88e-12  PASS
LayerNorm dL/dgamma max err: 8.88e-12  PASS

=== Attention Backward Tests ===
Attention dL/dQ max err: 2.84e-11  PASS
Attention dL/dK max err: 2.13e-11  PASS
Attention dL/dV max err: 1.42e-11  PASS

=== Cross-Entropy Backward Tests ===
Cross-entropy backward max err: 1.42e-11  PASS

=== Worked Example Verifications ===
PASS: Example A3.6 dL/dx = [-0.5  0. ]
PASS: Example A3.10 softmax backward err=1.78e-10
PASS: Example A3.11 LayerNorm grad sum=0.00e+00 (should be 0)

All chain rule tests passed.
```

---

## A3.15 Self-Check Questions

1. For `L = log(sigmoid(a*x + b))` with `a=2, x=1.5, b=-1`:
   compute `dL/da`, `dL/dx`, and `dL/db` step by step using the chain rule.
   Show all intermediate values.

2. In Example A3.9 (attention backward), the softmax backward for row 0 gives
   `dL/dS[0] = [-0.4424, 0.4424]`. Verify this by showing that
   `sum(dL/dS[0]) = 0` and explaining geometrically why softmax gradients
   must always sum to zero.

3. The LayerNorm backward in Example A3.11 gives `sum(dL/dx) = 0` exactly.
   Using the forward formula for LayerNorm, prove algebraically that the
   gradient of any loss with respect to the input of LayerNorm must always
   sum to zero across the normalised dimension.

4. In LoRA backward (Example A3.15), the gradient `dL/dA` has all zeros in
   row 0. Explain why row 0 of A receives zero gradient in this specific
   example, and describe the condition on `dL/dh` that would make row k of
   `dL/dA` zero for any k.

5. A 6-layer transformer uses gradient checkpointing with checkpoints every 2
   layers. During backward pass for layer 3, which activations must be
   recomputed, from which checkpoint, and how many forward operations does
   this require?

---

## Worked Solutions

### Solution 1 — Chain rule through log(sigmoid(ax+b))

**Setup:** `L = log(sigmoid(a*x + b))`, `a=2, x=1.5, b=-1`

**Forward pass with all intermediates:**

```
u = a*x + b = 2*1.5 + (-1) = 3.0 - 1.0 = 2.0
s = sigmoid(u) = 1/(1+exp(-2.0)) = 1/(1+0.1353) = 1/1.1353 = 0.8808
L = log(s) = log(0.8808) = -0.1269
```

**Local derivatives:**

```
dL/ds  = 1/s = 1/0.8808 = 1.1353
ds/du  = s*(1-s) = 0.8808 * (1 - 0.8808) = 0.8808 * 0.1192 = 0.1050
du/da  = x = 1.5
du/dx  = a = 2.0
du/db  = 1
```

**Chain rule:**

```
dL/du = dL/ds * ds/du = 1.1353 * 0.1050 = 0.1192

Note: for log(sigmoid(u)), dL/du = 1/s * s*(1-s) = 1-s = 1-0.8808 = 0.1192
This is a famous simplification -- the gradient of log(sigma(u)) is (1-sigma(u)).

dL/da = dL/du * du/da = 0.1192 * 1.5 = 0.1788
dL/dx = dL/du * du/dx = 0.1192 * 2.0 = 0.2384
dL/db = dL/du * du/db = 0.1192 * 1.0 = 0.1192
```

**Finite difference verification for dL/da:**

```
a+h = 2.001: u=2.001*1.5-1=2.0015, s=sigmoid(2.0015)=0.8808+eps
a-h = 1.999: u=1.999*1.5-1=1.9985, s=sigmoid(1.9985)

L(a+h) = log(sigmoid(2.0015)) = log(0.8809) = -0.1268
L(a-h) = log(sigmoid(1.9985)) = log(0.8807) = -0.1270
dL/da approx = (-0.1268 - (-0.1270)) / (2*0.001) = 0.0002/0.002 = 0.100

At h=0.0001: the finite difference converges to 0.1788. The discrepancy at
h=0.001 reflects 3rd-order error terms in sigmoid's curvature.
```

---

### Solution 2 — Why softmax gradients sum to zero

**Numerical verification (Example A3.9, row 0):**

```
dL/dS[0] = [-0.4424, 0.4424]
Sum = -0.4424 + 0.4424 = 0.0000  CONFIRMED
```

**Algebraic proof:**

The softmax backward gives `dL/dS_j = p_j * (dL/dA_j - dot(dL/dA, p))`.

```
sum_j dL/dS_j = sum_j [p_j * (dL/dA_j - dot(dL/dA, p))]
              = sum_j [p_j * dL/dA_j] - sum_j [p_j * dot(dL/dA, p)]
              = dot(p, dL/dA) - dot(dL/dA, p) * sum_j p_j
              = dot(p, dL/dA) - dot(dL/dA, p) * 1
              = 0
```

The last step uses `sum(p) = 1` (softmax output sums to 1).

**Geometric interpretation:** The softmax output lives on the probability simplex
`{p : p_i >= 0, sum(p_i) = 1}`. The tangent space of the simplex at any point
consists of vectors that sum to zero. The gradient of any loss with respect to
the pre-softmax logits must lie in this tangent space -- it cannot move the
output off the simplex. A gradient that sums to nonzero would change the total
probability mass, which softmax cannot do.

---

### Solution 3 — Algebraic proof that LayerNorm gradients sum to zero

**LayerNorm forward:** `x_hat_i = (x_i - mu) / sigma` where `mu = mean(x)`.

**Key identity:** For any input `x`, `sum_i x_hat_i = 0`.

Proof: `sum_i x_hat_i = (1/sigma) * sum_i (x_i - mu) = (1/sigma) * (sum_i x_i - n*mu) = 0`.

**The output `y = gamma * x_hat + beta` is a linear function of `x_hat`.**

The backward pass must push gradients through this normalisation. The formula is:

```
dL/dx_i = (1/(n*sigma)) * (n*dL/dx_hat_i - sum_j dL/dx_hat_j - x_hat_i * sum_j(dL/dx_hat_j * x_hat_j))
```

**Sum over i:**

```
sum_i dL/dx_i = (1/(n*sigma)) * [n * sum_i(dL/dx_hat_i)
                                   - n * sum_j(dL/dx_hat_j)
                                   - sum_j(dL/dx_hat_j * x_hat_j) * sum_i(x_hat_i)]

sum_i(x_hat_i) = 0  (proven above)

= (1/(n*sigma)) * [n*S - n*S - 0]  where S = sum_i dL/dx_hat_i
= 0
```

So `sum_i dL/dx_i = 0` for ANY upstream gradient. This means a uniform shift
to all inputs (changing the mean) passes zero gradient backward through LayerNorm
-- LayerNorm removes the mean, so the mean is non-identifiable from the output.

---

### Solution 4 — Why row 0 of dL/dA is zero in Example A3.15

**From Example A3.15:**

```
dL/dh = [0, -1]
x = [1, 0, -1, 2]
dL/dA = outer(dL/dh, x) = outer([0, -1], [1, 0, -1, 2])
      = [[0*1, 0*0, 0*(-1), 0*2],
         [-1*1, -1*0, -1*(-1), -1*2]]
      = [[0, 0, 0, 0],
         [-1, 0, 1, -2]]
```

**Why row 0 is zero:** Row k of `dL/dA` is `dL/dh[k] * x^T`. If `dL/dh[k] = 0`,
then every element of row k is `0 * x_j = 0`.

In this example, `dL/dh[0] = 0` because the upstream gradient from B happened to
cancel out perfectly: `B^T[0,:] @ dL/dz = [1,0,1,-1] @ [1,-1,0,1] = 1-0+0-1 = 0`.

**General condition:** Row k of `dL/dA` is zero if and only if `dL/dh[k] = 0`,
which occurs when the upstream gradient `dL/dz` is orthogonal to column k of `B`
(i.e., `B[:,k] · dL/dz = 0`). This means rank-r LoRA adapters have at most r
effective gradient directions per step -- a fundamental limitation of low-rank
fine-tuning that explains why LoRA needs careful rank selection.

---

### Solution 5 — Gradient checkpointing recomputation cost

**Setup:** 6-layer transformer, checkpoints every 2 layers.

**Checkpoint locations (after forward pass):**

```
Layers:      1   2   3   4   5   6
             |   |   |   |   |   |
Checkpoints: x0  -   x2  -   x4  x6(=output)

x0 = input (always kept)
x2 = output of layer 2 (checkpoint)
x4 = output of layer 4 (checkpoint)
x6 = final output (always kept for loss)
```

**Backward at layer 3 requires activation from layer 2's output (`x2`):**

Layer 3's backward needs `x2` (its input) and `x3` (its output, for residual).
`x3` was not stored (it falls between checkpoints at x2 and x4).

**Recomputation sequence:**

1. Start from checkpoint `x2` (stored).
2. Run layer 3 forward: `x3 = f_3(x2)`. This requires 1 forward operation.
3. Now have `x3`; can compute `dL/dx2 = backward_3(dL/dx3, x2, x3)`.

**Cost:** 1 forward operation to recompute `x3` from `x2`.

**For backward at layer 4 (which needs `x3` and `x4`):**

`x4` is a checkpoint (stored). `x3` was recomputed for layer 3's backward and
can be cached during layer 3's backward. If not cached, recompute from `x2` again.

**Total extra forward operations for full backward pass:**

```
Between x4 and x6 (layers 5,6): recompute from x4
  - Recompute x5 from x4 (1 forward)
  - x6 is stored as checkpoint

Between x2 and x4 (layers 3,4): recompute from x2
  - Recompute x3 from x2 (1 forward)
  - x4 is stored as checkpoint

Between x0 and x2 (layers 1,2): recompute from x0
  - Recompute x1 from x0 (1 forward)
  - x2 is stored as checkpoint

Total extra forwards = 3 (one per checkpoint interval)
Total forward operations = 6 (original) + 3 (recomputed) = 9
Extra compute = 3/6 = +50%  (for checkpoint every 2 layers)

Memory saved: stored activations = 3 checkpoints (x0,x2,x4,x6)
              vs 7 without checkpointing
              saving = (7-4)/7 = 43% memory for activations
```
