# Appendix V — Quantization Calibration Workflow

> *"The model you downloaded is not the model you will serve. Calibration is the gap between them."*

---

## V.1 What Calibration Does

Chapter 10 explained the mathematics of quantization: how FP16 weights are
compressed to INT8, INT4, or FP8, and what the quantization error looks like.
This appendix is about the practical workflow: given a model checkpoint in
BF16/FP16, how do you produce a calibrated quantized version that retains as
much of the original quality as possible?

The key insight is that naive round-to-nearest quantization (round each weight
to the nearest representable value) produces unnecessarily high quality loss.
**Calibration-aware methods** use a small dataset of representative inputs to
observe which weights and activations are most sensitive to quantization error,
and use that information to choose better quantization parameters.

---

## V.2 The Quantization Parameter Decision Tree

Before calibrating, make three decisions:

**1. Which format?**

| Format | Hardware | Memory | Inference speed | Quality loss |
|---|---|---|---|---|
| FP8 (E4M3/E5M2) | H100, MI300X | 2× vs FP16 | 1.5–1.8× faster | < 0.5% |
| INT8 (W8A8) | All CUDA ≥ SM80 | 2× vs FP16 | 1.3–1.5× faster | < 1% |
| AWQ INT4 (W4A16) | All CUDA | 4× vs FP16 | 2–3× faster | 1–3% |
| GPTQ INT4 (W4A16) | All CUDA | 4× vs FP16 | 2–3× faster | 1–4% |
| GGUF Q4_K_M (llama.cpp) | All (CPU+GPU) | ~4× vs FP16 | Varies | 2–4% |

**2. Per-tensor or per-channel (group) scaling?**

- Per-tensor: one scale per weight matrix. Fastest but lowest quality.
- Per-channel: one scale per output channel. Better quality.
- Per-group (AWQ/GPTQ default, group=128): one scale per 128-weight group.
  Best quality for INT4, only ~3% overhead.

**3. Static or dynamic activation quantization?**

- W8A16: weights INT8, activations FP16. No activation calibration needed.
- W8A8: weights INT8, activations INT8. Requires activation calibration.
- W4A16: weights INT4, activations FP16 (most common for LLMs).
- FP8 (E4M3): weights and activations FP8. Requires activation calibration.

---

## V.3 AWQ Calibration

Activation-aware Weight Quantization (AWQ) is the recommended method for INT4
quantization of LLMs for production use. It identifies *salient channels* —
weight channels that correspond to high-magnitude activations in the
calibration data — and protects them with higher-precision scaling.

### V.3.1 How AWQ works

**Observation**: A small subset of weight channels (typically < 1%) are
responsible for a disproportionate share of quantization error. These channels
correspond to activation dimensions with high variance on real data.

**Solution**: For salient channels, multiply the weight by a scale factor `s > 1`
before quantizing (making the quantization error smaller for those channels),
and divide the corresponding activation by `s` at runtime (which is free if
the activation is just a vector multiply).

```
W_quantized = quantize(W × diag(s))     (scale up before quantizing)
activation_adjusted = activation / s     (scale down at inference)
net_output = (activation / s) × (W × s) = activation × W   (unchanged)
```

The scale factors `s` are found by minimising reconstruction error on the
calibration dataset.

### V.3.2 Running AWQ calibration with `llm-awq`

```bash
pip install autoawq

python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'meta-llama/Llama-3.1-70B-Instruct'
quant_path = './Llama-3.1-70B-Instruct-AWQ'

# Load model in FP16
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    device_map='auto',          # distribute across available GPUs
    safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantization configuration
quant_config = {
    'zero_point': True,         # asymmetric quantization
    'q_group_size': 128,        # 128-weight groups (recommended)
    'w_bit': 4,                 # INT4
    'version': 'GEMM',          # GEMM kernel (fastest for batched inference)
    # 'version': 'GEMV',        # use for batch_size=1 (decode-heavy)
}

# Run calibration — typically 128 samples, ~30-90 minutes on A100
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data='pileval',       # built-in calibration dataset
    # calib_data=your_custom_dataset  # list of strings
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print('AWQ calibration complete.')
"
```

### V.3.3 Custom calibration data

The choice of calibration data matters. The goal is to sample inputs that
are representative of your production distribution:

```python
# Load a domain-specific calibration dataset
from datasets import load_dataset

# Option 1: General-purpose (default)
calib_data = 'pileval'          # 512 samples from The Pile

# Option 2: Code generation deployment
calib_data = load_dataset("codeparrot/github-code", split="train",
                           streaming=True)
calib_texts = [sample['code'] for sample in
               itertools.islice(calib_data, 128)]

# Option 3: Your own production logs (best practice)
# Sample 128–512 representative prompts from your request logs
calib_texts = load_production_sample(n=256)

# Apply to quantization
model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)
```

**Calibration data guidelines:**
- 128–512 samples is sufficient (diminishing returns beyond 512)
- Average length: 512–2048 tokens (longer samples cover more of the model)
- Distribution: match your production data, not generic internet text
- Diversity: include edge cases and difficult examples your model will encounter
- Do not include your test set (data leakage)

### V.3.4 AWQ calibration timeline

| Model size | Hardware | Time | Notes |
|---|---|---|---|
| 7–8B | 1× A100 | 15–25 min | Fast, single GPU |
| 13–14B | 1× A100 | 25–40 min | Single GPU |
| 70B | 2× A100 | 60–90 min | Multi-GPU distribution |
| 405B | 8× A100 | 4–8 hours | Node-level distribution |

---

## V.4 GPTQ Calibration

GPTQ (Generative Pre-trained Transformer Quantization) uses second-order
information (approximate Hessians) to find better quantization parameters
than AWQ for some models. It generally produces slightly higher quality at
the cost of longer calibration time.

### V.4.1 Running GPTQ calibration with `AutoGPTQ`

```bash
pip install auto-gptq optimum

python -c "
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset

model_name = 'meta-llama/Llama-3.1-8B-Instruct'
quant_config = BaseQuantizeConfig(
    bits=4,                     # INT4
    group_size=128,             # 128-weight groups
    damp_percent=0.1,           # Hessian damping (0.1 = recommended)
    desc_act=False,             # activation reordering (slower but better)
    sym=True,                   # symmetric quantization
    true_sequential=True,       # quantize layer by layer (lower memory)
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quant_config
)

# Calibration data — GPTQ requires tokenized inputs directly
traindataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
calibration = [
    tokenizer(sample['text'], return_tensors='pt',
              max_length=2048, truncation=True)
    for sample in traindataset.select(range(128))
    if len(sample['text'].strip()) > 50
]

# Calibrate (30–120 minutes depending on model size)
model.quantize(calibration)

# Save
model.save_quantized('./Llama-3.1-8B-GPTQ', use_safetensors=True)
tokenizer.save_pretrained('./Llama-3.1-8B-GPTQ')
"
```

### V.4.2 AWQ vs GPTQ comparison

| Property | AWQ | GPTQ |
|---|---|---|
| Calibration speed | Faster | 2–3× slower |
| Memory during calibration | Lower | Higher (Hessian storage) |
| Quality (PPL delta vs FP16) | +0.10–0.25 | +0.08–0.20 |
| Inference speed | GEMM-optimized | Depends on `desc_act` |
| Best for | Production serving | Maximum quality preservation |

**Rule of thumb**: use AWQ for production quantization where calibration needs
to be fast. Use GPTQ when you need the last 0.05 perplexity point preserved.

---

## V.5 FP8 Calibration with `llm-compressor`

FP8 quantization is the recommended format for H100 and MI300X deployments.
It requires activation calibration to set per-tensor scaling factors.

### V.5.1 What FP8 calibration measures

FP8 has a very small dynamic range (E4M3: max representable value ~448). For
weights, the maximum absolute value is typically < 1.0 and FP8 is fine.
For activations, occasional large values (outliers) can saturate the FP8
range and produce NaN or severe quantization error.

Calibration measures the maximum absolute activation value per tensor across
the calibration dataset and sets a `scale = max_abs / FP8_MAX` factor:

```
activation_fp8 = round_to_fp8(activation / scale)
actual_value   = activation_fp8 × scale
```

If `scale` is set too small (based on clean inputs), real outlier values
will saturate. If set too large, precision is wasted. Calibration finds the
right scale.

### V.5.2 Running FP8 calibration

```bash
pip install llmcompressor

python -c "
from llmcompressor.transformers import SparseAutoModelForCausalLM
from llmcompressor.transformers.compression.helpers import (
    calculate_offload_device_map,
    custom_offload_device_map
)
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoTokenizer

model_stub = 'meta-llama/Llama-3.1-70B-Instruct'
output_dir = './Llama-3.1-70B-FP8'

# Load model with automatic device mapping
device_map = calculate_offload_device_map(
    model_stub,
    reserve_for_hessians=False,
    num_gpus=2,
    torch_dtype='bfloat16'
)
model = SparseAutoModelForCausalLM.from_pretrained(
    model_stub, device_map=device_map, torch_dtype='bfloat16'
)
tokenizer = AutoTokenizer.from_pretrained(model_stub)

# FP8 quantization recipe
recipe = QuantizationModifier(
    targets='Linear',
    scheme='FP8_DYNAMIC',       # dynamic per-token activation scaling
    # scheme='FP8_STATIC',      # static (calibration-set) scaling
    ignore=['lm_head'],         # keep output projection in FP16
)

# Calibration dataset
from datasets import load_dataset
ds = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft')
samples = [tokenizer(row['prompt'], return_tensors='pt',
                     max_length=2048, truncation=True)
           for row in ds.select(range(512))]

# Run calibration
from llmcompressor.transformers import oneshot
oneshot(model=model, dataset=samples, recipe=recipe,
        max_seq_length=2048, num_calibration_samples=512)

# Save
model.save_pretrained(output_dir, save_compressed=True)
tokenizer.save_pretrained(output_dir)
"
```

### V.5.3 Dynamic vs static FP8

**FP8_DYNAMIC**: scale is computed per-token at inference time (max of absolute
activation values in the token's activation tensor). No calibration required
for the scale values, but adds runtime overhead. Recommended for models with
high activation variance.

**FP8_STATIC**: scale is fixed from calibration statistics. Zero runtime
overhead for scale computation. Requires careful calibration. Can degrade
if production data has more extreme activations than calibration data.

| Mode | Calibration needed | Runtime overhead | Recommended for |
|---|---|---|---|
| FP8_DYNAMIC | None | +2–5% latency | General use, unknown distribution |
| FP8_STATIC | Required | 0% | Production with stable input distribution |

---

## V.6 GGUF Quantization for llama.cpp

llama.cpp's `llama-quantize` tool converts GGUF FP16 models to compressed
formats. No Python calibration framework is needed — the quantization is done
entirely in C++.

### V.6.1 The quantize tool

```bash
# Step 1: Convert HuggingFace model to GGUF FP16
python convert_hf_to_gguf.py \
    meta-llama/Llama-3.1-70B-Instruct \
    --outfile Llama-3.1-70B-F16.gguf \
    --outtype f16

# Step 2: Quantize to target format
./build/bin/llama-quantize \
    Llama-3.1-70B-F16.gguf \
    Llama-3.1-70B-Q4_K_M.gguf \
    Q4_K_M

# Available formats (from fastest/smallest to slowest/largest)
# Q2_K:    2-bit, k-quant, very aggressive compression
# Q3_K_S:  3-bit, k-quant, small scale
# Q3_K_M:  3-bit, k-quant, medium scale (recommended minimum)
# Q4_0:    4-bit, original format
# Q4_K_S:  4-bit, k-quant, small
# Q4_K_M:  4-bit, k-quant, medium (RECOMMENDED: best quality/size)
# Q5_K_M:  5-bit, k-quant, medium
# Q6_K:    6-bit, k-quant (near-lossless for most use cases)
# Q8_0:    8-bit, minimal loss
# F16:     FP16 reference
# IQ4_XS:  4-bit importance-matrix quant (better than Q4_K_M at same size)
```

### V.6.2 Importance matrix quantization (IQ)

IQ-series quants (IQ3_XS, IQ4_XS, IQ4_NL) use an **importance matrix** — a
per-weight importance score derived from activation statistics on a calibration
set — to protect the most important weights during quantization.

```bash
# Generate importance matrix (requires calibration data)
./build/bin/llama-imatrix \
    --model Llama-3.1-70B-F16.gguf \
    --training-data calibration.txt \  # plain text file of calibration samples
    --output imatrix.dat \
    --chunks 128                       # number of calibration chunks

# Quantize using importance matrix
./build/bin/llama-quantize \
    --imatrix imatrix.dat \
    Llama-3.1-70B-F16.gguf \
    Llama-3.1-70B-IQ4_XS.gguf \
    IQ4_XS

# IQ4_XS typically achieves Q5_K_M quality at Q4_K_M size
```

### V.6.3 Preparing calibration text for IQ

```python
# Generate calibration.txt from a dataset
from datasets import load_dataset

ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
with open('calibration.txt', 'w') as f:
    for row in ds.select(range(512)):
        text = row['text'].strip()
        if len(text) > 100:
            f.write(text + '\n\n')
```

---

## V.7 Evaluating Quantization Quality

Never deploy a quantized model without measuring quality regression.
Use a structured evaluation pipeline.

### V.7.1 Perplexity measurement

Perplexity on a held-out text corpus is the fastest proxy for quality:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math

def measure_perplexity(model_path: str, dataset_name: str = 'wikitext',
                       num_samples: int = 512, max_length: int = 1024) -> float:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map='auto', torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    ds = load_dataset(dataset_name, 'wikitext-2-raw-v1', split='test')
    texts = [row['text'] for row in ds if len(row['text'].strip()) > 100][:num_samples]
    
    total_loss = 0
    total_tokens = 0
    
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt',
                          max_length=max_length, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
        
        total_loss   += outputs.loss.item() * inputs['input_ids'].shape[1]
        total_tokens += inputs['input_ids'].shape[1]
    
    ppl = math.exp(total_loss / total_tokens)
    return ppl

# Compare baseline vs quantized
base_ppl  = measure_perplexity('meta-llama/Llama-3.1-8B-Instruct')
awq_ppl   = measure_perplexity('./Llama-3.1-8B-AWQ')
gptq_ppl  = measure_perplexity('./Llama-3.1-8B-GPTQ')
gguf_ppl  = measure_perplexity('./Llama-3.1-8B-Q4_K_M')  # via llama.cpp server

print(f"Baseline FP16: {base_ppl:.3f}")
print(f"AWQ INT4:      {awq_ppl:.3f}  (+{awq_ppl - base_ppl:.3f})")
print(f"GPTQ INT4:     {gptq_ppl:.3f}  (+{gptq_ppl - base_ppl:.3f})")
print(f"GGUF Q4_K_M:   {gguf_ppl:.3f}  (+{gguf_ppl - base_ppl:.3f})")
```

### V.7.2 Task-specific evaluation with `lm-evaluation-harness`

Perplexity is fast but coarse. For production models, run task-specific
benchmarks:

```bash
pip install lm-eval

# Evaluate AWQ model on MMLU and HellaSwag
lm_eval --model vllm \
        --model_args pretrained=./Llama-3.1-8B-AWQ,quantization=awq \
        --tasks mmlu,hellaswag,arc_challenge \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path ./eval_results/awq/

# Evaluate baseline
lm_eval --model hf \
        --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
        --tasks mmlu,hellaswag,arc_challenge \
        --num_fewshot 5 \
        --batch_size 8 \
        --output_path ./eval_results/baseline/

# Compare
python compare_evals.py ./eval_results/baseline/ ./eval_results/awq/
```

### V.7.3 Quantization quality thresholds

Use these thresholds to decide if a quantization is acceptable for production:

| Metric | Acceptable degradation | Reject if |
|---|---|---|
| Perplexity delta | < +0.5 | > +1.0 |
| MMLU score drop | < 1.5% | > 3.0% |
| HumanEval drop | < 2.0% | > 4.0% |
| Task-specific metrics | < 2.0% | > 5.0% |

If a quantization exceeds the reject threshold, try:
1. Increase group size granularity (128 → 64 → 32)
2. Switch method (AWQ → GPTQ or vice versa)
3. Use a higher bit width (INT4 → INT6 → INT8)
4. Use a domain-matched calibration dataset

---

## V.8 Worked Example V.1 — End-to-End FP8 Calibration

**Goal**: Quantize Llama 3.1 70B to FP8 for deployment on H100.

**Step 1: Verify hardware capability**
```bash
nvidia-smi | grep "H100"
python -c "import torch; print(torch.cuda.get_device_capability())"
# Should print (9, 0) for H100 — FP8 requires SM89+ (H100=SM90)
```

**Step 2: Install llm-compressor**
```bash
pip install llmcompressor>=0.4.0
```

**Step 3: Prepare calibration dataset (1,000 samples from production logs)**
```python
import json
with open('production_samples.jsonl') as f:
    samples = [json.loads(line)['prompt'] for line in f][:1000]
```

**Step 4: Run calibration (~2 hours on 2× H100)**
```python
# [calibration script from §V.5.2, with your dataset]
```

**Step 5: Validate perplexity**
```
Baseline FP16: 7.23
FP8 Static:    7.28  (+0.05, acceptable)
FP8 Dynamic:   7.25  (+0.02, excellent)
```

**Step 6: Run MMLU benchmark**
```
FP16:         83.6%
FP8 Static:   83.2%  (-0.4%, within threshold)
FP8 Dynamic:  83.5%  (-0.1%, excellent)
```

**Step 7: Deploy**
```bash
vllm serve ./Llama-3.1-70B-FP8 \
    --quantization fp8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90
```

**Result**: 1.7× throughput improvement vs FP16, 0.4% MMLU degradation.

---

## V.8.5 Mixed-Precision Quantization: Which Layers Stay in FP16

Quantizing every layer uniformly (e.g., all to INT4) maximises compression but
concentrates error in sensitive layers. Mixed-precision quantization keeps the
most sensitive layers in FP16 (or FP8) and quantizes the rest aggressively.

### Which layers are most sensitive

Sensitivity is typically highest for:

1. **First and last transformer layers** — the first embedding layer and the
   final lm_head handle the raw token distribution; errors here propagate to
   all outputs.
2. **Attention layers vs FFN layers** — attention weight matrices are more
   sensitive than FFN in most models; the QKV projection and output projection
   show higher perplexity degradation per quantization step than the MLP.
3. **Layers with high activation variance** — calibrate per-layer and look at
   the activation distribution range. Layers where the ratio of max/mean
   activation exceeds 10× are candidates for FP16.

### Per-layer sensitivity analysis

```python
# Per-layer sensitivity analysis: quantize one layer at a time and measure PPL delta
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def per_layer_sensitivity(
    model_name: str,
    calibration_text: str,
    quant_bits: int = 4,
):
    """
    For each transformer layer, temporarily quantize it and measure PPL increase.
    Returns a dict mapping layer_name → delta_ppl.
    """
    from torch import nn

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    inputs = tokenizer(calibration_text, return_tensors="pt").to(model.device)

    def ppl(m):
        with torch.no_grad():
            out = m(**inputs, labels=inputs["input_ids"])
        return torch.exp(out.loss).item()

    baseline = ppl(model)
    sensitivity = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            original_weight = module.weight.data.clone()
            # Simulate INT4 quantization
            scale = original_weight.abs().max() / (2 ** (quant_bits - 1))
            quantized = torch.round(original_weight / scale).clamp(
                -(2**(quant_bits-1)), 2**(quant_bits-1)-1
            ) * scale
            module.weight.data = quantized

            delta = ppl(model) - baseline
            sensitivity[name] = round(delta, 4)

            module.weight.data = original_weight   # restore

    return sensitivity

# Usage (requires GPU):
# sens = per_layer_sensitivity("meta-llama/Llama-3.1-7B", calibration_text)
# sensitive = sorted(sens.items(), key=lambda x: x[1], reverse=True)[:10]
# print("Top 10 most sensitive layers:")
# for name, delta in sensitive:
#     print(f"  {name}: +{delta:.4f} PPL")
```

### Mixed-precision recipe

AWQ supports per-layer bit widths via `autoawq`:

```python
from awq import AutoAWQForCausalLM

# Override: keep first 2 and last 2 layers in 8-bit, rest in 4-bit
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
    # Layer-specific overrides (AWQ internal notation)
    "layer_overrides": {
        "model.layers.0": {"w_bit": 8},
        "model.layers.1": {"w_bit": 8},
        "model.layers.78": {"w_bit": 8},   # second-to-last (79 layers for 70B)
        "model.layers.79": {"w_bit": 8},   # last layer
        "lm_head": {"w_bit": 16},          # always keep lm_head in FP16
    }
}
```

---

## V.8.6 Calibration Data Bias Detection

The calibration dataset shapes the quantization. A biased calibration set
produces a quantized model that performs well on the calibration distribution
and poorly on everything else.

### Common bias patterns and detection

```python
from collections import Counter
import numpy as np

def analyse_calibration_data(texts: list[str], tokenizer) -> dict:
    """
    Analyse a calibration dataset for common bias patterns.
    Returns a report with statistics and warnings.
    """
    warnings = []

    # 1. Check language distribution
    from langdetect import detect
    langs = Counter(detect(t) for t in texts[:200])
    dominant_lang = langs.most_common(1)[0]
    if dominant_lang[1] / len(texts[:200]) > 0.95:
        warnings.append(
            f"WARNING: {dominant_lang[0]} dominates {dominant_lang[1]:.0%} "
            f"of calibration data — multilingual performance may degrade."
        )

    # 2. Check token length distribution
    lengths = [len(tokenizer.encode(t)) for t in texts]
    p10, p90 = np.percentile(lengths, [10, 90])
    if p90 / p10 > 10:
        warnings.append(
            f"WARNING: High length variance (p10={p10:.0f}, p90={p90:.0f}). "
            f"Very short or very long sequences may be under-represented."
        )

    # 3. Check for template repetition
    first_tokens = [t[:30] for t in texts]
    repetition = len(first_tokens) - len(set(first_tokens))
    if repetition / len(texts) > 0.5:
        warnings.append(
            f"WARNING: {repetition/len(texts):.0%} of samples share the "
            f"same prefix — template bias detected."
        )

    # 4. Check domain coverage (simple heuristic: vocabulary diversity)
    all_tokens = sum([tokenizer.encode(t) for t in texts[:100]], [])
    unique_ratio = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
    if unique_ratio < 0.05:
        warnings.append(
            f"WARNING: Low vocabulary diversity (unique ratio={unique_ratio:.3f}). "
            f"Calibration data may not cover the deployment domain."
        )

    return {
        "n_samples":     len(texts),
        "lang_dist":     dict(langs.most_common(5)),
        "length_p50":    int(np.percentile(lengths, 50)),
        "length_p90":    int(np.percentile(lengths, 90)),
        "unique_tok_ratio": round(unique_ratio, 3),
        "warnings":      warnings,
    }
```

### Recommended calibration data composition

For a general-purpose deployment calibration dataset (1,000 samples):

| Source | Proportion | Notes |
|---|---|---|
| Production request logs | 50% | Best proxy for real distribution |
| Wikipedia (diverse topics) | 20% | General knowledge coverage |
| Code (GitHub) | 15% | Coding task coverage |
| Books / long-form prose | 10% | Long-range coherence |
| Domain-specific (if known) | 5% | Prevent domain regression |

Avoid: social media, very short texts (< 64 tokens), single-topic dumps.

---

## V.8.7 Quantization and LoRA Interaction

LoRA fine-tunes a low-rank adapter on top of frozen base weights. When the
base model is quantized, the interaction between quantized weights and FP16
adapter weights introduces specific considerations.

### QLoRA: adapting on top of 4-bit base

QLoRA (Dettmers et al., 2023) is the canonical approach: keep base weights
in NF4 (4-bit), add LoRA adapters in FP16/BF16. The forward pass computes:

```
y = (W_quantized + AB) × x
```

Where `W_quantized` is the dequantized (FP16) base weight, `A` and `B` are
the LoRA matrices. The key insight: LoRA gradients flow only through `A` and
`B`, not through `W_quantized`, so quantization noise in `W` is fixed during
fine-tuning (but still affects inference quality).

### Serving a QLoRA-adapted quantized model

```python
from vllm import LLM, SamplingParams

# Load quantized base + LoRA adapter
llm = LLM(
    model="meta-llama/Llama-3.1-70B",
    quantization="awq",                     # base model is AWQ INT4
    enable_lora=True,
    max_lora_rank=64,
    gpu_memory_utilization=0.90,
)

from vllm.lora.request import LoRARequest

# Serve requests with the LoRA adapter applied
outputs = llm.generate(
    ["Write a poem about inference"],
    SamplingParams(temperature=0.7, max_tokens=256),
    lora_request=LoRARequest(
        "my_adapter",
        1,
        "/path/to/lora/adapter"
    )
)
```

### Memory overhead of LoRA on top of quantized base

```python
def lora_memory_gb(rank: int, d_model: int, n_layers: int,
                   n_modules: int = 4, dtype_bytes: int = 2) -> float:
    """
    Estimate LoRA adapter memory.
    n_modules: typically 4 (q_proj, k_proj, v_proj, o_proj) or 8 (+ gate/up/down)
    """
    # Each module: A (d_model × rank) + B (rank × d_model)
    params_per_module = 2 * d_model * rank
    total_params = params_per_module * n_modules * n_layers
    return total_params * dtype_bytes / 1e9

# Llama 3.1 70B: 80 layers, d_model=8192, rank=64, 4 attention modules
mem_gb = lora_memory_gb(64, 8192, 80, 4)
print(f"LoRA rank-64 memory: {mem_gb:.2f} GB")   # → ~0.27 GB — negligible
```

### When quantization degrades LoRA quality

If your fine-tuned LoRA adapter underperforms vs the FP16 baseline after
quantization, the most likely cause is **rank collapse**: the quantized base
weights have degraded to the point where the adapter's low-rank updates
cannot compensate. Remedies:

1. Use a higher quantization level for the base (INT8 instead of INT4)
2. Increase LoRA rank (64 → 128) to give the adapter more capacity
3. Use mixed-precision: quantize FFN to INT4, keep attention in INT8

### Test harness — quantization calibration arithmetic

```python
# ── test_appendix_v.py ───────────────────────────────────────────────────
"""Offline arithmetic tests for quantization calibration appendix.
No GPU required. Run with: python test_appendix_v.py"""

import math


def bits_to_dtype_bytes(bits: int) -> float:
    return bits / 8


def weight_memory_gb(params_B: float, bits: int) -> float:
    return params_B * 1e9 * bits_to_dtype_bytes(bits) / 1e9


def gptq_calibration_time_minutes(params_B: float, n_layers: int = 80) -> float:
    """Rough estimate: GPTQ calibration scales linearly with model size."""
    # Empirical: ~1 min per 7B parameters on A100
    return (params_B / 7) * n_layers / 80


def lora_memory_gb(rank: int, d_model: int, n_layers: int,
                   n_modules: int = 4, dtype_bytes: int = 2) -> float:
    total_params = 2 * d_model * rank * n_modules * n_layers
    return total_params * dtype_bytes / 1e9


# ── Tests ────────────────────────────────────────────────────────────────

def test_weight_compression_ratios():
    params_B = 70.0  # Llama 3.1 70B
    fp16_gb  = weight_memory_gb(params_B, 16)
    int8_gb  = weight_memory_gb(params_B, 8)
    int4_gb  = weight_memory_gb(params_B, 4)

    assert abs(fp16_gb - 140.0) < 1.0,  f"FP16 should be ~140 GB, got {fp16_gb:.1f}"
    assert abs(int8_gb - 70.0)  < 0.5,  f"INT8 should be ~70 GB, got {int8_gb:.1f}"
    assert abs(int4_gb - 35.0)  < 0.5,  f"INT4 should be ~35 GB, got {int4_gb:.1f}"
    assert abs(fp16_gb / int4_gb - 4.0) < 0.1, "INT4 should be 4× smaller than FP16"
    print(f"PASS: FP16={fp16_gb:.0f}GB, INT8={int8_gb:.0f}GB, INT4={int4_gb:.0f}GB "
          f"(4.0× compression)")


def test_ppl_delta_threshold():
    """Verify quantization quality thresholds."""
    # Acceptable: < 0.5 PPL delta for INT8, < 1.0 for INT4
    ppl_fp16  = 7.23
    ppl_int8  = 7.45
    ppl_int4  = 7.89

    delta_int8 = ppl_int8 - ppl_fp16
    delta_int4 = ppl_int4 - ppl_fp16

    assert delta_int8 < 0.5,  f"INT8 PPL delta {delta_int8:.2f} exceeds threshold 0.5"
    assert delta_int4 < 1.0,  f"INT4 PPL delta {delta_int4:.2f} exceeds threshold 1.0"
    print(f"PASS: INT8 ΔPP={delta_int8:.2f} (<0.5✓), INT4 ΔPP={delta_int4:.2f} (<1.0✓)")


def test_lora_memory_overhead():
    mem = lora_memory_gb(64, 8192, 80, 4)
    assert mem < 0.5, f"LoRA rank-64 should be < 0.5 GB, got {mem:.3f} GB"
    print(f"PASS: LoRA rank-64 memory = {mem:.3f} GB (negligible overhead)")


def test_awq_vs_gptq_calibration_time():
    """AWQ should be significantly faster than GPTQ for 70B models."""
    # AWQ: ~30–60 min; GPTQ: ~120–240 min on A100
    awq_time_min  = 45   # midpoint estimate
    gptq_time_min = 180  # midpoint estimate
    ratio = gptq_time_min / awq_time_min
    assert ratio > 2.0, f"GPTQ should be >2× slower than AWQ, got {ratio:.1f}×"
    print(f"PASS: GPTQ ~{ratio:.1f}× slower than AWQ for 70B calibration")


def test_gguf_format_compression():
    """Verify GGUF quantization compresses as expected."""
    params_B = 70.0
    formats = {
        "Q2_K": 2.625, "Q4_0": 4.0, "Q4_K_M": 4.5,
        "Q5_K_M": 5.5, "Q8_0": 8.0, "F16": 16.0,
    }
    for fmt, bits in formats.items():
        gb = weight_memory_gb(params_B, bits)
        # Just verify the formula is consistent
        assert gb > 0
        assert gb <= weight_memory_gb(params_B, 16)
    print(f"PASS: GGUF format memory calculations consistent "
          f"(Q2_K={weight_memory_gb(70, 2.625):.0f}GB → "
          f"F16={weight_memory_gb(70, 16):.0f}GB)")


if __name__ == "__main__":
    test_weight_compression_ratios()
    test_ppl_delta_threshold()
    test_lora_memory_overhead()
    test_awq_vs_gptq_calibration_time()
    test_gguf_format_compression()
    print("\n✓ All quantization calibration tests passed.")
```

**Expected output:**
```
PASS: FP16=140GB, INT8=70GB, INT4=35GB (4.0× compression)
PASS: INT8 ΔPP=0.22 (<0.5✓), INT4 ΔPP=0.66 (<1.0✓)
PASS: LoRA rank-64 memory = 0.268 GB (negligible overhead)
PASS: GPTQ ~4.0× slower than AWQ for 70B calibration
PASS: GGUF format memory calculations consistent (Q2_K=23GB → F16=140GB)

✓ All quantization calibration tests passed.
```

---

## V.9 Self-Check Questions

1. AWQ uses salient channel scaling: multiply salient weights by `s` before
   quantizing, divide activations by `s` at runtime. Why does this reduce
   quantization error for salient channels? What determines which channels
   are "salient"?

2. GPTQ uses second-order information (the Hessian of the quantization error).
   Explain intuitively why the Hessian provides better quantization parameters
   than just minimising the first-order (reconstruction) error.

3. FP8 E4M3 has a maximum representable value of 448.0. A weight matrix has
   max absolute value 2.3. What scale factor would you set for static FP8
   quantization? What happens if an activation at inference has max absolute
   value 600.0?

4. You quantize a 70B model to Q4_K_M GGUF and measure perplexity degradation
   of +1.8 (above the +1.0 reject threshold). List three concrete actions you
   would take, in order of increasing cost, to recover quality.

5. An IQ4_XS quantization requires an importance matrix. The importance matrix
   is generated from a calibration corpus. If the calibration corpus is very
   different from your production distribution (e.g., calibrated on Wikipedia
   but deployed for code generation), describe how this would affect the
   importance scores and what the consequence for code generation quality would be.
