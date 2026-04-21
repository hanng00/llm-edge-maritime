# Maritime Edge LLM - LoRA Fine-Tuning

Fine-tune Gemma 2B with LoRA on Colab T4 using Keras + JAX.

## Results


| Model                    | Threat | Action | Full  |
| ------------------------ | ------ | ------ | ----- |
| Gemma 2B (raw)           | 28.3%  | 30.0%  | 18.3% |
| Gemma 2B (LoRA, 1 epoch) | 23.3%  | 33.3%  | 20.0% |


**Improvement**: +1.7% full accuracy (minimal — needs more epochs/data)

---

## Setup

1. **Fresh runtime**: Runtime → Disconnect and delete runtime
2. Runtime → Change runtime type → **T4 GPU**
3. Add Kaggle secrets: 🔑 → `KAGGLE_USERNAME` and `KAGGLE_KEY`

---

## Cell 1: Setup

```python
!pip install -q keras-hub keras

import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from google.colab import userdata
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')

import keras
keras.mixed_precision.set_global_policy("bfloat16")

import keras_hub
import json
import time

print(f"Keras: {keras.__version__}")
```

## Cell 2: Load Model

```python
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_instruct_2b_en")
print(f"✓ Model loaded")
print(f"Dtype: {gemma_lm.backbone.dtype}")
```

## Cell 3: Upload Data

```python
from google.colab import files

uploaded = files.upload()  # Upload train_data.jsonl

scenarios = []
filename = list(uploaded.keys())[0]
with open(filename) as f:
    for line in f:
        scenarios.append(json.loads(line))
print(f"Loaded {len(scenarios)} scenarios")
```

## Cell 4: Train/Eval Split

```python
import random
random.seed(42)
random.shuffle(scenarios)

split = int(len(scenarios) * 0.8)
train_data = scenarios[:split]
eval_data = scenarios[split:]

print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
```

## Cell 5: Define Inference

```python
TEMPLATE = """You are a tactical AI for an autonomous maritime drone. Analyze the scenario and decide the threat level and action.

Scenario:
{scenario}

Respond with valid JSON only. Choose ONE value for each field:
- threat_level: one of "none", "low", "medium", "high", "critical"
- action: one of "continue", "monitor", "evade", "alert", "abort"
- reasoning: brief explanation
- confidence: number between 0.0 and 1.0

JSON response:"""

def format_scenario(s):
    contacts = "\n".join([
        f"- {v['vessel_type']} at {v['bearing']:.0f}°, {v['distance']:.1f}nm, {v['speed']:.0f}kn, {'AIS' if v['ais_active'] else 'NO AIS'}"
        for v in s["vessels"]
    ])
    return TEMPLATE.format(scenario=f"""Mission: {s["mission_type"]}
Own: {s["own_heading"]:.0f}°/{s["own_speed"]:.0f}kn
Conditions: {s["weather"]}, vis={s["visibility"]}, {s["time_of_day"]}, comms={s["comms_status"]}
Contacts:
{contacts}""")

# Compile sampler ONCE (outside predict loop)
sampler = keras_hub.samplers.TopKSampler(k=5, seed=42)
gemma_lm.compile(sampler=sampler)

def predict(scenario):
    prompt = format_scenario(scenario)
    response = gemma_lm.generate(prompt, max_length=600)
    return response[len(prompt):]

def parse(response):
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except: pass
    return None

# Test (first call compiles, may be slow)
print("Testing inference...")
result = predict(scenarios[0]["scenario"])
print(f"Response:\n{result[:400]}")
print(f"\nParsed: {parse(result)}")
```

## Cell 6: Evaluate Raw Model

```python
def run_eval(data, label=""):
    correct_threat, correct_action, correct_full, errors = 0, 0, 0, 0
    results = []
    
    print(f"Evaluating {len(data)} scenarios...")
    t0 = time.time()
    
    for i, item in enumerate(data):
        s, gt = item["scenario"], item["decision"]
        resp = predict(s)
        pred = parse(resp)
        
        if not pred:
            errors += 1
            continue
        
        pt = pred.get("threat_level", "").lower()
        pa = pred.get("action", "").lower()
        gt_t, gt_a = gt["threat_level"].lower(), gt["action"].lower()
        
        if pt == gt_t: correct_threat += 1
        if pa == gt_a: correct_action += 1
        if pt == gt_t and pa == gt_a: correct_full += 1
        
        results.append({"pred": f"{pt}/{pa}", "gt": f"{gt_t}/{gt_a}"})
        
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(data)} ({time.time()-t0:.0f}s)")
    
    valid = len(data) - errors
    acc = {
        "threat": correct_threat/valid if valid > 0 else 0,
        "action": correct_action/valid if valid > 0 else 0,
        "full": correct_full/valid if valid > 0 else 0,
    }
    
    print(f"\n{label} Results:")
    print(f"  Threat: {acc['threat']*100:.1f}%")
    print(f"  Action: {acc['action']*100:.1f}%")
    print(f"  Full:   {acc['full']*100:.1f}%")
    print(f"  Errors: {errors}")
    
    return {"accuracy": acc, "errors": errors, "results": results}

raw_results = run_eval(eval_data, "RAW MODEL")
```

## Cell 7: Save Raw Results

```python
with open("results_gemma2b_raw.json", "w") as f:
    json.dump({"model": "gemma_instruct_2b_en", "stage": "raw", **raw_results}, f, indent=2)
print("✓ Saved results_gemma2b_raw.json")
```

## Cell 8: Enable LoRA

```python
gemma_lm.backbone.enable_lora(rank=4)

trainable = sum(v.numpy().size for v in gemma_lm.backbone.trainable_weights)
total = sum(v.numpy().size for v in gemma_lm.backbone.weights)
print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
```

## Cell 9: Prepare Training Data

```python
train_texts = []
for item in train_data:
    prompt = format_scenario(item["scenario"])
    response = json.dumps({
        "threat_level": item["decision"]["threat_level"],
        "action": item["decision"]["action"],
        "reasoning": item["decision"]["reasoning"],
        "confidence": item["decision"]["confidence"]
    })
    train_texts.append(prompt + response)

print(f"Prepared {len(train_texts)} training examples")
```

## Cell 10: Configure Training

```python
gemma_lm.preprocessor.sequence_length = 512

optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
print("✓ Training configured")
```

## Cell 11: Train

```python
print(f"Training on {len(train_texts)} examples...")
print("Expected time: ~15-30 min on T4")

history = gemma_lm.fit(x=train_texts, epochs=1, batch_size=1)

with open("training_history.json", "w") as f:
    json.dump(history.history, f, indent=2)

print("✓ Training complete")
print(f"Final loss: {history.history['loss'][-1]:.4f}")
```

## Cell 12: Evaluate Fine-Tuned Model

```python
# Recompile with sampler for inference
gemma_lm.compile(sampler=sampler)

finetuned_results = run_eval(eval_data, "FINE-TUNED MODEL")
```

## Cell 13: Save Fine-Tuned Results

```python
with open("results_gemma2b_finetuned.json", "w") as f:
    json.dump({"model": "gemma_instruct_2b_en", "stage": "finetuned", **finetuned_results}, f, indent=2)
print("✓ Saved results_gemma2b_finetuned.json")
```

## Cell 14: Save Model

```python
gemma_lm.save("./gemma2b_maritime_lora.keras")

!zip -r gemma2b_maritime_lora.zip gemma2b_maritime_lora.keras/
files.download("gemma2b_maritime_lora.zip")
```

## Cell 15: Compare Results

```python
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print(f"{'Model':<35} {'Threat':>10} {'Action':>10} {'Full':>10}")
print("-" * 60)
print(f"{'Rule-based baseline':<35} {'44.3%':>10} {'57.7%':>10} {'36.0%':>10}")
print(f"{'Gemma 2B (raw)':<35} {raw_results['accuracy']['threat']*100:>9.1f}% {raw_results['accuracy']['action']*100:>9.1f}% {raw_results['accuracy']['full']*100:>9.1f}%")
print(f"{'Gemma 2B (fine-tuned)':<35} {finetuned_results['accuracy']['threat']*100:>9.1f}% {finetuned_results['accuracy']['action']*100:>9.1f}% {finetuned_results['accuracy']['full']*100:>9.1f}%")
print("=" * 60)

delta = (finetuned_results['accuracy']['full'] - raw_results['accuracy']['full']) * 100
print(f"\nFine-tuning improvement: {delta:+.1f}% (full accuracy)")
```

---

## Hyperparameter Tuning

To improve results, try:

```python
# More epochs
history = gemma_lm.fit(x=train_texts, epochs=3, batch_size=1)

# Higher LoRA rank
gemma_lm.backbone.enable_lora(rank=8)  # or 16

# Larger batch (if memory allows)
history = gemma_lm.fit(x=train_texts, epochs=3, batch_size=4)
```

---

## Notes

- **Why Keras/JAX?** Avoids `bitsandbytes` CUDA issues on Colab
- **Why Gemma 2B?** Gemma 4 E4B OOMs on T4 during fine-tuning
- **Inference speed**: ~4-6s/scenario after JIT compilation
- **Memory**: ~14GB VRAM during training

