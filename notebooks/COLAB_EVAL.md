# Maritime Edge LLM - Raw Model Evaluation

Evaluate Gemma 4 E2B without fine-tuning on Colab T4.

## Results


| Model             | Threat | Action | Full  | Parse Errors |
| ----------------- | ------ | ------ | ----- | ------------ |
| Gemma 4 E2B (raw) | 25.7%  | 37.0%  | 22.7% | 0%           |


---

## Setup

1. **Fresh runtime**: Runtime → Disconnect and delete runtime
2. Runtime → Change runtime type → **T4 GPU**
3. Add Kaggle secrets: 🔑 → `KAGGLE_USERNAME` and `KAGGLE_KEY`

---

## Cell 1: Setup

```python
# Fix CUDA version mismatch
!pip uninstall -y torchvision
!pip install torchvision --index-url https://download.pytorch.org/whl/cu130
!pip install -q -U transformers accelerate kagglehub

import os
from google.colab import userdata
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Cell 2: Load Model

```python
import kagglehub
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_PATH = kagglehub.model_download("google/gemma-4/transformers/gemma-4-e2b-it")

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print(f"✓ Model loaded")
print(f"Memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")
```

## Cell 3: Upload Data

```python
from google.colab import files
import json

uploaded = files.upload()  # Upload train_data.jsonl

scenarios = []
filename = list(uploaded.keys())[0]
with open(filename) as f:
    for line in f:
        scenarios.append(json.loads(line))
print(f"Loaded {len(scenarios)} scenarios")
```

## Cell 4: Define Inference

```python
import time

SYSTEM = """You are a tactical AI for an autonomous maritime drone. Given sensor data, respond with ONLY valid JSON:
{"threat_level": "none|low|medium|high|critical", "action": "continue|monitor|evade|alert|abort", "reasoning": "brief", "confidence": 0.0-1.0}"""

def format_scenario(s):
    contacts = "\n".join([
        f"- {v['vessel_type']} at {v['bearing']:.0f}°, {v['distance']:.1f}nm, {v['speed']:.0f}kn, {'AIS' if v['ais_active'] else 'NO AIS'}"
        for v in s["vessels"]
    ])
    return f"""Mission: {s["mission_type"]}
Own: {s["own_heading"]:.0f}°/{s["own_speed"]:.0f}kn
Conditions: {s["weather"]}, vis={s["visibility"]}, {s["time_of_day"]}, comms={s["comms_status"]}
Contacts:
{contacts}

JSON:"""

def predict(scenario):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": format_scenario(scenario)}
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    return processor.decode(outputs[0][input_len:], skip_special_tokens=True)

def parse(response):
    try:
        start, end = response.find("{"), response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except: pass
    return None

# Test
print("Testing...")
result = predict(scenarios[0]["scenario"])
print(f"Response: {result[:300]}")
print(f"Parsed: {parse(result)}")
```

## Cell 5: Run Evaluation

```python
N = 300  # Full dataset

correct_threat, correct_action, correct_full, errors = 0, 0, 0, 0
results = []

print(f"Evaluating {N} scenarios...")
t0 = time.time()

for i, item in enumerate(scenarios[:N]):
    s, gt = item["scenario"], item["decision"]
    resp = predict(s)
    pred = parse(resp)
    
    if not pred:
        errors += 1
        results.append({"id": s["id"], "error": True, "response": resp[:100]})
        continue
    
    pt = pred.get("threat_level", "").lower()
    pa = pred.get("action", "").lower()
    gt_t, gt_a = gt["threat_level"].lower(), gt["action"].lower()
    
    if pt == gt_t: correct_threat += 1
    if pa == gt_a: correct_action += 1
    if pt == gt_t and pa == gt_a: correct_full += 1
    
    results.append({"id": s["id"], "pred": f"{pt}/{pa}", "gt": f"{gt_t}/{gt_a}", "match": pt==gt_t and pa==gt_a})
    
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{N} ({time.time()-t0:.0f}s)")

valid = N - errors
print(f"\n{'='*50}")
print(f"RESULTS: Gemma 4 E2B (raw)")
print(f"{'='*50}")
print(f"Parse errors: {errors}/{N}")
print(f"Threat:  {correct_threat/valid*100:.1f}%")
print(f"Action:  {correct_action/valid*100:.1f}%")
print(f"Full:    {correct_full/valid*100:.1f}%")
```

## Cell 6: Save Results

```python
output = {
    "model": "gemma-4-e2b-it",
    "n_scenarios": N,
    "parse_errors": errors,
    "accuracy": {
        "threat": correct_threat/valid if valid > 0 else 0,
        "action": correct_action/valid if valid > 0 else 0,
        "full": correct_full/valid if valid > 0 else 0,
    },
    "results": results
}

with open("results_raw_e2b.json", "w") as f:
    json.dump(output, f, indent=2)
    
files.download("results_raw_e2b.json")
```

---

## Notes

- **Kaggle credentials**: kaggle.com → Account → Create New Token
- **E4B OOM**: E4B requires ~16GB, T4 only has 15GB
- **Inference speed**: ~6s/scenario on T4

