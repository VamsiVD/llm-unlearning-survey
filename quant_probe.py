from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
import json, torch

def load_model_8bit(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cuda", load_in_8bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def evaluate_rouge(model, tokenizer, data_path, num_samples=50, label=""):
    with open(data_path) as f:
        data = json.load(f)[:num_samples]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for i, item in enumerate(data):
        inputs = tokenizer(item["question"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        score = scorer.score(item["answer"], generated)
        scores.append(score["rougeL"].fmeasure)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{num_samples}...")
    avg = sum(scores) / len(scores)
    print(f"  {label}: {avg:.3f}")
    return avg

forget_path = "clinical_forget_set_final.json"

models = {
    "Base":     "saves/finetune/clinical_finetuned_v3",
    "GradDiff": "saves/unlearn/GradDiff_clinical",
    "SimNPO":   "saves/unlearn/SimNPO_clinical",
    "RMU":      "saves/unlearn/RMU_clinical",
    "NPO":      "saves/unlearn/NPO_clinical",
}

# 4-bit results from clinical_eval.py for comparison
results_4bit = {
    "Base":     0.530,
    "GradDiff": 0.157,
    "SimNPO":   0.185,
    "RMU":      0.168,
    "NPO":      0.202,
}

print("=" * 65)
print(f"{'Model':<12} {'4-bit ROUGE-L':>14} {'8-bit ROUGE-L':>14} {'Delta':>8}")
print("=" * 65)

for name, path in models.items():
    print(f"\nLoading {name} in 8-bit...")
    model, tokenizer = load_model_8bit(path)
    score_8bit = evaluate_rouge(model, tokenizer, forget_path, label="8-bit Forget")
    delta = score_8bit - results_4bit[name]
    print(f"{name:<12} {results_4bit[name]:>14.3f} {score_8bit:>14.3f} {delta:>+8.3f}")
    del model
    torch.cuda.empty_cache()

print("=" * 65)
print("\nInterpretation: Large delta = unlearning not robust to quantization changes")
print("Small delta = forgetting is stable across precision levels")
