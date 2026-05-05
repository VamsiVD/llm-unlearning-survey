# Survey: Unified Benchmarking of LLM Unlearning Methods
### Using the OpenUnlearning Framework

**Vamsi Dandu · CAI Spring 2026**

This repository contains all scripts and notebooks used to benchmark LLM unlearning methods using the [OpenUnlearning](https://github.com/locuslab/open-unlearning) framework.

---

## Files

| File | Description |
|---|---|
| `run_all.sh` | Train + eval all 8 methods on TOFU 1B |
| `run_auc_roc.py` | AUC-ROC faithfulness evaluation |
| `quant_probe.py` | Quantization stress test |
| `colab_llm_unlearning.ipynb` | Colab notebook for 3B scaling experiments |

---

## Setup

```bash
git clone https://github.com/locuslab/open-unlearning.git
cd open-unlearning
pip install ".[lm-eval]"
huggingface-cli login
python setup_data.py --eval
```

## Run Benchmark (1B)

```bash
bash run_all.sh
```

## Run 3B Scaling

Open `colab_llm_unlearning.ipynb` in Google Colab (A100 GPU recommended) and run all cells in order.

## Run AUC-ROC Faithfulness

```bash
# First eval the base model
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  forget_split=forget10 \
  task_name=positive_pool_base \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  ++model.model_args.attn_implementation=eager

# Then compute AUC-ROC
python run_auc_roc.py
```

## Results

| Method | Extr. Str↓ | Utility↑ | Relearn ES |
|---|---|---|---|
| GradAscent | 0.033 | 0.000 | 0.481 |
| GradDiff | 0.085 | 0.445 | 0.519 |
| NPO | 0.093 | 0.592 | 0.348 |
| SimNPO | 0.568 | 0.598 | 0.707 |
| IDK | 0.098 | 0.583 | 0.383 |
| RMU | 0.054 | 0.571 | 0.719 |
| SatImp | 0.665 | 0.597 | 0.766 |
| WGA | 0.167 | 0.595 | 0.543 |

## Compute

- LLaMA-3.2-1B: NVIDIA RTX 4070 Ti Super (16GB)
- LLaMA-3.2-3B: Google Colab A100 (40GB)
