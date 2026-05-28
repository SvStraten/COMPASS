# COMPASS: Continual Online Foundation Model-based Process Prediction with Adaptive SubSpaces

**COMPASS** is a framework for online continual next activity prediction using Foundation Models (FMs). It addresses the cold-start problem and catastrophic forgetting that affect existing Predictive Process Monitoring (PPM) approaches by combining parameter-efficient fine-tuning with unsupervised drift detection and gradient-based knowledge consolidation — without requiring explicit task boundaries.

> **Paper:** *Online Continual Fine-Tuning of Foundation Models for Process Prediction* — submitted to ECML-PKDD 2025 Workshop on Scalable Continual Learning.

---

## ✨ Key Features

- **Foundation Model backbone** — adapts pre-trained LLMs (Tiny-LLM, DistilGPT2) to process event streams via LoRA, eliminating the cold-start problem inherent to scratch-trained models.
- **Unsupervised drift detection** — monitors rolling loss variance to autonomously detect task boundaries without ground-truth labels (task-free setting).
- **Adaptive subspace expansion** — at each detected boundary, COMPASS extracts the dominant feature directions of the new task and projects gradient updates to be orthogonal to all prior knowledge, preventing catastrophic forgetting.
- **Backward knowledge transfer** — previously learned process knowledge is preserved and reused via a unified, growing subspace *M'ₜ = [Wₚ, Mₜ]* that spans all observed tasks.
- **Test-then-train protocol** — predictions are made before each weight update, ensuring no data leakage.

---

## 📁 Repository Structure

```text
COMPASS/
│
├── Data/                        # Raw event logs (.csv)
├── Methods/
│   └── COMPASS/
│       ├── run.py               # Entry point — CLI and experiment orchestration
│       ├── engine.py            # Streaming loop, drift detection, training
│       └── keeplora_handler.py  # Subspace management and LoRA reinitialization
├── Utils/
│   ├── preprocess.py            # Event log parsing and sequence extraction
│   └── metrics.py               # Accuracy, F1, and per-event logging
├── runs/                        # Output directory for results and CSVs
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python ≥ 3.10 and a CUDA-capable GPU. Experiments were run on NVIDIA Tesla V100 (16 GB).

---

## 📊 Data Preparation

Place raw event logs (`.csv` format) in the `Data/` directory. Preprocessing, timestamp formatting, and prefix sequence extraction are handled automatically by `Utils/preprocess.py`. Preprocessed objects are cached as `.pkl` files under `Preprocessed/` to speed up repeated runs.

Datasets used in the paper:

| Dataset | Type | Drift type |
|---|---|---|
| IRO5000, ORI5000, ROI5000, OIR5000, RIO5000 | Synthetic | Recurrent |
| BPI15-REC | Real-world (BPIC 2015) | Recurrent |
| BPI20-RFP, BPI20-DD, BPI20-ID | Real-world (BPIC 2020) | Natural |

All datasets are publicly available at [data.4tu.nl](https://data.4tu.nl).

---

## 🚀 How to Run

### Basic run (task-free, Tiny-LLM)

```bash
python -m Methods.COMPASS.run \
    --dataset Data/BPIC2015_Recurrent.csv \
    --model-name arnir0/Tiny-LLM \
    --window-size 100 \
    --seed 42
```

### Task-aware variant (oracle drift boundaries)

```bash
python -m Methods.COMPASS.run \
    --dataset Data/BPIC2015_Recurrent.csv \
    --model-name distilbert/distilgpt2 \
    --oracle-drift \
    --seed 42
```

### Reproduce benchmark results (SLURM)

```bash
sbatch run_benchmark.sh COMPASS Data/ORI5000.csv
```

---

## ⚗️ Ablation Study

COMPASS has three independently ablatable components. Use the flags below to isolate each:

| Variant | Flag(s) | Description |
|---|---|---|
| LoRA only | `--no-backward-transfer` | Continuous fine-tuning, no consolidation |
| +Wₚ | `--no-Mt` | Pre-trained subspace constraint only |
| +Wₚ +Mₜ (= COMPASS) | *(default)* | Full method |
| Task-aware | `--oracle-drift` | Oracle drift boundaries instead of plateau detection |

Run all ablation jobs for BPI15-REC:

```bash
bash submit_ablation.sh   # fires 21 individual SLURM jobs
```

---

## 📐 Hyperparameters

Hyperparameters were tuned on the first 15% of each event log (validation split). The remaining 85% is used for online evaluation.

### COMPASS search space

| Parameter | Values |
|---|---|
| Window size (*W*) | {100, 500, 1000} |
| Learning rate (*η*) | {2e-3, 2e-4, 2e-5} |
| Variance threshold (*τ*) | {5e-2, 5e-3, 5e-4} |
| LoRA rank (*r*) | {8, 64, 256, 512} |
| LoRA alpha (*α*) | {16, 128, 512, 1024} |

### Optimal COMPASS hyperparameters per dataset

| Dataset | *η* | *τ* | *r* | *α* |
|---|---|---|---|---|
| IRO5000 | 2e-3 | 5e-4 | 256 | 512 |
| ORI5000 | 2e-4 | 5e-2 | 256 | 512 |
| ROI5000 | 2e-3 | 5e-4 | 256 | 512 |
| OIR5000 | 2e-4 | 5e-4 | 256 | 512 |
| RIO5000 | 2e-3 | 5e-2 | 256 | 512 |
| BPI15-REC | 2e-4 | 5e-4 | 256 | 512 |
| BPI20-RFP | 2e-4 | 5e-4 | 256 | 512 |
| BPI20-DD | 2e-4 | 5e-4 | 256 | 512 |
| BPI20-ID | 2e-4 | 5e-3 | 256 | 512 |

All runs use *W* = 100, energy thresholds *ϵ*ᵥᵥ = 0.75, *ϵ*_f = 0.95, and max subspace rank = 64.

---

## 📊 Results

Average accuracy over 5 seeds (evaluation split, 85% of each log). **Bold** = best, *italic* = second, <u>underline</u> = third, excluding DoNothing.

| Method | IRO5000 | ORI5000 | ROI5000 | OIR5000 | RIO5000 | BPI15-REC | BPI20-RFP | BPI20-DD | BPI20-ID |
|---|---|---|---|---|---|---|---|---|---|
| DoNothing | .192 | .200 | .223 | .205 | .171 | .013 | .511 | .495 | .247 |
| w = LastDrift | .220 | .770 | .246 | .709 | .269 | .484 | **.886** | .863 | .840 |
| DynaTrainCDD | .775 | .785 | .790 | .729 | .784 | .455 | .831 | .830 | .793 |
| TFCLPM | *.803* | <u>.817</u> | <u>.825</u> | .775 | **.814** | .473 | .863 | .876 | .820 |
| CNAPwP | <u>.802</u> | .816 | *.829* | <u>.780</u> | *.812* | <u>.492</u> | <u>.879</u> | <u>.881</u> | *.846* |
| COMPASS TinyLLM (task-free) | .800 | **.826** | **.831** | *.784* | <u>.812</u> | *.631* | <u>.882</u> | *.892* | <u>.845</u> |
| COMPASS TinyLLM (task-aware) | **.803** | *.826* | *.831* | **.785** | .807 | **.632** | *.882* | **.892** | **.847** |

Full Macro F1, Weighted F1 and runtime tables are available in `runs/tables.tex`.

---

## 📄 Citation

If you use COMPASS in your research, please cite:

```bibtex
@inproceedings{compass2025,
  title     = {Online Continual Fine-Tuning of Foundation Models for Process Prediction},
  booktitle = {ECML-PKDD Workshop on Scalable Continual Learning},
  year      = {2025}
}
```

---

## 🔗 Acknowledgements

COMPASS builds on [KeepLoRA](https://arxiv.org/abs/2601.19659) (Luo et al., 2026) for residual gradient adaptation and [Online-LoRA](https://openaccess.thecvf.com/content/WACV2025/papers/Wei_Online-LoRA_Task-Free_Online_Continual_Learning_via_Low_Rank_Adaptation_WACV_2025_paper.pdf) (Wei et al., 2025) for loss-plateau drift detection.
