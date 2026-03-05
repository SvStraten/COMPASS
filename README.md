# CONAP-FM: Continual Online Next Activity Prediction with Foundation Models

**CONAP-FM** is a novel framework that establishes Foundation Process Prediction Models (FPPMs) for sustainable online continual learning (OCL) in Predictive Process Monitoring (PPM). Our approach successfully addresses the computational bottlenecks of continuous Foundation Model retraining by combining **parameter-efficient fine-tuning**, **unsupervised loss plateau detection**, and **causal-aware gradient adaptation**. This allows the model to autonomously manage concept drift, prevent catastrophic forgetting, and enable backward knowledge transfer without relying on explicit task boundaries.

## 📁 Repository Structure

```text
CONAP-FM/
│
├── Data/                   # Raw event logs (.csv)
├── Methods/                
│   ├── CONAP_FM/           # Core streaming engine and LoRA handlers
│   └── CONAP_FM_pretrained/# Offline pre-training logic
├── Setup/                  # Core experiment setup, samplers, and data loaders
├── Utils/                  # Preprocessing routines and metric calculation
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## ⚙️ Installation
```bash
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```
## Hyperparameters

### Hyperparameter Search Space
* **Window Size ($\mathcal{W}$):** {100, 500, 1000, 1500} 
* **Learning Rate ($\eta$):** {2e-3, 2e-4, 2e-5} 
* **Variance Threshold ($\tau$):** {5e-2, 5e-3, 5e-4} 
* **LoRA Rank ($r$):** {8, 64, 128, 256} 
* **LoRA Alpha ($\alpha$):** {16, 128, 256, 512} 
* **Hard Buffer Size ($|B_{hard}|$):** {100, 300, 500} 
* **Regularization ($\lambda$):** {0.5, 1.0, 1.5} 
* **Buffer Size ($\rho$):** {50, 100, 200} 
* **Correlation Threshold ($\epsilon$):** {0.3, 0.5, 0.7} 
* **Weibull Shape ($W_k$):** {0.6, 0.8, 1.0}
* **Weibull Scale ($W_\lambda$):** {0.25, 0.50, 0.75} 

![Optimal Hyperparameters](hyperparameters.png)


## Macro F1-Scores 
![Macro F1-Score](macro_f1.png)
