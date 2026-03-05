#!/bin/bash
#SBATCH --job-name=pretraining
#SBATCH --output=OutputLogs/pretraining_%j.txt
#SBATCH --time=48:00:00
#SBATCH --partition=mcs.gpu.q
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1        

# --- SETUP ---
mkdir -p runs OutputLogs checkpoints 

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

# --- 1. CONVERT XES TO CSV ---
XES_FILE="Data/PrepaidTravelCost.xes"
CSV_FILE="Data/PrepaidTravelCost.csv"

if [ -f "$XES_FILE" ] && [ ! -f "$CSV_FILE" ]; then
    echo "Converting $XES_FILE to $CSV_FILE ..."
    python -c "import pm4py; log = pm4py.read_xes('$XES_FILE'); pm4py.convert_to_dataframe(log).to_csv('$CSV_FILE', index=False)"
fi

OFFLINE_DATASET="$CSV_FILE"
DATASETS=(
    "Data/RequestForPayment.csv"
)

# --- GLOBAL CONFIGURATION ---
DEVICE="auto"
MODEL_NAME="arnir0/Tiny-LLM"
FM_RANK=128
FM_ALPHA=256

PRETRAIN_SEED=40
CHECKPOINT_DIR="checkpoints/BPI20_PTC_seed_${PRETRAIN_SEED}"
LORA_PATH="${CHECKPOINT_DIR}/offline_lora.pt"
SUBS_PATH="${CHECKPOINT_DIR}/offline_subspace.pt"

echo "========================================================"
echo "PHASE 1: TRAINING FOUNDATION MODEL"
echo "Offline Data: $OFFLINE_DATASET"
echo "Seed: $PRETRAIN_SEED"
echo "========================================================"

if [ ! -f "$LORA_PATH" ]; then
    mkdir -p "$CHECKPOINT_DIR"
    
    python -m Methods.CONAP_FM_pretrained.pretrain_offline \
        --dataset "$OFFLINE_DATASET" \
        --output-dir "$CHECKPOINT_DIR" \
        --model-name "$MODEL_NAME" \
        --epochs 10 \
        --batch-size 32 \
        --calora-energy 0.97 \
        --calora-batches 50 \
        --lora-r $FM_RANK \
        --lora-alpha $FM_ALPHA 
else
    echo "Found existing checkpoint at $CHECKPOINT_DIR. Skipping pre-training."
fi

if [ ! -f "$LORA_PATH" ]; then
    echo "CRITICAL ERROR: Pre-training failed."
    exit 1
fi

done