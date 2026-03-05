#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=OutputLogs/benchmark_%j.txt
#SBATCH --time=48:00:00
#SBATCH --partition=mcs.gpu.q
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1        

# --- SETUP ---
mkdir -p runs
mkdir -p OutputLogs 

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

# --- Configuration ---
# List of all datasets to benchmark
DATASETS=(
    # "Data/IRO5000.csv"
    # "Data/ORI5000.csv"
    # "Data/ROI5000.csv"
    # "Data/OIR5000.csv"
    # "Data/RIO5000.csv"
    # "Data/BPIC2015_Recurrent.csv"
    "Data/RequestForPayment.csv"
    # "Data/InternationalDeclarations.csv"
    # "Data/DomesticDeclarations.csv"
    # "Data/BPIC2017.csv"
)

# Global Defaults
DEVICE="auto"

# --- PREPARE OUTPUT DIRECTORIES ---
mkdir -p "runs/CONAP_FM"

echo "========================================================"
echo "STARTING BENCHMARK"
echo "Datasets: ${#DATASETS[@]}"
echo "Seeds: 40 to 44"
echo "========================================================"

for DATASET in "${DATASETS[@]}"; do
    BASENAME=$(basename "$DATASET" .csv)

    case "$BASENAME" in
        *"RequestForPayment"*|*"RFP"*)
            LR=2e-4; SIGMA=5e-4; RANK=128; ALPHA=256; CONAP_WIN=100;
        *)
            echo "WARNING: Dataset '$BASENAME' not found in lookup table. Using defaults." ;;
    esac

    echo ""
    echo "========================================================"
    echo "PROCESSING DATASET: $BASENAME"
    echo "CONAP-FM Config  -> LR: $LR | Sigma: $SIGMA | Rank: $RANK"
    echo "========================================================"

    for SEED in {40..44}; do
        echo "   > Running Seed: $SEED"

        # --- 1. CONAP-FM ---
        python -m Methods.CONAP_FM.run \
            --dataset "$DATASET" \
            --window-size $CONAP_WIN \
            --seed $SEED \
            --device $DEVICE \
            --model-name "arnir0/Tiny-LLM" \
            --lr $LR \
            --loss-variance-threshold $SIGMA \
            --lora-r $RANK \
            --lora-alpha $ALPHA \
            --data-split evaluation \
            --plots-dir "runs/CONAP_FM"
    done
done

echo ""
echo "========================================================"
echo "ALL DATASETS AND SEEDS COMPLETE"
echo "========================================================"