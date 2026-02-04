# QLoRA Fine-Tuning for Product Price Prediction

Fine-tune an open-source LLM with **QLoRA** to predict product prices from short descriptions, then evaluate against baselines using **MAE**.

This project is structured:
- Dataset loading
- Prompt formatting
- QLoRA training (4-bit quantization + LoRA adapters)
- Evaluation + inference
- Adapter saving

## Repo contents

- `qlora-price-prediction-finetune.ipynb`  
  End-to-end workflow (train, evaluate, run inference)

## Setup

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U transformers datasets accelerate peft trl bitsandbytes evaluate python-dotenv plotly
```

### 2) Configure environment variables

Create a `.env` file:
```bash
HF_TOKEN="YOUR_HF_TOKEN"
BASE_MODEL="meta-llama/Llama-3.2-3B"
DATASET_NAME="ed-donner/items_full"
TEXT_FIELD="text"
PRICE_FIELD="price"

# Optional
LITE_MODE="false"
MAX_TRAIN_SAMPLES="0"
MAX_EVAL_SAMPLES="0"
OUTPUT_DIR="./outputs_price_qlora"
```

Notes:
- Some models (for example, Llama family) require accepting a license on Hugging Face.
- Training requires a GPU. Colab is the simplest option.

## How to run

1. Open the notebook in Jupyter or VSCode.
2. Run cells top to bottom.
3. To train, uncomment:
   - `trainer.train()`
4. To evaluate, run:
   - `evaluate_model(eval_ds, n=...)`
