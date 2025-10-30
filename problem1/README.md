# Problem 1: Multi-Head Attention - Multi-Digit Addition

Implementation of scaled dot-product attention and multi-head attention mechanisms for a sequence-to-sequence addition task.

## Architecture

The model is a standard encoder-decoder transformer:

- **Encoder**: Multi-head self-attention + position-wise FFN with residual connections
- **Decoder**: Masked self-attention + encoder-decoder cross-attention + FFN with residual connections
- **Positional Encoding**: Sinusoidal encoding for position information
- **Task**: Add two 3-digit numbers with carry propagation

Input format: `[d1, d2, d3, +, d4, d5, d6]` → Output: `[d7, d8, d9, d10]` (4 digits, zero-padded)

Vocabulary: `{0, 1, 2, ..., 9, +, PAD}` (12 tokens total)

## File Structure

```
problem1/
├── generate_data.py       # Dataset generation
├── dataset.py             # DataLoader and preprocessing
├── attention.py           # Attention mechanisms
├── model.py               # Transformer architecture
├── train.py               # Training loop
├── analyze.py             # Attention analysis and visualization
└── data/                  # Generated datasets (after running generate_data.py)
```

## Data Generation

```bash
python generate_data.py --num-digits 3 --seed 641
```

**Options:**
- `--num-digits`: Digits per operand (default: 3)
- `--seed`: Random seed (default: 641)
- `--train-size`: Training samples (default: 10000)
- `--val-size`: Validation samples (default: 2000)
- `--test-size`: Test samples (default: 2000)
- `--output-dir`: Output directory (default: data)

Generates `train.json`, `val.json`, `test.json` with unique addition problems.

## Training

```bash
python train.py --epochs 50 --batch-size 64 --lr 1e-3
```

**Options:**
- `--data-dir`: Data directory (default: data)
- `--output-dir`: Output directory (default: results)
- `--epochs`: Training epochs (default: 50)
- `--batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 1e-3)
- `--d-model`: Model dimension (default: 128)
- `--num-heads`: Attention heads (default: 4)
- `--num-layers`: Encoder/decoder layers (default: 2)
- `--d-ff`: Feed-forward dimension (default: 512)
- `--dropout`: Dropout rate (default: 0.1)
- `--device`: Device (default: cuda if available)
- `--seed`: Random seed (default: 42)

Saves `best_model.pth` and `training_log.json` to output directory.

## Analysis

```bash
python analyze.py --model-path results/best_model.pth
```

**Options:**
- `--model-path`: Path to trained model (required)
- `--data-dir`: Data directory (default: data)
- `--output-dir`: Analysis output directory (default: results)
- `--batch-size`: Batch size (default: 32)
- `--num-samples`: Samples to analyze (default: 100)
- `--device`: Device (default: cuda if available)

Generates:
- `attention_patterns/`: Attention heatmaps per head
- `head_analysis/`: Head specialization statistics and ablation results
- Example predictions with attention visualizations

## Implementation Requirements

### Core Components

**`attention.py`:**
- `scaled_dot_product_attention()`: Attention(Q,K,V) = softmax(QK^T / √d_k)V
- `MultiHeadAttention`: Parallel attention with head splitting/combining

**`model.py`:**
- `PositionalEncoding`: Sinusoidal position embeddings
- `FeedForward`: Two-layer position-wise network
- `EncoderLayer`: Self-attention + FFN with residuals
- `DecoderLayer`: Masked self-attention + cross-attention + FFN
- `Seq2SeqTransformer`: Full encoder-decoder model

**`train.py`:**
- Training loop with loss computation and optimization
- Sequence-level accuracy metric (exact match)
- Model checkpointing and logging

**`analyze.py`:**
- Attention weight extraction and visualization
- Head specialization analysis
- Ablation study for head importance

## Model Hyperparameters

Default configuration:
- d_model = 128
- num_heads = 4 (d_k = 32 per head)
- num_layers = 2 (encoder and decoder)
- d_ff = 512
- dropout = 0.1
- vocab_size = 12

Total parameters: ~500K (varies with configuration)