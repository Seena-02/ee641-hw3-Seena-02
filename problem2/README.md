# Problem 2: Positional Encoding and Length Extrapolation

Analysis of positional encoding strategies and their ability to generalize to sequences longer than training lengths.

## Task

Binary classification: determine if a sequence of integers is sorted in ascending order.

- **Training**: Sequences of length 8-16, integers 0-99
- **Testing**: Extrapolation to lengths 32, 64, 128, 256
- **Objective**: Compare length generalization of different encoding strategies

## Architecture

Transformer encoder for sequence classification:

- **Encoder**: Stack of self-attention layers with position-wise FFN
- **Positional Encoding**: Configurable (sinusoidal, learned, none)
- **Classification Head**: Global pooling + MLP → binary output

## Positional Encoding Strategies

1. **Sinusoidal**: Fixed encoding using sin/cos functions at different frequencies
2. **Learned**: Trainable embedding per position (limited to max_len)
3. **None**: No positional information (permutation-invariant baseline)

## File Structure

```
problem2/
├── generate_data.py           # Dataset generation
├── dataset.py                 # DataLoader for sorting task
├── positional_encoding.py     # Three encoding implementations
├── model.py                   # Transformer classifier
├── train.py                   # Training script
├── analyze.py                 # Extrapolation analysis
└── data/                      # Generated datasets
    └── extrapolation/         # Length-specific test sets
```

## Data Generation

```bash
python generate_data.py --seed 641 --generate-extrapolation
```

**Options:**
- `--seed`: Random seed (default: 641)
- `--train-size`: Training samples (default: 10000)
- `--val-size`: Validation samples (default: 2000)
- `--test-size`: Test samples (default: 2000)
- `--min-train-len`: Minimum training length (default: 8)
- `--max-train-len`: Maximum training length (default: 16)
- `--generate-extrapolation`: Generate length-specific test sets

Generates balanced datasets (50% sorted, 50% unsorted) at specified lengths.

## Training

Train models with each positional encoding type:

```bash
python train.py --encoding sinusoidal
python train.py --encoding learned
python train.py --encoding none
```

**Options:**
- `--encoding`: Positional encoding type (required)
- `--data-dir`: Data directory (default: data)
- `--output-dir`: Output directory (default: results)
- `--epochs`: Training epochs (default: 30)
- `--batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 1e-3)
- `--d-model`: Model dimension (default: 128)
- `--num-heads`: Attention heads (default: 4)
- `--num-layers`: Encoder layers (default: 4)
- `--d-ff`: Feed-forward dimension (default: 512)
- `--dropout`: Dropout rate (default: 0.1)

Each encoding type saves to `results/<encoding>/`:
- `best_model.pth`: Model weights
- `training_log.json`: Loss and accuracy history
- `training_curves.png`: Training visualization

## Analysis

```bash
python analyze.py --test-lengths 8 12 16 32 64 128 256
```

**Options:**
- `--data-dir`: Data directory (default: data)
- `--results-dir`: Models directory (default: results)
- `--output-dir`: Analysis output (default: results/extrapolation)
- `--batch-size`: Batch size (default: 32)
- `--test-lengths`: Sequence lengths to test
- `--device`: Device (default: cuda if available)

Generates:
- `extrapolation_results.json`: Accuracy at each length
- `extrapolation_curves.png`: Comparison plot
- `position_viz/`: Learned embedding visualizations
- `encoding_comparison/`: Side-by-side encoding patterns

## Implementation Requirements

### Core Components

**`positional_encoding.py`:**
- `SinusoidalPositionalEncoding`: Fixed sinusoidal patterns
- `LearnedPositionalEncoding`: Trainable position embeddings
- `NoPositionalEncoding`: Identity function (baseline)

**`model.py`:**
- `MultiHeadAttention`: Self-attention mechanism
- `TransformerEncoderLayer`: Attention + FFN block
- `SortingClassifier`: Full classification model

**`train.py`:**
- Training loop for each encoding type
- Binary cross-entropy loss
- Accuracy metric

**`analyze.py`:**
- Extrapolation testing at multiple lengths
- Comparative visualization
- Failure case analysis

## Model Configuration

Default hyperparameters:
- vocab_size = 100 (integers 0-99)
- d_model = 128
- num_heads = 4
- num_layers = 4
- d_ff = 512
- dropout = 0.1
- max_len = 300 (for position encoding)