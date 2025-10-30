# EE 641 - Homework 3: Attention Mechanisms and Transformers

**Note: No changes have been in regards to code exection. Refer to the README.md in \problem1 and \problem2 for specifics.**

## Structure

- **`problem1/`** - Multi-head attention implementation and analysis
- **`problem2/`** - Positional encoding extrapolation experiments

See individual problem READMEs for detailed instructions:

- [Problem 1 README](problem1/README.md)
- [Problem 2 README](problem2/README.md)

## Requirements

```bash
pip install torch>=2.0.0 numpy>=1.24.0 matplotlib>=3.7.0 tqdm>=4.65.0
```

## Quick Start

### Problem 1: Multi-Head Attention

```bash
cd problem1
python generate_data.py --seed 641
python train.py
python analyze.py
```

### Problem 2: Positional Encoding

```bash
cd problem2
python generate_data.py --seed 641
python train.py --encoding sinusoidal
python train.py --encoding learned
python train.py --encoding none
python analyze.py
```

## Full Assignment

See the course website for complete assignment instructions, deliverables, and submission requirements.
