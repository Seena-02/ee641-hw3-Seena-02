#!/usr/bin/env python3
"""
Generate multi-digit addition dataset.

Creates training, validation, and test sets for the addition task
where the model learns to add two n-digit numbers with carry propagation.
"""

import json
import random
import argparse
from pathlib import Path


def generate_addition_sample(num_digits, base=10):
    """
    Generate a single addition problem.

    Args:
        num_digits: Number of digits in each operand
        base: Numerical base (default 10)

    Returns:
        dict with 'input' and 'target' sequences
    """
    # Generate two random n-digit numbers
    max_val = base ** num_digits - 1
    min_val = base ** (num_digits - 1)  # Ensure n digits

    num1 = random.randint(min_val, max_val)
    num2 = random.randint(min_val, max_val)

    # Compute sum
    result = num1 + num2

    # Convert to digit sequences
    # Input format: [digit1, digit2, ..., +, digit1, digit2, ...]
    input_seq = []

    # First number digits (most significant first)
    for digit in str(num1):
        input_seq.append(int(digit))

    # Add operator token (using base as the '+' token)
    input_seq.append(base)

    # Second number digits
    for digit in str(num2):
        input_seq.append(int(digit))

    # Target: result padded to num_digits+1 (to handle carry)
    target_str = str(result).zfill(num_digits + 1)
    target_seq = [int(digit) for digit in target_str]

    return {
        'input': input_seq,
        'target': target_seq,
        'num1': num1,
        'num2': num2,
        'result': result
    }


def generate_dataset(num_samples, num_digits, seed=None):
    """
    Generate complete dataset.

    Args:
        num_samples: Number of samples to generate
        num_digits: Number of digits in each operand
        seed: Random seed for reproducibility

    Returns:
        List of samples
    """
    if seed is not None:
        random.seed(seed)

    dataset = []
    seen = set()

    while len(dataset) < num_samples:
        sample = generate_addition_sample(num_digits)

        # Ensure uniqueness based on the actual problem
        problem_key = (sample['num1'], sample['num2'])
        if problem_key not in seen:
            seen.add(problem_key)
            # Only keep input and target for the dataset
            dataset.append({
                'input': sample['input'],
                'target': sample['target']
            })

    return dataset


def save_dataset(dataset, filepath):
    """Save dataset to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset)} samples to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Generate addition dataset')
    parser.add_argument('--num-digits', type=int, default=3,
                        help='Number of digits in each operand')
    parser.add_argument('--seed', type=int, default=641,
                        help='Random seed')
    parser.add_argument('--train-size', type=int, default=10000,
                        help='Number of training samples')
    parser.add_argument('--val-size', type=int, default=2000,
                        help='Number of validation samples')
    parser.add_argument('--test-size', type=int, default=2000,
                        help='Number of test samples')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for datasets')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Generate datasets with different seeds for train/val/test
    print(f"Generating {args.num_digits}-digit addition datasets...")

    # Training set
    train_data = generate_dataset(
        args.train_size, args.num_digits, seed=args.seed
    )
    save_dataset(train_data, output_dir / 'train.json')

    # Validation set
    val_data = generate_dataset(
        args.val_size, args.num_digits, seed=args.seed + 1
    )
    save_dataset(val_data, output_dir / 'val.json')

    # Test set
    test_data = generate_dataset(
        args.test_size, args.num_digits, seed=args.seed + 2
    )
    save_dataset(test_data, output_dir / 'test.json')

    # Print some examples
    print("\nExample samples:")
    for i in range(3):
        sample = train_data[i]
        input_str = ' '.join(map(str, sample['input']))
        target_str = ''.join(map(str, sample['target']))
        print(f"  Input: {input_str} -> Target: {target_str}")

    print("\nDataset statistics:")
    print(f"  Vocabulary size: {10 + 1 + 1}")  # digits 0-9, operator token, padding token
    print(f"  Input length: {2 * args.num_digits + 1}")
    print(f"  Output length: {args.num_digits + 1}")


if __name__ == '__main__':
    main()