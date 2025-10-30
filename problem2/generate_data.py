#!/usr/bin/env python3
"""
Generate sorting detection dataset.

Creates datasets for binary classification task: determining if a sequence
of integers is sorted in ascending order.
"""

import json
import random
import argparse
from pathlib import Path


def generate_sorted_sequence(length, min_val=0, max_val=99):
    """
    Generate a sorted sequence of integers.

    Args:
        length: Sequence length
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        List of sorted integers
    """
    # Generate random integers and sort
    sequence = sorted([random.randint(min_val, max_val) for _ in range(length)])
    return sequence


def generate_unsorted_sequence(length, min_val=0, max_val=99):
    """
    Generate an unsorted sequence of integers.

    Ensures the sequence is definitely not sorted by performing swaps.

    Args:
        length: Sequence length
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        List of unsorted integers
    """
    # Start with a sorted sequence and shuffle
    sequence = generate_sorted_sequence(length, min_val, max_val)

    # Perform multiple swaps to ensure it's not sorted
    num_swaps = max(2, length // 3)
    for _ in range(num_swaps):
        i, j = random.sample(range(length), 2)
        sequence[i], sequence[j] = sequence[j], sequence[i]

    # Verify it's not sorted
    if sequence == sorted(sequence):
        # If accidentally still sorted, force unsort
        if length > 1:
            sequence[0], sequence[-1] = sequence[-1], sequence[0]

    return sequence


def is_sorted(sequence):
    """Check if sequence is sorted in ascending order."""
    return all(sequence[i] <= sequence[i+1] for i in range(len(sequence)-1))


def generate_sample(min_len, max_len, sorted_prob=0.5):
    """
    Generate a single sample.

    Args:
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        sorted_prob: Probability of generating sorted sequence

    Returns:
        Dictionary with sequence and label
    """
    length = random.randint(min_len, max_len)

    if random.random() < sorted_prob:
        sequence = generate_sorted_sequence(length)
        label = 1
    else:
        sequence = generate_unsorted_sequence(length)
        label = 0

    # Verify label correctness
    assert is_sorted(sequence) == (label == 1), "Label mismatch"

    return {
        'sequence': sequence,
        'is_sorted': label,
        'length': length
    }


def generate_dataset(num_samples, min_len, max_len, sorted_prob=0.5, seed=None):
    """
    Generate complete dataset.

    Args:
        num_samples: Number of samples
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        sorted_prob: Probability of sorted sequences
        seed: Random seed

    Returns:
        List of samples
    """
    if seed is not None:
        random.seed(seed)

    dataset = []

    for _ in range(num_samples):
        sample = generate_sample(min_len, max_len, sorted_prob)
        dataset.append(sample)

    # Print statistics
    num_sorted = sum(s['is_sorted'] for s in dataset)
    avg_length = sum(s['length'] for s in dataset) / len(dataset)

    print(f"  Generated {num_samples} samples")
    print(f"  Sorted: {num_sorted} ({100*num_sorted/num_samples:.1f}%)")
    print(f"  Unsorted: {num_samples - num_sorted} ({100*(num_samples - num_sorted)/num_samples:.1f}%)")
    print(f"  Average length: {avg_length:.1f}")

    return dataset


def generate_extrapolation_sets(base_seed=641):
    """
    Generate test sets for extrapolation analysis.

    Creates test sets at specific lengths: 32, 64, 128, 256

    Args:
        base_seed: Base random seed

    Returns:
        Dictionary of test sets by length
    """
    extrapolation_sets = {}
    test_lengths = [8, 12, 16, 32, 64, 128, 256]

    for test_len in test_lengths:
        print(f"\nGenerating extrapolation set for length {test_len}...")

        # Generate balanced dataset at this specific length
        dataset = generate_dataset(
            num_samples=500,
            min_len=test_len,
            max_len=test_len,
            sorted_prob=0.5,
            seed=base_seed + test_len
        )

        extrapolation_sets[test_len] = dataset

    return extrapolation_sets


def save_dataset(dataset, filepath):
    """Save dataset to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset)} samples to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Generate sorting detection dataset')
    parser.add_argument('--seed', type=int, default=641,
                        help='Random seed')
    parser.add_argument('--train-size', type=int, default=10000,
                        help='Number of training samples')
    parser.add_argument('--val-size', type=int, default=2000,
                        help='Number of validation samples')
    parser.add_argument('--test-size', type=int, default=2000,
                        help='Number of test samples')
    parser.add_argument('--min-train-len', type=int, default=8,
                        help='Minimum training sequence length')
    parser.add_argument('--max-train-len', type=int, default=16,
                        help='Maximum training sequence length')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory')
    parser.add_argument('--generate-extrapolation', action='store_true',
                        help='Generate extrapolation test sets')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("Generating sorting detection datasets...")
    print(f"Training lengths: {args.min_train_len}-{args.max_train_len}")

    # Training set
    print("\nTraining set:")
    train_data = generate_dataset(
        args.train_size,
        args.min_train_len,
        args.max_train_len,
        sorted_prob=0.5,
        seed=args.seed
    )
    save_dataset(train_data, output_dir / 'train.json')

    # Validation set
    print("\nValidation set:")
    val_data = generate_dataset(
        args.val_size,
        args.min_train_len,
        args.max_train_len,
        sorted_prob=0.5,
        seed=args.seed + 1
    )
    save_dataset(val_data, output_dir / 'val.json')

    # Test set (same length range as training)
    print("\nTest set:")
    test_data = generate_dataset(
        args.test_size,
        args.min_train_len,
        args.max_train_len,
        sorted_prob=0.5,
        seed=args.seed + 2
    )
    save_dataset(test_data, output_dir / 'test.json')

    # Extrapolation test sets
    if args.generate_extrapolation:
        print("\nGenerating extrapolation test sets...")
        extrap_sets = generate_extrapolation_sets(args.seed + 1000)

        # Save each extrapolation set
        extrap_dir = output_dir / 'extrapolation'
        extrap_dir.mkdir(parents=True, exist_ok=True)

        for length, dataset in extrap_sets.items():
            save_dataset(dataset, extrap_dir / f'test_len_{length}.json')

    # Print examples
    print("\nExample samples:")
    for i in range(4):
        sample = train_data[i]
        seq_str = str(sample['sequence'][:10]) + ('...' if len(sample['sequence']) > 10 else '')
        print(f"  Sequence: {seq_str}")
        print(f"  Length: {sample['length']}, Sorted: {sample['is_sorted']}")
        print()


if __name__ == '__main__':
    main()