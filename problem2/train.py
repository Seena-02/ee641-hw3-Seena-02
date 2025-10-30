"""
Training script for sorting classifier with different positional encodings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import create_model
from dataset import create_dataloaders


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.

    Args:
        model: Classification model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on

    Returns:
        Average loss, average accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        # Move to device
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        masks = batch['mask'].to(device)

        # TODO: Forward pass
        # TODO: Compute loss
        # TODO: Backward pass
        # TODO: Update weights
        # Forward pass
        logits = model(sequences)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute accuracy
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        progress.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.2%}'
        })

        total_loss += loss.item()

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model.

    Args:
        model: Classification model
        dataloader: Evaluation dataloader
        criterion: Loss function
        device: Device to run on

    Returns:
        Average loss, accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            masks = batch['mask'].to(device)

            # TODO: Forward pass
            # TODO: Compute loss
            # Forward pass
            logits = model(sequences)

            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Compute accuracy
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()

    return total_loss / len(dataloader), correct / total


def plot_training_curves(history, save_path):
    """
    Plot training curves.

    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Train')
    ax1.plot(epochs, history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], label='Train')
    ax2.plot(epochs, history['val_acc'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train sorting classifier')
    parser.add_argument('--encoding', choices=['sinusoidal', 'learned', 'none'],
                        required=True, help='Positional encoding type')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--d-ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory for this encoding type
    output_dir = Path(args.output_dir) / args.encoding
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training with {args.encoding} positional encoding")
    print(f"Output directory: {output_dir}")

    # Load data
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size
    )

    # Create model
    model = create_model(
        encoding_type=args.encoding,
        vocab_size=101,  # 0-99 for integers + 100 for padding
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout
    ).to(args.device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # TODO: Initialize optimizer (Adam recommended)
    # TODO: Initialize loss function (use nn.CrossEntropyLoss)
    # TODO: Initialize learning rate scheduler (ReduceLROnPlateau recommended)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0

    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )

        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, args.device
        )

        # TODO: Step learning rate scheduler (pass val_loss)
        scheduler.step(val_loss)

        # Log results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f"Saved best model with validation accuracy: {val_acc:.2%}")

    # Test final model
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}")

    # Save training history
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    plot_training_curves(history, output_dir / 'training_curves.png')

    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()