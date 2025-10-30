"""
Training script for sequence-to-sequence addition model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import time

from model import Seq2SeqTransformer
from dataset import create_dataloaders, get_vocab_size
from attention import create_causal_mask
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import Seq2SeqTransformer


def compute_accuracy(outputs, targets, pad_token=0):
    """
    Compute sequence-level accuracy.

    Args:
        outputs: Model predictions [batch, seq_len, vocab_size]
        targets: Ground truth [batch, seq_len]
        pad_token: Padding token to ignore

    Returns:
        Accuracy (fraction of completely correct sequences)
    """
    predictions = outputs.argmax(dim=-1)  # [batch, seq_len]

    mask = (targets != pad_token)

    correct_tokens = ((predictions == targets) & mask).sum().item()
    total_tokens = mask.sum().item()
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    seq_correct = ((predictions == targets) | ~mask).all(dim=1)
    seq_acc = seq_correct.float().mean().item()

    return token_acc, seq_acc


# def train_epoch(model, dataloader, criterion, optimizer, device):
#     """
#     Train for one epoch.

#     Args:
#         model: Transformer model
#         dataloader: Training dataloader
#         criterion: Loss function
#         optimizer: Optimizer
#         device: Device to run on

#     Returns:
#         Average loss, average accuracy
#     """
#     model.train()
#     total_loss = 0
#     total_acc = 0
#     num_batches = 0

#     progress = tqdm(dataloader, desc="Training")
#     for batch in progress:
#         # Move to device
#         inputs = batch['input'].to(device)
#         targets = batch['target'].to(device)

#         # TODO: Prepare decoder input and output for teacher forcing
#         # Decoder input should be targets shifted right (exclude last token)
#         # Decoder output should be targets shifted left (exclude first token)

#         # TODO: Create causal mask for decoder (using shifted sequence length)
#         # TODO: Forward pass
#         # TODO: Compute loss
#         # Hint: Flatten for cross entropy - need 2D tensors
#         # TODO: Backward pass and optimization
#         # TODO: Compute accuracy

#         # ===== Prepare decoder input/output (teacher forcing) =====
#         # Decoder input: all tokens except the last
#         dec_input = targets[:, :-1]
#         # Decoder output (labels): all tokens except the first
#         dec_target = targets[:, 1:]

#         # ===== Create causal mask for decoder =====
#         seq_len = dec_input.size(1)
#         causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
#         # causal_mask[i, j] = True means position j is masked for position i

#         optimizer.zero_grad()

#         # ===== Forward pass =====
#         # Model should output logits of shape [batch, seq_len, vocab_size]
#         logits = model(inputs, dec_input, tgt_mask=causal_mask)

#         # ===== Compute loss =====
#         # Flatten logits and targets for cross-entropy
#         logits_flat = logits.reshape(-1, logits.size(-1))
#         targets_flat = dec_target.reshape(-1)
#         loss = criterion(logits_flat, targets_flat)

#         # ===== Backward and optimization =====
#         loss.backward()
#         optimizer.step()

#         # ===== Compute accuracy (ignoring padding) =====
#         preds = torch.argmax(logits, dim=-1)
#         non_pad_mask = dec_target.ne(0)
#         correct = (preds == dec_target) & non_pad_mask
#         acc = correct.sum().float() / non_pad_mask.sum().float()

#         # Update progress bar
#         progress.set_postfix({
#             'loss': f'{loss.item():.4f}',
#             'acc': f'{acc:.2%}'
#         })

#         total_loss += loss.item()
#         total_acc += acc
#         num_batches += 1

#     return total_loss / num_batches, total_acc / num_batches


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        # Move to device
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        # Teacher forcing: decoder input/output
        dec_input = targets[:, :-1]   # all but last
        dec_target = targets[:, 1:]   # all but first

        batch_size, tgt_len = dec_input.size()
        src_len = inputs.size(1)

        # Masks
        enc_pad_mask = inputs.ne(0)        # [batch, src_len]
        dec_pad_mask = dec_input.ne(0)     # [batch, tgt_len]
        causal_mask = create_causal_mask(tgt_len, device=device)  # [1,1,tgt_len,tgt_len]

        optimizer.zero_grad()

        # Forward pass
        logits = model(
            src=inputs,
            tgt=dec_input,
            src_mask=enc_pad_mask,
            tgt_mask=causal_mask,
        )  # [batch, seq_len, vocab_size]

        # Flatten for loss
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = dec_target.reshape(-1)

        # Compute loss
        loss = criterion(logits_flat, targets_flat)

        # Backward + gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Token accuracy (ignore padding)
        preds = logits.argmax(dim=-1)
        non_pad_mask = dec_target.ne(0)
        num_non_pad = non_pad_mask.sum().float()
        acc = ((preds == dec_target) & non_pad_mask).sum().float() / max(num_non_pad, 1)

        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1

        progress.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.2%}'
        })

    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_acc / max(num_batches, 1)

    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            # Decoder input/output
            dec_input = targets[:, :-1]
            dec_target = targets[:, 1:]

            batch_size, tgt_len = dec_input.size()
            src_len = inputs.size(1)

            # Masks
            enc_pad_mask = inputs.ne(0)
            dec_pad_mask = dec_input.ne(0)
            causal_mask = create_causal_mask(tgt_len, device=device)

            # Forward pass
            logits = model(
                src=inputs,
                tgt=dec_input,
                src_mask=enc_pad_mask,
                tgt_mask=causal_mask,
            )

            # Flatten for loss
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = dec_target.reshape(-1)
            loss = criterion(logits_flat, targets_flat)

            # Token accuracy
            preds = logits.argmax(dim=-1)
            non_pad_mask = dec_target.ne(0)
            num_non_pad = non_pad_mask.sum().float()
            acc = ((preds == dec_target) & non_pad_mask).sum().float() / max(num_non_pad, 1)

            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1

            progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2%}'
            })

    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_acc / max(num_batches, 1)
    return avg_loss, avg_acc



# def evaluate(model, dataloader, criterion, device):
#     """
#     Evaluate the model on validation or test set safely.

#     Returns:
#         avg_loss: Average loss over batches
#         avg_acc: Average token accuracy over batches
#     """
#     model.eval()
#     total_loss = 0.0
#     total_acc = 0.0
#     num_batches = 0

#     with torch.no_grad():
#         progress = tqdm(dataloader, desc="Evaluating", leave=False)
#         for batch in progress:
#             inputs = batch['input'].to(device)
#             targets = batch['target'].to(device)

#             # Prepare decoder input/output
#             dec_input = targets[:, :-1]
#             dec_target = targets[:, 1:]

#             seq_len = dec_input.size(1)
#             causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

#             # Forward pass
#             logits = model(inputs, dec_input, tgt_mask=causal_mask)

#             # Flatten for loss
#             logits_flat = logits.reshape(-1, logits.size(-1))
#             targets_flat = dec_target.reshape(-1)

#             # Loss
#             loss = criterion(logits_flat, targets_flat)

#             # Safe accuracy
#             preds = logits.argmax(dim=-1)
#             non_pad_mask = dec_target.ne(0)
#             num_non_pad = non_pad_mask.sum().float()
#             if num_non_pad == 0:
#                 acc = torch.tensor(0.0, device=device)
#             else:
#                 acc = ((preds == dec_target) & non_pad_mask).sum().float() / num_non_pad

#             total_loss += loss.item()
#             total_acc += acc.item()
#             num_batches += 1

#             progress.set_postfix({
#                 'loss': f'{loss.item():.4f}',
#                 'acc': f'{acc:.2%}'
#             })

#     avg_loss = total_loss / max(num_batches, 1)
#     avg_acc = total_acc / max(num_batches, 1)

#     return avg_loss, avg_acc



# def evaluate(model, dataloader, criterion, device):
#     """
#     Evaluate model on validation/test set.

#     Args:
#         model: Transformer model
#         dataloader: Evaluation dataloader
#         criterion: Loss function
#         device: Device to run on

#     Returns:
#         Average loss, average accuracy
#     """
#     model.eval()
#     total_loss = 0
#     total_acc = 0
#     num_batches = 0

#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Evaluating"):
#             inputs = batch['input'].to(device)
#             targets = batch['target'].to(device)

#             # ===== Prepare decoder input/output =====
#             dec_input = targets[:, :-1]      # all but last token
#             dec_target = targets[:, 1:]      # all but first token

#             # ===== Create causal mask =====
#             seq_len = dec_input.size(1)
#             causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

#             # ===== Forward pass =====
#             logits = model(inputs, dec_input, tgt_mask=causal_mask)

#             # ===== Compute loss =====
#             logits_flat = logits.reshape(-1, logits.size(-1))
#             targets_flat = dec_target.reshape(-1)
#             loss = criterion(logits_flat, targets_flat)

#             # ===== Compute accuracy (ignore padding) =====
#             preds = torch.argmax(logits, dim=-1)
#             non_pad_mask = dec_target.ne(0)
#             correct = (preds == dec_target) & non_pad_mask
#             acc = correct.sum().float() / non_pad_mask.sum().float()


#             total_loss += loss.item()
#             total_acc += acc
#             num_batches += 1

#     return total_loss / num_batches, total_acc / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train addition transformer')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of encoder/decoder layers')
    parser.add_argument('--d-ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'mps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size
    )

    # Create model
    vocab_size = get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout
    ).to(args.device)

    # TODO: Initialize optimizer (Adam recommended)
    # TODO: Initialize learning rate scheduler (ReduceLROnPlateau recommended)
    # TODO: Initialize loss function (use nn.CrossEntropyLoss)

    optimizer = optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training loop
    best_val_acc = -1
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'val_loss': [],
        'val_acc': [],

    }

    print(f"Starting training for {args.epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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


        training_history['train_loss'].append(train_loss.item() if torch.is_tensor(train_loss) else float(train_loss))
        training_history['train_acc'].append(train_acc.item() if torch.is_tensor(train_acc) else float(train_acc))
        training_history['val_loss'].append(val_loss.item() if torch.is_tensor(val_loss) else float(val_loss))
        training_history['val_acc'].append(val_acc.item() if torch.is_tensor(val_acc) else float(val_acc))


        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f"Saved best model with validation accuracy: {val_acc:.2%}")

    # Test final model
    #model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    model.load_state_dict(torch.load(output_dir / 'best_model.pth', weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}")

    # Save training history

    training_history['test_loss'] = test_loss.item() if torch.is_tensor(test_loss) else float(test_loss)
    training_history['test_acc'] = test_acc.item() if torch.is_tensor(test_acc) else float(test_acc)

    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\nTraining complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()