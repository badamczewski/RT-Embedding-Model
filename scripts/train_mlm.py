"""
MLM Pre-training script for the embedding model.

Trains the model using Masked Language Modeling (MLM) with BERT-style masking.
"""

import argparse
import sys
import os
import random
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import EmbeddingModel, SentencePieceTokenizer, TrainingLogger


class MLMDataset(Dataset):
    """Dataset for Masked Language Modeling."""
    
    def __init__(self, texts: List[str], tokenizer: SentencePieceTokenizer, 
                 context_window: int = 16, mask_prob: float = 0.15):
        """
        Initialize MLM dataset.
        
        Args:
            texts: List of text strings (one per line)
            tokenizer: SentencePieceTokenizer instance
            context_window: Maximum sequence length
            mask_prob: Probability of masking a token (default: 0.15)
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.mask_prob = mask_prob
        self.mask_id = tokenizer.mask_id  # Use proper MASK token ID
        self.pad_id = tokenizer.pad_id
        
        # Encode all texts
        self.encoded_texts = []
        for text in texts:
            token_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
            if len(token_ids) > 0:
                self.encoded_texts.append(token_ids)
    
    def __len__(self):
        return len(self.encoded_texts)
    
    def __getitem__(self, idx):
        """Get a sample with MLM masking applied."""
        token_ids = self.encoded_texts[idx].copy()
        
        # Truncate or pad to context_window
        if len(token_ids) > self.context_window:
            token_ids = token_ids[:self.context_window]
        elif len(token_ids) < self.context_window:
            token_ids = token_ids + [self.pad_id] * (self.context_window - len(token_ids))
        
        #
        # Apply BERT-style masking
        #
        masked_ids, labels = self._apply_masking(token_ids)
        
        return {
            'input_ids': torch.tensor(masked_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            # attention_mask polarity (True = pad)
            'attention_mask': torch.tensor([tid == self.pad_id for tid in masked_ids], dtype=torch.bool)
        }
    
    def _apply_masking(self, token_ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        Apply BERT-style masking.
        
        Strategy:
        - Select 15% of tokens randomly
        - Of selected tokens:
          - 80% -> Replace with [MASK] token
          - 10% -> Replace with random token
          - 10% -> Keep original
        
        Args:
            token_ids: Original token IDs
        
        Returns:
            Tuple of (masked_token_ids, labels)
            - labels: -100 for non-masked positions, original token_id for masked positions
        """
        labels = [-100] * len(token_ids)
        masked_ids = token_ids.copy()
        
        # Find positions that can be masked (not padding)
        candidate_positions = [i for i, tid in enumerate(token_ids) if tid != self.pad_id]
        
        if not candidate_positions:
            return masked_ids, labels
        
        #
        # Exclude special tokens - don't mask special tokens
        #
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []))
        candidate_positions = [i for i, tid in enumerate[int](token_ids) if tid not in special_ids]
        if not candidate_positions:
            return masked_ids, labels
        #
        # Select mask_prob% of tokens to mask
        #
        num_to_mask = max(1, int(len(candidate_positions) * self.mask_prob))
        positions_to_mask = random.sample(candidate_positions, min(num_to_mask, len(candidate_positions)))
        #
        # Select valid random ids - don't use special tokens
        #
        valid_random_ids = [i for i in range(len(self.tokenizer)) if i not in special_ids]
        #
        # Mask tokens
        #
        for pos in positions_to_mask:
            original_id = token_ids[pos]
            labels[pos] = original_id
            #
            # BERT-style masking: 80% -> MASK, 10% -> random, 10% -> unchanged
            #
            rand = random.random()
            if rand < 0.8:
                masked_ids[pos] = self.mask_id
            elif rand < 0.9:
                masked_ids[pos] = random.choice(valid_random_ids)
            # else: keep original
        
        return masked_ids, labels


def load_corpus(corpus_path: str) -> List[str]:
    """Load corpus from file (one sentence per line)."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts


def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps, 
                training_logger, epoch, step_counter, writer=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_steps = 0

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Calculate total steps per epoch for progress tracking
    total_steps_per_epoch = len(dataloader) // gradient_accumulation_steps
    if len(dataloader) % gradient_accumulation_steps != 0:
        total_steps_per_epoch += 1
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Check if there are any valid labels (not all -100)
        valid_labels = (labels != -100).any()
        if not valid_labels:
            # Skip batches with no valid labels
            continue
        
        # Forward pass
        logits = model(input_ids, attention_mask=attention_mask)
        
        # Check for NaN or Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            training_logger.log_warning("NaN or Inf detected in training logits, skipping batch")
            continue
        
        # Compute loss (only on masked positions)
        # Reshape for cross-entropy: (batch_size * seq_len, vocab_size)
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        loss = loss_fn(logits_flat, labels_flat)
        
        # Check if loss is valid
        if torch.isnan(loss) or torch.isinf(loss):
            training_logger.log_warning("NaN or Inf loss detected in training, skipping batch")
            continue
        
        # Scale loss by gradient accumulation steps
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Calculate gradient norm before clipping (only when we're about to step)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            
            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
            # Update model parameters
            optimizer.step()
            optimizer.zero_grad()
            # Update learning rate scheduler
            scheduler.step()

            # Logging
            step_loss = loss.item() * gradient_accumulation_steps  # Unscale for logging
            total_loss += step_loss
            num_steps += 1
            step_counter[0] += 1
            
            if num_steps % 10 == 0:  # Log every 10 steps
                current_lr = optimizer.param_groups[0]['lr']
                
                if torch.cuda.is_available():
                    memory_alloc = torch.cuda.memory_allocated(device) / 1e6
                    writer.add_scalar('System/GPU_Memory_MB', memory_alloc, step_counter[0])

                # Calculate perplexity
                perplexity = torch.exp(torch.tensor(step_loss)).item()
                
                # Calculate token-level accuracy (only on masked positions)
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    valid_mask = (labels != -100)
                    if valid_mask.any():
                        token_accuracy = (preds[valid_mask] == labels[valid_mask]).float().mean().item()
                    else:
                        token_accuracy = 0.0
                
                # Log step progress
                step_progress = f"Step {num_steps}/{total_steps_per_epoch}"
                
                training_logger.log_train_step(
                    step=step_counter[0],
                    loss=step_loss,
                    lr=current_lr,
                    epoch=epoch
                )
                # Log additional metrics
                training_logger.log_info(
                    f"{step_progress} | Perplexity: {perplexity:.4f} | "
                    f"Token Accuracy: {token_accuracy:.4f} | Grad Norm: {grad_norm:.4f}"
                )
                
                # Log to TensorBoard
                if writer is not None:
                    writer.add_scalar('Train/Loss', step_loss, step_counter[0])
                    writer.add_scalar('Train/LearningRate', current_lr, step_counter[0])
                    writer.add_scalar('Train/Perplexity', perplexity, step_counter[0])
                    writer.add_scalar('Train/TokenAccuracy', token_accuracy, step_counter[0])
                    writer.add_scalar('Train/GradientNorm', grad_norm.item(), step_counter[0])
    
    # Handle remaining gradients if any
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        # Calculate gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        # Gradient clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / max(num_steps, 1)  # safe division
    return avg_loss


def validate(model, dataloader, device, training_logger, epoch, writer=None):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_token_accuracy = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Check if there are any valid labels (not all -100)
            valid_labels = (labels != -100).any()
            if not valid_labels:
                # Skip batches with no valid labels
                continue
            
            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask)
            
            # Check for NaN or Inf in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                training_logger.log_warning("NaN or Inf detected in validation logits, skipping batch")
                continue
            
            # Compute loss
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            
            loss = loss_fn(logits_flat, labels_flat)
            
            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                training_logger.log_warning(f"NaN or Inf loss detected in validation, skipping batch")
                continue
            
            # Calculate token-level accuracy (only on masked positions)
            preds = logits.argmax(dim=-1)
            valid_mask = (labels != -100)
            if valid_mask.any():
                batch_token_accuracy = (preds[valid_mask] == labels[valid_mask]).float().mean().item()
                total_token_accuracy += batch_token_accuracy
            
            loss_value = loss.item()
            total_loss += loss_value
            num_batches += 1
    
    if num_batches == 0:
        training_logger.log_warning("No valid validation batches found (all labels were -100 or invalid)")
        return float('nan')
    
    avg_loss = total_loss / max(num_batches, 1)  # safe division
    avg_token_accuracy = total_token_accuracy / max(num_batches, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    training_logger.log_validation(epoch=epoch, val_loss=avg_loss)
    training_logger.log_info(
        f"Validation | Perplexity: {perplexity:.4f} | Token Accuracy: {avg_token_accuracy:.4f}"
    )
    
    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('Validation/Loss', avg_loss, epoch)
        writer.add_scalar('Validation/Perplexity', perplexity, epoch)
        writer.add_scalar('Validation/TokenAccuracy', avg_token_accuracy, epoch)
    
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, training_logger):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'learning_rate': optimizer.param_groups[0]['lr'],
    }
    
    torch.save(checkpoint, checkpoint_path)
    training_logger.log_checkpoint(
        epoch=epoch,
        checkpoint_path=checkpoint_path,
        metrics={'loss': loss}
    )
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description='Train embedding model with MLM pre-training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--tokenizer', type=str, default='models/tokenizer.model',
                       help='Path to tokenizer model')
    parser.add_argument('--corpus', type=str, default='data/corpus.txt',
                       help='Path to training corpus')
    parser.add_argument('--output_dir', type=str, default='models/checkpoints',
                       help='Output directory for checkpoints')
    
    # Model arguments
    parser.add_argument('--context_window', type=int, default=16,
                       help='Context window size')
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--validation_split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--mask_prob', type=float, default=0.15,
                       help='Masking probability')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Get project root directory (parent of scripts/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Resolve relative paths to project root
    if not os.path.isabs(args.tokenizer):
        args.tokenizer = os.path.join(project_root, args.tokenizer)
    if not os.path.isabs(args.corpus):
        args.corpus = os.path.join(project_root, args.corpus)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize training logger (logs directory relative to project root)
    log_dir = os.path.join(project_root, "logs")
    training_logger = TrainingLogger(log_dir=log_dir, experiment_name="mlm_pretraining")
    
    # Initialize TensorBoard writer
    tb_log_dir = os.path.join(project_root, "runs", "mlm_pretraining")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    training_logger.log_info(f"TensorBoard logs will be saved to: {tb_log_dir}")
    training_logger.log_info(f"View with: tensorboard --logdir {tb_log_dir}")
    
    # Log configuration
    config = {
        'tokenizer': args.tokenizer,
        'corpus': args.corpus,
        'output_dir': args.output_dir,
        'context_window': args.context_window,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'validation_split': args.validation_split,
        'mask_prob': args.mask_prob,
        'seed': args.seed,
        'device': str(device),
    }
    training_logger.log_config(config)
    
    # Load tokenizer
    training_logger.log_info(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = SentencePieceTokenizer(model_path=args.tokenizer)
    vocab_size = len(tokenizer)
    training_logger.log_info(f"Tokenizer loaded. Vocabulary size: {vocab_size}")
    
    # Load corpus
    training_logger.log_info(f"Loading corpus from {args.corpus}")
    texts = load_corpus(args.corpus)
    training_logger.log_info(f"Loaded {len(texts)} sentences")
    
    # Train/validation split
    split_idx = int(len(texts) * (1 - args.validation_split))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    training_logger.log_info(f"Train: {len(train_texts)} sentences, Val: {len(val_texts)} sentences")
    
    # Create datasets
    train_dataset = MLMDataset(train_texts, tokenizer, args.context_window, args.mask_prob)
    val_dataset = MLMDataset(val_texts, tokenizer, args.context_window, args.mask_prob)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    training_logger.log_info("Initializing model...")
    model = EmbeddingModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        context_window=args.context_window,
        hidden_dim=args.hidden_dim,
        is_masked_language=True,  # Enable MLM head
        use_temperature=False  # Disable temperature during pre-training
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    training_logger.log_info(f"Model initialized. Parameters: {num_params:,} (Trainable: {trainable_params:,})")
    
    # Verify that everything is set up correctly
    training_logger.log_info("Verify that everything is set up correctly")
    training_logger.log_info(f"mask_id: {tokenizer.mask_id}")
    training_logger.log_info(f"mask embedding weights (first 5 dims): {model.token_embedding.weight[tokenizer.mask_id][:5]}")
    training_logger.log_info(f"mask embedding weights (last 5 dims): {model.token_embedding.weight[tokenizer.mask_id][-5:]}")
    


    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # Learning rate scheduler
    total_steps = math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    warmup_steps = int(0.05 * total_steps)  # 5% warmup

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay after warmup
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    training_logger.log_info("Learning rate scheduler initialized.")
    training_logger.log_info(f"Learning rate: {args.learning_rate}")
    training_logger.log_info(f"Learning rate scheduler: {scheduler}")

    # Training loop
    training_logger.log_info("Starting training...")
    step_counter = [0]  # Use list to allow modification in nested function
    
    try:
        #
        # Early stopping parameters:
        # - patience: number of epochs without improvement allowed
        # - best_val_loss: initialize with a large value for the best validation loss
        # - epochs_no_improve: initialize the number of epochs without improvement
        #
        patience = 3
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(1, args.epochs + 1):
            training_logger.log_info(f"\n{'='*60}")
            training_logger.log_info(f"Epoch {epoch}/{args.epochs}")
            training_logger.log_info(f"{'='*60}")
            
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, device,
                args.gradient_accumulation_steps, training_logger, epoch, step_counter, writer
            )
            
            # Validate
            val_loss = validate(model, val_loader, device, training_logger, epoch, writer)
            
            # Log epoch summary
            current_lr = optimizer.param_groups[0]['lr']
            training_logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                lr=current_lr
            )
            
            # Log epoch summaries to TensorBoard
            writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            if not torch.isnan(torch.tensor(val_loss)):
                writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
            writer.add_scalar('Epoch/LearningRate', current_lr, epoch)
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, val_loss, args.output_dir, training_logger)

            # Early stopping
            if val_loss < best_val_loss - 1e-5:  # small delta for float stability
                best_val_loss = val_loss
                epochs_no_improve = 0
                training_logger.log_info(f"Validation improved to {val_loss:.6f}. Reset patience counter.")
            else:
                epochs_no_improve += 1
                training_logger.log_info(
                    f"No improvement for {epochs_no_improve} epoch(s) "
                    f"(best: {best_val_loss:.6f})."
                )
                if epochs_no_improve >= patience:
                    training_logger.log_warning(
                        f"Early stopping triggered after {patience} epochs without improvement."
                    )
                    break

    except KeyboardInterrupt:
        training_logger.log_info("Training interrupted by user.")
        training_logger.log_info("Final checkpoint saved to: {args.output_dir}")
        save_checkpoint(model, optimizer, epoch, val_loss, args.output_dir, training_logger)
        training_logger.log_info("Training completed!")
        training_logger.log_info("TensorBoard writer closed.")
        writer.close()
        return
    
    training_logger.log_info("\nTraining completed!")
    training_logger.log_info(f"Final checkpoint saved to: {args.output_dir}")
    
    # Close TensorBoard writer
    writer.close()
    training_logger.log_info("TensorBoard writer closed.")


if __name__ == '__main__':
    main()

