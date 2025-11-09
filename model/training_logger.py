from loguru import logger
from typing import Optional, Dict, Any
from datetime import datetime
import os


class TrainingLogger:
    """
    Logger for training loop that logs to both console and file.
    
    Usage:
        training_logger = TrainingLogger(log_dir="logs")
        training_logger.log_train_step(step=1, loss=0.5, lr=1e-4)
        training_logger.log_validation(epoch=1, val_loss=0.4)
        training_logger.log_epoch(epoch=1, train_loss=0.5, val_loss=0.4)
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save log files
            experiment_name: Optional name for the experiment (used in log filename)
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name or "training"
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.experiment_name}_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)
        
        # Configure loguru to log to file
        logger.remove()  # Remove default handler
        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="INFO",
            rotation="100 MB",  # Rotate when file gets too large
            retention=10,  # Keep last 10 log files
        )
        logger.add(
            lambda msg: print(msg, end=''),  # Console output
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level="INFO",
            colorize=True
        )
        
        self.log_path = log_path
        logger.info(f"Training logger initialized. Log file: {log_path}")
    
    def log_train_step(self, step: int, loss: float, lr: float, epoch: Optional[int] = None):
        """
        Log a training step.
        
        Args:
            step: Current training step
            loss: Training loss for this step
            lr: Current learning rate
            epoch: Current epoch (optional)
        """
        epoch_str = f"Epoch {epoch} | " if epoch is not None else ""
        logger.info(f"{epoch_str}Step {step} | Loss: {loss:.6f} | LR: {lr:.2e}")
    
    def log_validation(self, epoch: int, val_loss: float, metrics: Optional[Dict[str, float]] = None):
        """
        Log validation results.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            metrics: Optional dictionary of additional metrics
        """
        metrics_str = ""
        if metrics:
            metrics_str = " | " + " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        logger.info(f"Epoch {epoch} | Validation Loss: {val_loss:.6f}{metrics_str}")
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None, 
                  lr: Optional[float] = None, metrics: Optional[Dict[str, float]] = None):
        """
        Log epoch summary.
        
        Args:
            epoch: Current epoch number
            train_loss: Average training loss for the epoch
            val_loss: Average validation loss (optional)
            lr: Learning rate (optional)
            metrics: Optional dictionary of additional metrics
        """
        val_str = f" | Val Loss: {val_loss:.6f}" if val_loss is not None else ""
        lr_str = f" | LR: {lr:.2e}" if lr is not None else ""
        metrics_str = ""
        if metrics:
            metrics_str = " | " + " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        
        logger.info(f"Epoch {epoch} Summary | Train Loss: {train_loss:.6f}{val_str}{lr_str}{metrics_str}")
    
    def log_checkpoint(self, epoch: int, checkpoint_path: str, metrics: Optional[Dict[str, Any]] = None):
        """
        Log checkpoint saving.
        
        Args:
            epoch: Epoch number for the checkpoint
            checkpoint_path: Path where checkpoint was saved
            metrics: Optional dictionary of checkpoint metrics
        """
        metrics_str = ""
        if metrics:
            metrics_str = " | " + " | ".join([f"{k}: {v}" for k, v in metrics.items()])
        logger.info(f"Checkpoint saved | Epoch {epoch} | Path: {checkpoint_path}{metrics_str}")
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log training configuration.
        
        Args:
            config: Dictionary of training configuration parameters
        """
        logger.info("=" * 60)
        logger.info("Training Configuration")
        logger.info("=" * 60)
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
    
    def log_info(self, message: str):
        """Log an info message."""
        logger.info(message)
    
    def log_warning(self, message: str):
        """Log a warning message."""
        logger.warning(message)
    
    def log_error(self, message: str):
        """Log an error message."""
        logger.error(message)

