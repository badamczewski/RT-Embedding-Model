from loguru import logger
from typing import Optional, Set
from contextlib import contextmanager
import torch


class ModelLogger:
    """
    Selective logging for embedding models using Loguru.
    
    Usage:
        # Enable logging with specific categories
        model.logger.enable(categories={'shapes', 'statistics'})
        
        # Or use context manager for temporary logging
        with model.logger.enabled(categories={'shapes'}):
            output = model(input)
        
        # Disable all logging
        model.logger.disable()
        
        # Check if logging is enabled
        if model.logger.is_enabled:
            print("Logging is on")
    """
    
    def __init__(self, model_name: str = "EmbeddingModel"):
        self.model_name = model_name
        self.categories: Set[str] = set()
        self._enabled = False
    
    def enable(self, categories: Optional[Set[str]] = None):
        """Enable logging for specific categories. If None, enables all."""
        if categories is None:
            categories = {'all'}
        self.categories = set(categories)
        self._enabled = True
        logger.info(f"[{self.model_name}] Logging enabled for categories: {self.categories}")
    
    def disable(self):
        """Disable all logging."""
        self._enabled = False
        self.categories.clear()
        logger.info(f"[{self.model_name}] Logging disabled")
    
    @contextmanager
    def enabled(self, categories: Optional[Set[str]] = None):
        """Context manager for temporary logging."""
        was_enabled = self._enabled
        old_categories = self.categories.copy()
        
        self.enable(categories)
        try:
            yield self
        finally:
            self._enabled = was_enabled
            self.categories = old_categories
    
    @property
    def is_enabled(self) -> bool:
        """Check if logging is currently enabled."""
        return self._enabled
    
    def should_log(self, category: str) -> bool:
        """Check if a category should be logged."""
        return self._enabled and ('all' in self.categories or category in self.categories)
    
    def log_shapes(self, name: str, tensor: torch.Tensor, category: str = "shapes"):
        """Log tensor shapes."""
        if self.should_log(category):
            logger.debug(f"[{category}] {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    
    def log_statistics(self, name: str, tensor: torch.Tensor, category: str = "statistics"):
        """Log tensor statistics."""
        if self.should_log(category):
            with torch.no_grad():
                stats = {
                    'mean': tensor.float().mean().item(),
                    'std': tensor.float().std().item(),
                    'min': tensor.float().min().item(),
                    'max': tensor.float().max().item(),
                }
                logger.debug(f"[{category}] {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
    
    def log_value(self, name: str, value, category: str = "values"):
        """Log arbitrary values."""
        if self.should_log(category):
            logger.debug(f"[{category}] {name}: {value}")
    
    def log_info(self, message: str, category: str = "info"):
        """Log info messages."""
        if self.should_log(category):
            logger.info(f"[{category}] {message}")
    
    def log_warning(self, message: str):
        """Log warnings (always enabled)."""
        logger.warning(f"[{self.model_name}] {message}")
    
    def log_error(self, message: str):
        """Log errors (always enabled)."""
        logger.error(f"[{self.model_name}] {message}")
    
    def log_exception(self, message: str):
        """Log exceptions with full traceback."""
        logger.exception(f"[{self.model_name}] {message}")

