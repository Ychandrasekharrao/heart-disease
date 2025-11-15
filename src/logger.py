"""
Enhanced Logging System for Heart Disease Prediction
Provides comprehensive logging with rotating file handlers and structured output
"""

import logging as std_logging  # Rename to avoid collision
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(
    log_file: Optional[Path] = None,
    log_level: int = std_logging.INFO,
    console_output: bool = True
) -> std_logging.Logger:
    """
    Configure comprehensive logging system.
    
    Args:
        log_file: Path to log file (default: logs/YYYY-MM-DD_HH-MM-SS.log)
        log_level: Logging level (default: INFO)
        console_output: Whether to output to console (default: True)
    
    Returns:
        Configured logger instance
    """
    if log_file is None:
        log_file = LOG_FILE_PATH
    
    # Create formatter
    log_format = (
        "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
    )
    formatter = std_logging.Formatter(log_format)
    
    # Configure root logger
    logger = std_logging.getLogger("heart_disease_prediction")
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler
    try:
        file_handler = std_logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create file handler: {e}")
    
    # Console handler
    if console_output:
        console_handler = std_logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


# ============================================================================
# GLOBAL LOGGER INSTANCE
# ============================================================================

# Create the main logger instance
logging = setup_logging()

# Log initialization
logging.info("=" * 80)
logging.info("Logging system initialized successfully")
logging.info(f"Log file: {LOG_FILE_PATH}")
logging.info(f"Log level: {std_logging.getLevelName(logging.level)}")
logging.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logging.info("=" * 80)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_section(message: str, logger: Optional[std_logging.Logger] = None) -> None:
    """
    Log a section header with visual separation.
    
    Args:
        message: Section header message
        logger: Logger instance (default: global logger)
    """
    if logger is None:
        logger = logging
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(message.center(80))
    logger.info("=" * 80)


def log_dataframe_info(df, name: str = "DataFrame", logger: Optional[std_logging.Logger] = None) -> None:
    """
    Log comprehensive DataFrame information.
    
    Args:
        df: pandas DataFrame
        name: Name of the DataFrame
        logger: Logger instance (default: global logger)
    """
    if logger is None:
        logger = logging
    
    logger.info(f"\n{name} Information:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.info(f"  Missing values:\n{missing[missing > 0]}")
    else:
        logger.info("  No missing values")


def log_model_metrics(metrics: dict, logger: Optional[std_logging.Logger] = None) -> None:
    """
    Log model performance metrics in a structured format.
    
    Args:
        metrics: Dictionary of metric name -> value
        logger: Logger instance (default: global logger)
    """
    if logger is None:
        logger = logging
    
    logger.info("\nModel Performance Metrics:")
    logger.info("-" * 50)
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            logger.info(f"  {metric_name:.<40} {metric_value:.4f}")
        else:
            logger.info(f"  {metric_name:.<40} {metric_value}")
    logger.info("-" * 50)


def log_exception(e: Exception, context: str = "", logger: Optional[std_logging.Logger] = None) -> None:
    """
    Log exception with full traceback and context.
    
    Args:
        e: Exception instance
        context: Additional context information
        logger: Logger instance (default: global logger)
    """
    if logger is None:
        logger = logging
    
    logger.error("=" * 80)
    logger.error("EXCEPTION OCCURRED")
    if context:
        logger.error(f"Context: {context}")
    logger.error(f"Exception Type: {type(e).__name__}")
    logger.error(f"Exception Message: {str(e)}")
    logger.exception("Full Traceback:")
    logger.error("=" * 80)


def get_logger(name: str = "heart_disease_prediction") -> std_logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return std_logging.getLogger(name)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test logging functionality
    log_section("LOGGING SYSTEM TEST")
    
    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    
    # Test DataFrame logging
    import pandas as pd
    test_df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, None]
    })
    log_dataframe_info(test_df, "Test DataFrame")
    
    # Test metrics logging
    test_metrics = {
        'Accuracy': 0.9542,
        'Precision': 0.9123,
        'Recall': 0.8876,
        'F1-Score': 0.8997
    }
    log_model_metrics(test_metrics)
    
    # Test exception logging
    try:
        1 / 0
    except Exception as e:
        log_exception(e, context="Division by zero test")
    
    logging.info("✓ Logging system test complete")