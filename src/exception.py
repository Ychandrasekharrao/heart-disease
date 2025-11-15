"""
Custom Exception Handler for Heart Disease Prediction
Provides detailed error tracking and logging
"""

import sys
from typing import Optional

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Generate detailed error message with file, line number, and message.
    
    Args:
        error: Exception instance
        error_detail: sys module for extracting traceback
    
    Returns:
        Formatted error message string
    """
    _, _, exc_tb = error_detail.exc_info()
    
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = (
            f"Error in [{file_name}] "
            f"line [{line_number}]: "
            f"{str(error)}"
        )
    else:
        error_message = f"Error: {str(error)}"
    
    return error_message


class CustomException(Exception):
    """
    Custom exception class with enhanced error reporting.
    """
    
    def __init__(self, error_message: str, error_detail: Optional[Exception] = None):
        """
        Initialize custom exception.
        
        Args:
            error_message: Error message
            error_detail: Original exception (optional)
        """
        super().__init__(error_message)
        
        # FIX: Use sys module, not the exception object
        if error_detail is not None:
            self.error_message = error_message_detail(error_detail, sys)
        else:
            self.error_message = error_message
        
        self.original_error = error_detail
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        return self.error_message