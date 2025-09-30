import sys
from src.logger import logging

def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_no = "Unknown"
    error_message = "Error occurred in python script [{0}] at line [{1}] with message: {2}".format(
        file_name, line_no, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message