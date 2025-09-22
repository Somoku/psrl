import os
import logging
import socket
import torch
import time
import inspect
from contextlib import contextmanager
from enum import Enum


def get_worker_info():
    """Get the worker info from the environment variables."""
    worker_ip = os.getenv("LOCAL_IP", None)
    if worker_ip is None:
        worker_ip = socket.gethostbyname(socket.gethostname())
    worker_gpu = None
    if torch.cuda.is_available():
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible_devices:
            visible_devices = list(map(int, visible_devices.split(',')))
            current_logical = torch.cuda.current_device()
            current_physical = visible_devices[current_logical]
            worker_gpu = f"GPU {current_physical}"
    return worker_ip, worker_gpu


class EventType(Enum):
    PULL = "PULL"
    PUSH = "PUSH"
    BUFFER_READY = "BUFFER_READY"
    INIT = "INIT"
    TRAIN = "TRAIN"
    GEN = "GEN"
    VAL = "VAL"
    WAIT = "WAIT"
    OTHER = "OTHER"


def _log_with_caller_info(psrl_logger: logging.Logger, level: int, message: str):
    """Log a message with the caller's file and line information."""
    # Get the caller's frame (skip this function and the wrapper function)
    frame = inspect.currentframe()
    try:
        # Skip current frame and the wrapper function frame
        caller_frame = frame.f_back.f_back
        filename = caller_frame.f_code.co_filename
        lineno = caller_frame.f_lineno
        
        # Create a LogRecord with the caller's information
        record = psrl_logger.makeRecord(
            name=psrl_logger.name,
            level=level,
            fn=filename,
            lno=lineno,
            msg=message,
            args=(),
            exc_info=None
        )
        psrl_logger.handle(record)
    finally:
        del frame


@contextmanager
def log_dual_events(message: str, psrl_logger: logging.Logger, level: int = logging.INFO, event_type: EventType = EventType.OTHER):
    start_time = time.time()
    _log_with_caller_info(psrl_logger, level, f"[Begin Event] {event_type.value} - {message}")  # Log with label when entering
    try:
        yield  # Execute code within the with block
    finally:
        end_time = time.time()
        _log_with_caller_info(psrl_logger, level, f"[End Event] {event_type.value} - {message} - Time taken: {end_time - start_time:.2f} seconds")  # Log end tag when exiting
        
        
def log_single_event(message: str, psrl_logger: logging.Logger, level: int = logging.INFO, event_type: EventType = EventType.OTHER):
    _log_with_caller_info(psrl_logger, level, f"[Single Event] {event_type.value} - {message}")
    
    
class DualOutputHandler(logging.Handler):
    """A logger handler that writes to both the original stdout and a file."""
    def __init__(self, log_dir, log_prefix):
        super().__init__()
        self.log_prefix = log_prefix
        # Create log file
        log_dir = os.path.expanduser(log_dir) 
        os.makedirs(log_dir, exist_ok=True) 
        file_path = os.path.join(log_dir, log_prefix + ".log")
        # Create handler
        self.file_handler = logging.FileHandler(file_path, mode='w')
        self.stream_handler = logging.StreamHandler()
        # Define file log formats
        file_log_format = '%(asctime)s - %(filename)s - %(lineno)d - %(message)s'
        file_formatter = logging.Formatter(file_log_format)
        self.file_handler.setFormatter(file_formatter)

    def emit(self, record):
        # Emit the original log record to file handler
        self.file_handler.emit(record)
        
        # For stream handler, create a copy of the record and modify the message
        stream_record = logging.makeLogRecord(record.__dict__)
        stream_record.msg = f"<{self.log_prefix}> - {record.getMessage()}"
        stream_record.args = ()  # Clear args since we already formatted the message
        
        # Emit the modified record to stream handler
        self.stream_handler.emit(stream_record)