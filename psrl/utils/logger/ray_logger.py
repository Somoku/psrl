import inspect
import logging
import os
import socket
import time
from contextlib import contextmanager
from enum import Enum

import torch


def get_worker_info():
    """Get the worker info from the environment variables."""
    worker_ip = os.getenv("LOCAL_IP", None)
    if worker_ip is None:
        worker_ip = socket.gethostbyname(socket.gethostname())
    worker_gpu = None
    if torch.cuda.is_available():
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            visible_devices = list(map(int, visible_devices.split(",")))
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
    SWITCH = "SWITCH"
    OTHER = "OTHER"


def _log_with_caller_info(psrl_logger: logging.Logger, level: int, message: str):
    """Log a message with the caller's file and line information."""
    # Check if the logger is enabled for the given level
    if not psrl_logger.isEnabledFor(level):
        return

    # Try to use stacklevel to get the caller's file and line information
    try:
        psrl_logger.log(level, message, stacklevel=3)
    except TypeError:
        # If the Python version does not support stacklevel, fall back to the old method
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back.f_back
            pathname = caller_frame.f_code.co_filename
            lineno = caller_frame.f_lineno
            record = psrl_logger.makeRecord(psrl_logger.name, level, pathname, lineno, message, (), None)
            psrl_logger.handle(record)
        finally:
            del frame


@contextmanager
def log_dual_events(
    message: str,
    psrl_logger: logging.Logger,
    level: int = logging.INFO,
    event_type: EventType = EventType.OTHER,
):
    start_time = time.time()
    log_begin_event(message, psrl_logger, level, event_type)  # Log with label when entering
    try:
        yield  # Execute code within the with block
    finally:
        end_time = time.time()
        log_end_event(message, psrl_logger, level, event_type, end_time - start_time)  # Log end tag when exiting


def log_single_event(
    message: str,
    psrl_logger: logging.Logger,
    level: int = logging.INFO,
    event_type: EventType = EventType.OTHER,
):
    _log_with_caller_info(psrl_logger, level, f"[Single Event] {event_type.value} - {message}")


def log_begin_event(
    message: str,
    psrl_logger: logging.Logger,
    level: int = logging.INFO,
    event_type: EventType = EventType.OTHER,
):
    _log_with_caller_info(psrl_logger, level, f"[Begin Event] {event_type.value} - {message}")


def log_end_event(
    message: str,
    psrl_logger: logging.Logger,
    level: int = logging.INFO,
    event_type: EventType = EventType.OTHER,
    duration: float = None,
):
    if duration is None:
        _log_with_caller_info(psrl_logger, level, f"[End Event] {event_type.value} - {message}")
    else:
        _log_with_caller_info(
            psrl_logger,
            level,
            f"[End Event] {event_type.value} - {message} - Time taken: {duration:.2f} seconds",
        )


class DualOutputHandler(logging.Handler):
    """A logger handler that writes to both the original stdout and a file."""

    def __init__(self, log_dir, log_prefix):
        super().__init__()
        self.log_prefix = log_prefix
        # Create log file
        log_dir = os.path.expanduser(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, log_prefix + ".log")
        # Create handler
        self.file_handler = logging.FileHandler(self.file_path, mode="w")
        self.stream_handler = logging.StreamHandler()
        # Define file log formats
        file_log_format = "%(asctime)s - %(filename)s - %(lineno)d - %(message)s"
        self.file_formatter = logging.Formatter(file_log_format)
        self.file_handler.setFormatter(self.file_formatter)

    def emit(self, record):
        # formatted_message = self.file_formatter.format(record)
        # print(formatted_message, file=open(self.file_path, "a"))
        # Emit the original log record to file handler
        self.file_handler.emit(record)

        # For stream handler, create a copy of the record and modify the message
        stream_record = logging.makeLogRecord(record.__dict__)
        stream_record.msg = f"<{self.log_prefix}> - {record.getMessage()}"
        stream_record.args = ()  # Clear args since we already formatted the message

        # Emit the modified record to stream handler
        self.stream_handler.emit(stream_record)


class FileOnlyHandler(logging.Handler):
    """A logger handler that writes only to a file."""

    def __init__(self, log_dir, log_prefix):
        super().__init__()
        self.log_prefix = log_prefix
        # Create log file
        log_dir = os.path.expanduser(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, log_prefix + ".log")
        # Create handler
        self.file_handler = logging.FileHandler(self.file_path, mode="w")
        # Define file log formats
        file_log_format = "%(asctime)s - %(filename)s - %(lineno)d - %(message)s"
        self.file_formatter = logging.Formatter(file_log_format)
        self.file_handler.setFormatter(self.file_formatter)

    def emit(self, record):
        # Emit the log record to file handler only
        self.file_handler.emit(record)
