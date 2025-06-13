import os
import logging
import socket
import torch

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

class DualOutputHandler(logging.Handler):
    """A logger handler that writes to both the original stdout and a file."""
    def __init__(self, log_prefix):
        super().__init__()
        self.log_prefix = log_prefix
        # Create log file
        log_dir = os.getenv("PSRL_LOGGING_PATH", "~/psrl_log")
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
        # Emit the log record to both handlers
        self.file_handler.emit(record)
        record.msg = f"<{self.log_prefix}> - {record.getMessage()}"
        self.stream_handler.emit(record)