class DualLogger:
    """A logger that writes to both the original stdout and a file."""
    def __init__(self, original_stdout, filename):
        self.original_stdout = original_stdout
        self.log_file = open(filename, 'w')

    def write(self, data):
        self.original_stdout.write(data)
        self.log_file.write(data)
        self.flush()

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()