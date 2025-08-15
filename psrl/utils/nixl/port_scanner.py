import socket
import ray

    
@ray.remote
class PortScanner:
    def __init__(self):
        self.min_port = 20000
        self.max_port = 60000
        
    def find_free_port(self, host="127.0.0.1"):
        for port in range(self.min_port, self.max_port + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex((host, port)) != 0:  # port is available
                    self.min_port = port + 1
                    return port
        raise RuntimeError("No free ports")


# Global port scanner instance
GLOBAL_PORT_SCANNER = PortScanner.remote()
