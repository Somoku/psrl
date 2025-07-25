from psrl.utils.nixl.network_topology import NetworkTopology
from psrl.utils.nixl.port_scanner import PortScanner

# Global network topology instance
global_topology = NetworkTopology() 

# Global port scanner instance
global_port_scanner = PortScanner.remote()

# Global name of meta server
global_meta_server_name = "NIXLMetaServer"