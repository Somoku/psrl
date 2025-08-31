from psrl.utils.nixl.network_topology import NetworkTopology
from psrl.utils.nixl.port_scanner import PortScanner

# Global network topology instance
GLOBAL_TOPOLOGY = NetworkTopology() 

# Global port scanner instance
GLOBAL_PORT_SCANNER = PortScanner.remote()

# Global name
GLOBAL_META_SERVER_NAME = "NIXLMetaServer"
GLOBAL_TRAIN_CLIENT_NAME = "NIXLTrainClient"
GLOBAL_GEN_CLIENT_NAME = "NIXLGenClient"
GLOBAL_PS_CLIENT_NAME = "NIXLPSClient"