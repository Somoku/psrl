from psrl.utils.nixl.network_topology import NetworkTopology
from psrl.utils.nixl.port_scanner import PortScanner

# Global network topology instance
GLOBAL_TOPOLOGY = NetworkTopology()

# Global port scanner instance
GLOBAL_PORT_SCANNER = PortScanner.remote()

# NOTE(claude): All NIXL string constants (NIXL_META_SERVER_NAME and client-name
# prefixes) have been moved to psrl.utils.common.nixl_names, which is the single
# source of truth for NIXL identifiers
