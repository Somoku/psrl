# NIXL Meta Server Protocol: Step-by-Step

This document describes the full protocol for NIXL meta server and client communication, including all steps for sharding, tensor registration, communication plan, and data transfer.

---

## 1. Clients Send Sharding Information
- `client.connect_to_server` and `client.send_local_sharding`: Each client sends its local sharding dictionary to the meta server.

## 2. Server Builds Unified Sharding
- `server.wait_for_client_shardings`: The meta server collects all sharding dicts.
- `server.make_unified_sharding`: The meta server builds a unified sharding for all keys.

## 3. Server Notifies All Clients of Unified Sharding
- `server.notify_all_client_shardings`: The meta server sends the unified sharding dict to all clients.
- `client.wait_for_server_sharding`: Each clients wait for the unified sharding.

## 4. Clients Register Tensors and Build Local Client Info
- `client.register_local_tensors`: Each client registers its local tensors according to the unified sharding, and builds its own `NIXLClientInfo` and temporary mapping for non-contiguous tensors.

## 5. Clients Send Client Info to Server
- `client.send_local_info`: Each client sends its `NIXLClientInfo` to the meta server.

## 6. Server Builds Communication Plan
- `server.wait_for_client_infos`: The meta server collects all client infos.
- `server.make_comm_plan`: The meta server builds a communication plan (comm plan).

## 7. Server Notifies All Clients of All Client Infos and Comm Plan
- `server.notify_all_client_infos_and_comm_plan`: The meta server sends all client infos and the comm plan to every client.
- `client.wait_server_info`: Each clients wait for all client infos and the comm plan.

## 8. Clients Send All Temporary Mappings to Server
- `client.send_local_temp_mapping`: Each client sends its temporary mapping (for non-contiguous tensors) to the meta server.
- `server.wait_for_client_temp_mappings`: The meta server collects all client temporary mappings.

## 9. Server Notifies All Clients of All Temporary Mappings
- `server.notify_all_client_temp_mappings`: The meta server sends all clients' temporary mappings to every client.
- `client.wait_for_server_temp_mappings`: Each client waits for all client temporary mappings.

## 10. Clients Perform Data Transfer According to Comm Plan
- `client.client_write` for every keys: PUSH_SIDE clients write (push) state dict shards to PS clients.
- `client.client_read` for every keys: PULL_SIDE clients read (pull) state dict shards from PS clients.

## 11. Clients Wait for the Data Transfers
- `client.wait`: wait for a specific data transfer. 

---

This protocol ensures all clients and the meta server are fully synchronized before any data transfer, supporting flexible and efficient distributed parameter storage and communication. 