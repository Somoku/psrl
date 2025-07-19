import enum
from enum import Enum
from typing import Union, List, Optional

import ray

from psrl.utils.ray import add_lock
from psrl.workers.ps.staleness_controller import EntryInfo
from psrl.utils.server.command import CommandType, Command

class RequestStatus(Enum):
    """
    Representing the status of a request in the system.
    
    PENDING: Request is queued in data queue, waiting for dispatch
    DISPATCHED: Request is dispatched to a specific Gen Worker, but not yet in the engine request queue
    ROLLOUT_RUNNING: Request is in the engine request queue and is being rolled out
    ROLLOUT_COMPLETED: Rollout completed, waiting for reward processing
    ROLLOUT_INTERRUPTED: Rollout was interrupted by the user for Partial Rollout, put into replay buffer
    REWRAD_RUNNING: Request is running in the reward server
    REWARD_COMPLETED: Request is completed in the reward server
    """
    
    PENDING = enum.auto()
    DISPATCHED = enum.auto()
    ROLLOUT_RUNNING = enum.auto()
    ROLLOUT_COMPLETED = enum.auto()
    ROLLOUT_INTERRUPTED = enum.auto()
    REWARD_RUNNING = enum.auto()
    REWARD_COMPLETED = enum.auto()

@ray.remote
class RequestStatusManager:
    """
    Manages the status of requests in the system.

    This class provides methods to update and retrieve the status of requests.
    It is used to track the lifecycle of requests as they move through different stages.
    """
    
    def __init__(self, config):
        self.config = config
        if self.config.psrl.rollout_test.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.rollout_test.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.rollout_test.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        
        self._request_id_to_status: dict[int, RequestStatus] = {} # Maps request ID to their statuses
        self._request_infos = {}  # Maps request IDs to EntryInfo objects
        # Maps statuses to sets of request IDs for quick access
        self._status_to_request_ids = {
            status: set() for status in RequestStatus
        }
        self._abort_request_ids = set() # Set of request IDs that are marked for abortion
        self._running_min_version = 0 # Minimum version of requests that are currently running
        self.rollout_request_buffer = {}  # Buffer for storing request data during rollout processing
        
        # Communication handle with rollout and reward servers
        self.rollout_server = None
        self.reward_server = None

    def set_rollout_server(self, rollout_server: ray.actor.ActorHandle):
        """
        Set the reference to the rollout server.
        
        Args:
            rollout_server: The reference to the rollout server actor.
        """
        self.rollout_server = rollout_server

    def set_reward_server(self, reward_server_ref):
        """
        Set the reference to the reward server.
        
        Args:
            reward_server_ref: The reference to the reward server actor.
        """
        self.reward_server = reward_server_ref

    def update_status(self, request_id: Union[List[int], int], status: RequestStatus, rollout_instance_id=-1):
        """
        Update the status of a request.
        This method will first check if the request will be aborted, and if so,
        it will return False and removed from the status map.
        
        Args:
            request_id (List[int], int): The unique identifier of the request.
            status (RequestStatus): The new status to set for the request.
        
        Returns:
            True if the status was updated successfully, False if the request was aborted.
        """
        if isinstance(request_id, int):
            request_id = [request_id]  # Convert single request_id to a list for uniform processing
        
        # Ensure the status is valid
        if status not in RequestStatus:
            raise ValueError(f"Invalid status: {status}")
        
        request_update_success = [True for _ in range(len(request_id))]
        
        for i, req_id in enumerate(request_id):
            # Check if the request is marked for abortion
            if req_id in self._abort_request_ids:
                request_update_success[i] = False
                self._abort_request_ids.remove(req_id)  # Remove from abort set
                self.remove_request(req_id)  # Remove from status and info maps
                if req_id in self.rollout_request_buffer:
                    del self.rollout_request_buffer[req_id]  # Remove from buffer if exists
                continue
            
            # If the request is stale, we should not update its status
            if req_id in self._request_infos:
                if self._request_infos[req_id].version < self._running_min_version:
                    request_update_success[i] = False
                    continue
            
            if req_id in self._request_id_to_status:
                old_status = self._request_id_to_status[req_id]
                if old_status != status:
                    self._status_to_request_ids[old_status].discard(req_id)
                self._status_to_request_ids[status].add(req_id)
                self._request_id_to_status[req_id] = status
                # Update the rollout instance ID if provided
                self._request_infos[req_id].rollout_instance_id = rollout_instance_id
            else:
                raise KeyError(f"Request ID {req_id} not found in status map.")
        
        return request_update_success

    def get_status(self, request_id: Union[List[int], int]):
        """
        Get the current status of a request.
        
        Args:
            request_id (List[int], int): The info or identifier of the requests.
        
        Returns:
            RequestStatus: The current status of the requests. If the request ID does not exist, returns None.
        """
        if isinstance(request_id, int):
            request_id = [request_id]

        status_list = [self._request_id_to_status.get(req_id, None) for req_id in request_id]
        if any(status is None for status in status_list):
            raise KeyError(f"One or more request IDs not found: {request_id}")

        return status_list

    def remove_train_ready_request(self, request_id: Union[List[int], int]):
        """
        Remove a request that is ready for training.
        
        Args:
            request_id (List[int], int): The unique identifier of the request to remove.
        """
        if isinstance(request_id, int):
            request_id = [request_id]

        for req_id in request_id:
            assert req_id in self._request_id_to_status, f"Request ID {req_id} not found in status map."
            assert req_id in self._request_infos, f"Request ID {req_id} not found in request infos."
            assert self._request_id_to_status[req_id] == RequestStatus.ROLLOUT_COMPLETED, \
                f"Request ID {req_id} is not in ROLLOUT_COMPLETED status."
            
            assert req_id not in self._abort_request_ids, \
                f"Request ID {req_id} is marked for abortion and cannot be removed."
            
            # Remove the request from the status map and request infos
            del self._request_id_to_status[req_id]
            del self._request_infos[req_id]
            self._status_to_request_ids[RequestStatus.ROLLOUT_COMPLETED].discard(req_id)

    def get_all_statuses(self) -> dict:
        """
        Get the statuses of all requests.
        
        Returns:
            dict: A dictionary mapping request infos to their statuses.
        """
        return self._request_id_to_status.copy()

    def get_requests_by_status(self, status: RequestStatus) -> set:
        """
        Get all requests that have a specific status.
        
        Args:
            status (RequestStatus): The status to filter requests by.
        
        Returns:
            set: A set of request ids that match the specified status.
        """
        return self._status_to_request_ids.get(status, set())
    
    def add_request(
        self,
        request_id: int,
        rollout_instance_id: int = -1,
        model_version: int = -1,
        status: RequestStatus = RequestStatus.PENDING,
    ):
        """
        Add a new request with its initial status.
        
        Args:
            request_id (int): The unique identifier of the request.
            rollout_instance_id (int): The ID of the rollout instance this request belongs to. Defaults to -1 as a placeholder.
            model_version (int): The version of the model associated with this request. Defaults to -1 as a placeholder.
            status (RequestStatus): The initial status of the request. Defaults to PENDING.
        """
        request_id = int(request_id)
        self._request_infos[request_id] = EntryInfo(
            request_id=request_id,
            rollout_instance_id=rollout_instance_id,
            model_version=model_version,
        )
        self._request_id_to_status[request_id] = status
        self._status_to_request_ids[status].add(request_id)
    
    def abort_requests(self, request_ids: Union[List[int], int], blocking: bool = False):
        """
        Add request IDs to the abort set.
        This method will classify the requests in `request_ids` into their current statuses
        and abort them accordingly.
        
        Args:
            request_ids (List[int], int): The unique identifiers of the requests to abort.
            blocking (bool): If True, will block until the requests are aborted. Defaults to False.
        """
        if isinstance(request_ids, int):
            request_ids = [request_ids]
        
        request_ids = set(request_ids) # Ensure uniqueness
        filtered_request_ids = [req_id for req_id in request_ids if req_id in self._request_id_to_status]
        self._abort_request_ids.update(filtered_request_ids)
        
        # Classify the requests in `request_ids` into their current statuses
        classified_requests = self.classify_requests_in_status(filtered_request_ids)
        
        abort_requests_for_rollout = set()
        abort_requests_for_reward = set()
        
        for status, req_ids in classified_requests.items():
            if status in {RequestStatus.ROLLOUT_RUNNING}:
                abort_requests_for_rollout.update(req_ids)
            elif status in {RequestStatus.REWARD_RUNNING}:
                abort_requests_for_reward.update(req_ids)
        
        ray.get(self.rollout_server.exec_command.remote(
            Command(
                type=CommandType.ABORT,
                uids=list(abort_requests_for_rollout),
            ),
            blocking=blocking,
        ))
        
        ray.get(self.reward_server.exec_command.remote(
            Command(
                type=CommandType.ABORT,
                uids=list(abort_requests_for_reward),
            ),
            blocking=blocking,
        ))
    
    def classify_requests_in_status(self, request_ids: Union[List[int], int]) -> dict:
        """
        Classify the requests in `request_ids` into their current statuses.
        
        Args:
            request_ids (List[int], int): The unique identifiers of the requests to classify.
        
        Returns:
            dict: A dictionary mapping statuses to sets of request IDs that are in those statuses.
        """
        if isinstance(request_ids, int):
            request_ids = [request_ids]
        
        classified_requests = {}
        for req_id in request_ids:
            if req_id in self._request_id_to_status:
                status = self._request_id_to_status[req_id]
                classified_requests.setdefault(status, set()).add(req_id)
            else:
                raise KeyError(f"Request ID {req_id} not found in status map.")
        
        return classified_requests

    def classify_requests_in_instance(self, request_ids: Union[List[int], int]) -> dict:
        """
        Classify the requests in `request_ids` into their associated rollout instances.
        
        Args:
            request_ids (List[int], int): The unique identifiers of the requests to classify.
        
        Returns:
            dict: A dictionary mapping instance IDs to sets of request IDs that belong to those instances.
        """
        if isinstance(request_ids, int):
            request_ids = [request_ids]
        
        classified_requests = {}
        for req_id in request_ids:
            if req_id in self._request_infos:
                instance_id = self._request_infos[req_id].rollout_instance_id
                classified_requests.setdefault(instance_id, set()).add(req_id)
            else:
                raise KeyError(f"Request ID {req_id} not found in request infos.")
        
        return classified_requests
    
    def get_recorded_child_requests(self, parent_id: Union[List[int], int]) -> set[int]:
        """
        Get the recorded child requests for a given parent request ID.
        
        Args:
            parent_id (List[int], int): The unique identifier of the parent request.
        
        Returns:
            set[int]: A list of child request IDs associated with the parent request.
        """
        if isinstance(parent_id, int):
            parent_id = [parent_id]
        
        child_requests = []
        for pid in parent_id:
            for i in range(self.rollout_n):
                child_id = pid * self.rollout_n + i
                if child_id in self._request_infos:
                    child_requests.append(child_id)
        
        return set(child_requests)
    
    def get_request_info(self, request_id: Union[List[int], int]) -> Union[List[EntryInfo], EntryInfo, None]:
        """
        Get the EntryInfo for a specific request ID.
        
        Args:
            request_id (List[int], int): The unique identifier of the request.
        
        Returns:
            EntryInfo: The EntryInfo objects associated with the request IDs, or None if not found.
        """
        if isinstance(request_id, int):
            request_id = [request_id]
        
        request_infos = [self._request_infos.get(req_id, None) for req_id in request_id]
        if any(info is None for info in request_infos):
            raise KeyError(f"One or more request IDs not found: {request_id}")

        if len(request_infos) == 1:
            return request_infos[0]
        else:
            return self._request_infos.get(request_id, None)

    def update_request_info(self, entry_info: Union[List[EntryInfo], EntryInfo]):
        """
        Update the EntryInfo for a specific request ID.
        
        Args:
            entry_info (List[EntryInfo], EntryInfo): The new EntryInfo object to associate with the request ID.
        """
        if isinstance(entry_info, EntryInfo):
            entry_info = [entry_info]
        
        # Ensure all request IDs in entry_info exist in the manager
        for info in entry_info:
            request_id = info.request_id
            if request_id not in self._request_infos:
                raise KeyError(f"Request ID {request_id} not found.")
        
        # Update the EntryInfo for each request ID
        for info in entry_info:
            request_id = info.request_id
            if request_id in self._request_infos:
                self._request_infos[request_id] = entry_info
            else:
                raise KeyError(f"Request ID {request_id} not found.")

    def remove_request(self, request_id: Union[List[int], int]):
        """
        Remove a request and its associated status.
        
        Args:
            request_id (List[int], int): The unique identifier of the request to abort.
        """
        if isinstance(request_id, int):
            request_id = [request_id]

        for req_id in request_id:
            if req_id in self._request_id_to_status:
                status = self._request_id_to_status[req_id]
                self._status_to_request_ids[status].discard(req_id)
                del self._request_id_to_status[req_id]
            
            if req_id in self._request_infos:
                del self._request_infos[req_id]
            
            if req_id in self._abort_request_ids:
                self._abort_request_ids.remove(req_id)

    def clear(self):
        """
        Clear all requests and their statuses.
        """
        self._request_id_to_status.clear()
        self._status_to_request_ids = {status: set() for status in RequestStatus}
        self._request_infos.clear()
        self._abort_request_ids.clear()
    
    def get_requests_of_version(self, version: int) -> set[int]:
        """
        Get all requests associated with a specific version.
        
        Args:
            version (int): The version to filter requests by.
        
        Returns:
            set[int]: A set of request IDs that match the specified version.
        """
        assert version >= 0, "Version must be a non-negative integer."
        return {req_id for req_id, info in self._request_infos.items() if info.version == version}
    
    def abort_requests_of_version(self, version: int):
        """
        Abort all requests associated with a specific version.
        This method will collect all request IDs that match the specified version
        and abort them by `abort_requests` method. It will also set the min running
        verion to decline future requests with version tag less than it.
        
        Args:
            version (int): The version to abort requests for.
        """
        assert version >= 0, "Version must be a non-negative integer."

        abort_request_ids = set()
        for req_id, info in self._request_infos.items():
            if info.version == version:
                # If the request version matches, we will abort it
                abort_request_ids.add(req_id)
            elif info.version < version:
                raise ValueError(f"Found request ID {req_id} with version {info.version} when buffer version is {version}.")
        self._running_min_version = max(self._running_min_version, version + 1)
        self.abort_requests(list(abort_request_ids), blocking=True)

    def update_request_version(self, request_id: Union[List[int], int], new_version: int):
        """
        Update the version of a request.
        
        Args:
            request_id (List[int], int): The unique identifier of the request.
            new_version (int): The new version to set for the request.
        """
        if isinstance(request_id, int):
            request_id = [request_id]

        for req_id in request_id:
            if req_id in self._request_infos:
                self._request_infos[req_id].version = new_version
            else:
                raise KeyError(f"Request ID {req_id} not found.")
    
    def update_request_instance(self, request_id: Union[List[int], int], new_instance: int):
        """
        Update the instance of a request.
        
        Args:
            request_id (List[int], int): The unique identifier of the request.
            new_instance (id): The new instance to set for the request.
        """
        if isinstance(request_id, int):
            request_id = [request_id]

        for req_id in request_id:
            if req_id in self._request_infos:
                self._request_infos[req_id].instance = new_instance
            else:
                raise KeyError(f"Request ID {req_id} not found.")

    def add_request_data_to_buffer(self, request_data: dict):
        """
        Add request data to the rollout request buffer.
        
        Args:
            request_data (dict): A dictionary mapping request IDs to their corresponding data.
        """
        for req_id, data in request_data.items():
            if req_id not in self._request_id_to_status:
                raise KeyError(f"Request ID {req_id} not found in status map.")
            self.rollout_request_buffer[req_id] = data
    
    def get_request_data_from_buffer(self, request_id: int) -> Optional[dict]:
        """
        Get request data from the rollout request buffer.
        
        Args:
            request_id (int): The unique identifier of the request.
        
        Returns:
            dict: The data associated with the request ID, or None if not found.
        """
        return self.rollout_request_buffer.get(request_id, None)

    def pop_request_data_from_buffer(self, request_id: int) -> Optional[dict]:
        """
        Pop request data from the rollout request buffer.
        
        Args:
            request_id (int): The unique identifier of the request.
        
        Returns:
            dict: The data associated with the request ID, or None if not found.
        """
        return self.rollout_request_buffer.pop(request_id, None)
    
    def remove_request_data_from_buffer(self, request_id: int):
        """
        Remove request data from the rollout request buffer.
        
        Args:
            request_id (int): The unique identifier of the request.
        """
        if request_id in self.rollout_request_buffer:
            del self.rollout_request_buffer[request_id]
        else:
            raise KeyError(f"Request ID {request_id} not found in rollout request buffer.")