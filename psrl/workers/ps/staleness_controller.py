import os
import logging
import enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Tuple, Set

import ray

from verl import DataProto

from psrl.utils.logger import deprecated
from psrl.utils.logger import get_ps_logger

# Use the unified PS logger
psrl_logger = get_ps_logger()

class EntryCategory(enum.Enum):
    """Enum for the category of an entry in the buffer.
    
    EMPTY: The entry is empty and available for reservation
    RESERVED: The entry is reserved for a request but not yet occupied
    OCCUPIED: The entry is occupied with data and available for training
    """
    EMPTY = 0
    RESERVED = 1
    OCCUPIED = 2

@dataclass
class EntryInfo:
    """
    Metadata for an entry in the buffer.

    This class is used to track the metadata of an entry in the buffer,
    including the rollout instance ID, request ID, and model version.

    Args:
        rollout_instance_id (Union[str, int]): The ID of the rollout instance this entry belongs to.
        request_id (Union[str, int]): The global unique request ID.
        model_version (int): The model version when generating this entry.
    """
    rollout_instance_id: Union[str, int]  # The ID of the rollout instance this entry belongs to
    request_id: Union[str, int] # The global unique request ID
    # The model version when generating this entry, which should be within staleness control
    # (i.e., higher than the final occupied buffer ID minus the staleness limit)
    model_version: int

    def __hash__(self):
        return hash(self.request_id)
    
    def __eq__(self, other):
        return (
            isinstance(other, EntryInfo) and self.request_id == other.request_id
        )

@dataclass
class Entry:
    """
    An entry in the buffer.

    This class is used to track the category, rollout data of an entry in the buffer and
    the metadata of the entry.

    Args:
        category (EntryCategory): The category of the entry (EMPTY, RESERVED, OCCUPIED).
        data (Optional[DataProto]): The data associated with the entry.
        entry_info (Optional[EntryInfo]): The metadata of the entry.
    """
    category: EntryCategory
    data: Optional[DataProto] = None
    entry_info: Optional[EntryInfo] = None

class BufferStatus(enum.Enum):
    """Enum for the status of a buffer.
    
    READY: All entries are OCCUPIED and the buffer is ready for training
    STUCK: Mixed OCCUPIED and RESERVED entries with no EMPTY slots  
    PENDING: Has at least one EMPTY slot and is still accepting new entries
    """
    READY = 0    # All entries are OCCUPIED
    STUCK = 1    # Mixed OCCUPIED and RESERVED with no EMPTY slots
    PENDING = 2  # Has at least one EMPTY slot

class StalenessBuffer:
    """Buffer for managing staleness-controlled entries.

    This class manages a fixed-size buffer of entries, each of which can be EMPTY, 
    RESERVED, or OCCUPIED. It provides methods for inserting, deleting, and querying 
    entries, as well as determining the buffer's status for training readiness.

    Args:
        num_entries (int): The number of entries in the buffer
        staleness (int): The staleness tolerance for this buffer
    """
    def __init__(self, num_entries: int, ready_num_entries: int, staleness: int):
        self.num_entries = num_entries
        self.ready_num_entries = ready_num_entries
        self.entries: List[Entry] = [
            Entry(category=EntryCategory.EMPTY) 
            for _ in range(num_entries)
        ]
        self.staleness = staleness

    def get_first_non_occupied(self) -> int:
        """Get the index of the first non-occupied entry.

        Returns:
            int: Index of the first non-occupied entry, or `self.num_entries` if all are OCCUPIED
        """
        for idx in range(self.num_entries):
            if self.entries[idx].category != EntryCategory.OCCUPIED:
                return idx
        return self.num_entries 

    def get_last_non_reserved(self) -> int:
        """
        Returns the index of the last entry that is not RESERVED, or -1 if all are RESERVED.

        Returns:
            int: Index of the last non-reserved entry, or -1 if none found.
        """
        for idx in reversed(range(self.num_entries)):
            if self.entries[idx].category != EntryCategory.RESERVED:
                return idx
        return -1 

    def insert(
        self, 
        entry_id: int, 
        category: EntryCategory,
        data: Optional[DataProto] = None,
        entry_info: Optional[EntryInfo] = None,
    ):
        """
        Insert an entry at the specified position.

        Args:
            entry_id (int): The index to insert the entry at.
            category (EntryCategory): The category of the entry.
            data (DataProto, optional): The data to store in the entry.
            entry_info (EntryInfo, optional): The metadata for the entry.
        Raises:
            AssertionError: If entry_id is out of range.
        """
        assert 0 <= entry_id < self.num_entries, f"Invalid entry ID: {entry_id} for bound [0, {self.num_entries})"

        self.entries[entry_id] = Entry(
            category=category,
            data=data,
            entry_info=entry_info
        )

    def delete(self, entry_id: int):
        """
        Delete (reset) the entry at the specified position, making it EMPTY.

        Args:
            entry_id (int): The index of the entry to delete.
        Raises:
            AssertionError: If entry_id is out of range.
        """
        assert 0 <= entry_id < self.num_entries, f"Invalid entry ID: {entry_id} for bound [0, {self.num_entries})"

        self.entries[entry_id] = Entry(
            category=EntryCategory.EMPTY,
            data=None,
            entry_info=None
        )

    def get_status(self) -> BufferStatus:
        """
        Determine the buffer status based on entry states.
        
        - READY: All entries are occupied
        - STUCK: All entries are reserved/occupied
        - PENDING: At least one EMPTY entry in the buffer

        Returns:
            BufferStatus: The current status of the buffer (READY, STUCK, or PENDING).
        Raises:
            AssertionError: If buffer state invariants are violated.
        """
        first_non_occupied = self.get_first_non_occupied()
        
        # READY state: All entries are occupied
        if first_non_occupied == self.ready_num_entries:
            return BufferStatus.READY
        
        # Check for STUCK state
        last_non_reserved = self.get_last_non_reserved()
        if first_non_occupied == last_non_reserved + 1:
            # Verify all entries before first_non_occupied are OCCUPIED
            # and all entries after last_non_reserved are RESERVED
            assert all(entry.category == EntryCategory.OCCUPIED for entry in self.entries[:first_non_occupied]) \
                and all(entry.category == EntryCategory.RESERVED for entry in self.entries[last_non_reserved+1:]), \
                "STUCK buffer must have all OCCUPIED before first non-occupied and all RESERVED after last non-reserved"
            return BufferStatus.STUCK
        
        # Must be PENDING state - verify at least one EMPTY entry
        assert any(entry.category == EntryCategory.EMPTY for entry in self.entries), \
            "PENDING buffer must have at least one EMPTY entry"
        return BufferStatus.PENDING
    
    def get_empty_entries_num(self) -> int:
        """Count number of EMPTY entries in the buffer"""
        return sum(1 for entry in self.entries if entry.category == EntryCategory.EMPTY)
    
    def get_all_data(self) -> DataProto:
        """
        Get all data in the buffer. All entries must be OCCUPIED (have rollout data).

        Returns:
            DataProto: Concatenated data from all entries in the buffer.
        Raises:
            ValueError: If any entry has not been occupied.
        """
        data_list = []
        for entry in self.entries:
            if entry.data is None:
                raise ValueError("One or more entries have None data. All entries must be occupied.")
            data_list.append(entry.data)
        return DataProto.concat(data_list)
    
    def update_all_data(self, data: DataProto):
        batch_size = len(data)
        assert batch_size == self.num_entries, \
            f"Mismatch between buffer size {self.num_entries} and batch size {batch_size} when updating data"
        data_list = data.chunk(batch_size)
        for entry, data in zip(self.entries, data_list):
            entry.data = data

class StalenessInventory:
    """
    Inventory for managing multiple staleness buffers and their data.

    This class manages a collection of staleness buffers, tracks data and entry mappings, and provides
    methods for reserving, occupying, and consuming buffer entries under staleness constraints.

    Args:
        num_entries (int): Number of entries per buffer.
    """
    def __init__(
        self,
        num_entries: int,
        ready_num_entries: int,
        staleness: int,
        buffer_post_process_fn: Optional[callable] = None,
    ):
        self.staleness = staleness
        self.buffer_id = 0
        self.num_entries = num_entries
        self.ready_num_entries = ready_num_entries
        self.buffer_post_process_fn = buffer_post_process_fn

        self.data_pool: Dict[EntryInfo, DataProto] = {} # Rollout data pool for requests in Group Sampling
        self.buffers: Dict[int, StalenessBuffer] = {}
        self.data_tracker: Dict[EntryInfo, Tuple[int, int]] = {} # Maps entry to location (buffer_id, entry_id)
        # Status tracking for buffer IDs
        # this can reduce the need to iterate through all buffers and call `get_status` frequently
        self._buffer_ids_by_status: Dict[BufferStatus, Set[int]] = {
            status: set() for status in BufferStatus
        }

        # Agent Loop Manager reference
        self.agent_loop_manager: Optional[ray.actor.ActorHandle] = None

    def set_agent_loop_manager(self, agent_loop_manager: ray.actor.ActorHandle):
        """Set the reference to the agent loop manager."""
        self.agent_loop_manager = agent_loop_manager

    def create_buffer(self, buffer_id: int):
        """
        Create a new buffer with the specified ID and fixed-size.

        Args:
            buffer_id (int): The ID of the buffer to create.
        Raises:
            AssertionError: If the buffer already exists.
        """
        assert buffer_id == self.buffer_id, f"Buffer ID {buffer_id} must be the next in sequence (current: {self.buffer_id})"

        buffer = StalenessBuffer(self.num_entries, self.ready_num_entries, self.staleness)
        self.buffers[buffer_id] = buffer
        psrl_logger.debug(f"Created buffer {buffer_id}, current buffer count: {len(self.buffers)}")
        self._update_buffer_status(buffer_id)
        self.buffer_id += 1

    def delete_buffer(self, buffer_id: int):
        """
        Delete the buffer with the specified ID and remove all associated entries from the data tracker.

        Args:
            buffer_id (int): The ID of the buffer to delete.
        """
        if buffer_id not in self.buffers:
            return

        # Remove entries associated with this buffer from data tracker
        entries_to_remove = [
            entry.entry_info for entry in self.buffers[buffer_id].entries
        ]
        for entry_info in entries_to_remove:
            assert entry_info in self.data_tracker, \
                f"Entry info {entry_info} not found in data tracker"
            del self.data_tracker[entry_info]
        # Remove from status tracking
        for status_set in self._buffer_ids_by_status.values():
            status_set.discard(buffer_id)
        # Remove buffer from inventory
        del self.buffers[buffer_id]
            
    def get_buffer_status(self, buffer_id: int) -> BufferStatus:
        """
        Get the status of a specific buffer.

        Args:
            buffer_id (int): The ID of the buffer.
        Returns:
            BufferStatus: The status of the buffer.
        Raises:
            ValueError: If the buffer does not exist or has no status.
        """
        # Use cached status from _buffer_ids_by_status, rather than calling `get_status`
        if buffer_id not in self.buffers:
            raise ValueError(f"Buffer {buffer_id} does not exist")
        for status in BufferStatus:
            if buffer_id in self._buffer_ids_by_status[status]:
                return status
        raise ValueError(f"Buffer {buffer_id} has no status in inventory")

    def _update_buffer_status(self, buffer_id: int):
        """
        Update the internal status tracking for a buffer.

        Args:
            buffer_id (int): The ID of the buffer to update.
        """
        if buffer_id not in self.buffers:
            return
            
        buffer = self.buffers[buffer_id]
        # Remove from the original status track
        for status in BufferStatus:
            if buffer_id in self._buffer_ids_by_status[status]:
                self._buffer_ids_by_status[status].remove(buffer_id)
                break
                
        new_status = buffer.get_status()
        reserve_buffer = True
        if new_status == BufferStatus.READY:
            if self.buffer_post_process_fn:
                reserve_buffer = self._buffer_post_process(buffer_id, buffer)
        
        if reserve_buffer:    
            # Update in the new status track
            self._buffer_ids_by_status[new_status].add(buffer_id)

    def _buffer_post_process(self, buffer_id: int, buffer: StalenessBuffer) -> bool:
        assert self.buffer_post_process_fn is not None, "Buffer post-processing function is not set."
        
        buffer_data = buffer.get_all_data()
        processed_buffer_data = self.buffer_post_process_fn(buffer_data)
        # NOTE(linsh): Current implementation will retry with new global batch if any sample is filtered
        if not processed_buffer_data or len(processed_buffer_data) < self.num_entries:
            psrl_logger.info(f"Post-processing function returned "
                             f"{len(processed_buffer_data) if processed_buffer_data else 0} requests "
                             f"for buffer {buffer_id}, less than {self.num_entries}. Retrying later.")
            uid_of_processed_buffer = processed_buffer_data.non_tensor_batch["uid"].tolist()
            for entry_id in range(self.num_entries):
                data = buffer.entries[entry_id].data
                if int(data.non_tensor_batch["uid"][0]) not in uid_of_processed_buffer:
                    buffer.delete(entry_id)
            self._update_buffer_status(buffer_id)
            pending_buffers = self._buffer_ids_by_status[BufferStatus.PENDING]
            candidate_ids = list(pending_buffers)
            if not candidate_ids:
                waiting_buffer_id = self.buffer_id
            else:
                waiting_buffer_id = min(candidate_ids)

            # Notify agent loop manager to retry new requests
            self.notify_request_retry(waiting_buffer_id)
            return False
        else:
            buffer.update_all_data(processed_buffer_data)
            return True

    def min_ready_buffer_id(self) -> Optional[int]:
        """
        Get the min buffer ID that is in READY state.

        Returns:
            Optional[int]: The min READY buffer ID, or None if none exist.
        """
        ready_buffers = self._buffer_ids_by_status[BufferStatus.READY]
        return min(ready_buffers) if ready_buffers else None

    def max_ready_buffer_id(self) -> Optional[int]:
        """
        Get the max buffer ID that is in READY state.

        Returns:
            Optional[int]: The max READY buffer ID, or None if none exist.
        """
        ready_buffers = self._buffer_ids_by_status[BufferStatus.READY]
        return max(ready_buffers) if ready_buffers else None

    def ready_buffer_ids(self) -> Set[int]:
        """
        Get all buffer IDs that are in READY state.

        Returns:
            Set[int]: A set of READY buffer IDs.
        """
        return self._buffer_ids_by_status[BufferStatus.READY]

    def min_not_ready_buffer_id(self) -> Optional[int]:
        """
        Get the smallest buffer ID that is not in READY state (i.e., STUCK or PENDING).

        Returns:
            Optional[int]: The smallest non-READY buffer ID, or None if none exist.
        """
        non_ready = (
            self._buffer_ids_by_status[BufferStatus.STUCK] | 
            self._buffer_ids_by_status[BufferStatus.PENDING]
        )
        return min(non_ready) if non_ready else None

    def ensure_buffer_exists(self, max_staleness_buffer_id: int):
        """
        Ensure all buffers up to max_buffer_id exist. Create missing buffers as needed.

        Args:
            max_staleness_buffer_id (int): The maximum buffer ID to ensure exists.
        """
        next_buffer_id = self.buffer_id
        for buffer_id in range(next_buffer_id, max_staleness_buffer_id + 1):
            self.create_buffer(buffer_id)

    def get_empty_entries_total_num(
        self,
        max_staleness_buffer_id: int
    ) -> int:
        """
        Count the total number of EMPTY entries in all PENDING buffers up to max_staleness_buffer_id.

        Args:
            max_staleness_buffer_id (int): The maximum buffer ID to consider.
        Returns:
            int: The total number of EMPTY entries in eligible buffers.
        """
        # Ensure at least num_requests EMPTY entries are available before max_staleness_buffer_id
        self.ensure_buffer_exists(max_staleness_buffer_id)

        pending_buffers = self._buffer_ids_by_status[BufferStatus.PENDING]
        return sum(
            self.buffers[bid].get_empty_entries_num() for bid in pending_buffers if bid <= max_staleness_buffer_id
        )

    def add_to_data_pool(
        self,
        entry_info: EntryInfo,
        data: DataProto,
    ):
        """
        Add rollout data to the group data pool for a specific entry.

        Args:
            entry_info (EntryInfo): The entry metadata.
            data (DataProto): The data to add.
        Raises:
            AssertionError: If data for the entry already exists.
        """
        assert entry_info not in self.data_pool, f"Data pool already has data for entry info {entry_info}"

        self.data_pool[entry_info] = data
    
    def update_to_data_pool(
        self,
        entry_info: EntryInfo,
        data: DataProto,
    ):
        """
        Update rollout data to the group data pool for a specific entry.

        Args:
            entry_info (EntryInfo): The entry metadata.
            data (DataProto): The data to add.
        """
        self.data_pool.pop(entry_info, None)
        self.data_pool[entry_info] = data

    def pop_from_data_pool(
        self,
        entry_info: EntryInfo,
    ) -> DataProto:
        """
        Pop rollout data from the group data pool for a specific entry.

        Args:
            entry_info (EntryInfo): The entry metadata.
        Returns:
            DataProto: The popped data.
        Raises:
            AssertionError: If data for the entry does not exist.
        """
        assert entry_info in self.data_pool, f"Data pool must have data for entry info {entry_info}"

        return self.data_pool.pop(entry_info)

    def get_from_data_pool(
        self,
        entry_info: EntryInfo,
    ) -> DataProto:
        """
        Retrieve rollout data from the group data pool for a specific entry.

        Args:
            entry_info (EntryInfo): The entry metadata.
        Returns:
            DataProto: The retrieved data.
        Raises:
            AssertionError: If data for the entry does not exist.
        """
        assert entry_info in self.data_pool, f"Data pool must have data for entry info {entry_info}"

        return self.data_pool[entry_info]

    def remove_from_data_pool(
        self,
        entry_info: EntryInfo,
    ):
        """
        Delete data from the group data pool for a specific entry.

        Args:
            entry_info (EntryInfo): The entry metadata.
        Raises:
            AssertionError: If data for the entry does not exist.
        """
        assert entry_info in self.data_pool, f"Data pool must have data for entry info {entry_info}"

        del self.data_pool[entry_info]

    def reserve_data(
        self, 
        entry_info: EntryInfo, 
        max_staleness_buffer_id: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Reserve an entry for a rollout instance in an appropriate buffer.

        Args:
            entry_info (EntryInfo): The entry metadata to reserve.
            max_staleness_buffer_id (int): The maximum buffer ID allowed by staleness.
        Returns:
            Tuple[Optional[int], Optional[int]]: The buffer ID and entry ID reserved, or (None, None) if not available.
        """
        assert entry_info not in self.data_tracker, f"Entry info {entry_info} must not have existing mapping: {self.data_tracker}"
        # Ensure buffer IDs up to max_staleness_buffer_id exist
        self.ensure_buffer_exists(max_staleness_buffer_id)

        # Get all PENDING buffers within the staleness limit
        pending_buffers = self._buffer_ids_by_status[BufferStatus.PENDING]
        candidate_ids = [
            bid for bid in pending_buffers if bid <= max_staleness_buffer_id
        ]

        if not candidate_ids:
            # Cases where no PENDING buffers are available
            # the rollout instance should wait for a buffer to become available
            # raise RuntimeError("No suitable PENDING buffer found")
            return None, None

        # Select the highest buffer ID for reservation, reserve buffer entry in reversed order
        target_buffer_id = max(candidate_ids)
        buffer = self.buffers[target_buffer_id]
        entry_id = buffer.get_last_non_reserved()
        assert entry_id != -1 and buffer.entries[entry_id].category == EntryCategory.EMPTY, \
            "Found non-reserved entry must be EMPTY"

        # Create entry info and update buffer
        buffer.insert(
            entry_id, 
            EntryCategory.RESERVED, 
            data=None, 
            entry_info=entry_info
        )
        self.data_tracker[entry_info] = (target_buffer_id, entry_id)
        self._update_buffer_status(target_buffer_id)

        return target_buffer_id, entry_id

    def update_request_version_tag(
        self,
        request_id: Union[str, int],
        new_version_tag: int,
    ):
        """
        Update the version tag of a specific request in the data tracker and buffer.
        Args:
            request_id (Union[str, int]): The global unique request ID to update.
            new_version_tag (int): The new model version tag to set.
        Raises:
            AssertionError: If the request ID is not found or the new version tag is out of staleness bounds.
        """

        entry_to_update = None
        for entry_info in self.data_tracker.keys():
            if entry_info.request_id == request_id:
                entry_to_update = entry_info
                break
        assert entry_to_update is not None, f"Request ID {request_id} not found in data tracker"
        
        old_buffer_id, old_entry_id = self.data_tracker[entry_to_update]
        assert old_buffer_id - self.staleness <= new_version_tag <= old_buffer_id, \
            f"New version tag {new_version_tag} is not within staleness bounds for request ID {request_id}"
        
        new_entry_info = EntryInfo(
            rollout_instance_id=entry_to_update.rollout_instance_id,
            request_id=entry_to_update.request_id,
            model_version=new_version_tag
        )
        # Update data tracker with new entry info
        del self.data_tracker[entry_to_update]
        self.data_tracker[new_entry_info] = (old_buffer_id, old_entry_id)
        psrl_logger.debug(f"Updated request ID {entry_to_update.request_id} from version tag {entry_to_update.model_version} to {new_version_tag}")
        # Update entry info in the buffer
        buffer = self.buffers[old_buffer_id]
        buffer.entries[old_entry_id].entry_info = new_entry_info

    def clear_reserved_entries(
        self,
        entry_infos: Union[EntryInfo, List[EntryInfo]],
    ):
        """
        Clear RESERVED entries from buffers and update data tracker.
        It may involve moving other RESERVED entries with the same model version to fill the blanks across buffers.
        
        Args:
            entry_infos (Union[EntryInfo, List[EntryInfo]]): The entry metadata or list of metadata to clear.
        Raises:
            AssertionError: If any entry_info is not tracked.
        """

        if not isinstance(entry_infos, list):
            entry_infos = [entry_infos]
        changed_buffer_ids = []
        for entry_info in entry_infos:
            if entry_info in self.data_tracker:
                buffer_id, entry_id = self.data_tracker[entry_info]
                buffer = self.buffers[buffer_id]
                # Delete the entry from the buffer
                buffer.delete(entry_id)
                del self.data_tracker[entry_info]
                psrl_logger.debug(f"Cleared RESERVED entry {entry_info} from (buffer {buffer_id}, entry {entry_id})")
                changed_buffer_ids.append(buffer_id)
                # Entry movement
                model_version = entry_info.model_version
                min_buffer_id = model_version - self.staleness
                first_reserved_entry_id = None
                exchange_buffer_id = None
                for bid in range(min_buffer_id, buffer_id + 1):
                    if bid not in self.buffers:
                        continue
                    b = self.buffers[bid]
                    for eid, entry in enumerate(b.entries):
                        if (
                            entry.category == EntryCategory.RESERVED and
                            entry.entry_info.model_version == model_version and
                            entry.entry_info not in entry_infos
                        ):
                            first_reserved_entry_id = eid
                            exchange_buffer_id = bid
                            break
                if (
                    first_reserved_entry_id != None and 
                    (exchange_buffer_id < buffer_id or first_reserved_entry_id < entry_id)
                ):
                    exchange_buffer = self.buffers[exchange_buffer_id]
                    first_reserved_entry_info = exchange_buffer.entries[first_reserved_entry_id].entry_info
                    # Move the RESERVED entry to the position of the deleted (i.e., EMPTY) entry
                    buffer.entries[entry_id] = exchange_buffer.entries[first_reserved_entry_id]
                    exchange_buffer.delete(first_reserved_entry_id)
                    # Update data tracker with the new position
                    if first_reserved_entry_info is not None:
                        self.data_tracker[first_reserved_entry_info] = (buffer_id, entry_id)
                        psrl_logger.debug(f"Moved RESERVED entry {first_reserved_entry_info} "
                                          f"from (buffer {exchange_buffer_id}, entry {first_reserved_entry_id}) "
                                          f"to (buffer {buffer_id}, entry {entry_id})")
                    if exchange_buffer_id != buffer_id and exchange_buffer_id not in changed_buffer_ids:
                        changed_buffer_ids.append(exchange_buffer_id)
        for buffer_id in set(changed_buffer_ids):
            self._update_buffer_status(buffer_id)

    def occupy_data_without_reserve(
        self,
        entry_info: EntryInfo,
        data: Optional[DataProto] = None,
    ):
        """
        Append data to an appropriate buffer, occupying it.
        NOTE: Since we simplified the reserve process as `set_version_tag` in rollout server,
        this method is used to occupy the data in the buffer directly, such that the data tracker
        is not used.

        Args:
            entry_info (EntryInfo): The entry metadata to occupy.
            data (Optional[DataProto]): The data to occupy with. If None, will use data from data_pool.
        """
        if data is None:
            # For group sampling, the rollout data is stored in the group data pool.
            assert entry_info in self.data_pool, f"Data pool must have data for entry info {entry_info}"
            data = self.data_pool.pop(entry_info)
            if data is None:
                return

        # Step 2: Get all PENDING buffers within the staleness limit
        pending_buffers = self._buffer_ids_by_status[BufferStatus.PENDING]
        candidate_ids = list(pending_buffers)
        if not candidate_ids:
            buffer_id = self.buffer_id
            self.create_buffer(buffer_id)
            candidate_ids = [buffer_id]
        assert candidate_ids, f"No suitable PENDING buffer found."

        # Step 3: Select the lowest PENDING buffer + EMPTY entry to insert
        target_buffer_id = min(candidate_ids)
        
        # Check staleness constraint
        if entry_info.model_version + self.staleness < target_buffer_id:
            raise ValueError(f"Entry {entry_info} is too stale for buffer {target_buffer_id} with model version {entry_info.model_version}.")

        buffer = self.buffers[target_buffer_id]
        entry_id = buffer.get_first_non_occupied()
        assert entry_id < buffer.num_entries and buffer.entries[entry_id].category == EntryCategory.EMPTY, \
            "Found non-occupied entry must be EMPTY"
        
        # Create entry info and update buffer
        buffer.insert(
            entry_id,
            EntryCategory.OCCUPIED, 
            data=data, 
            entry_info=entry_info
        )
        self._update_buffer_status(target_buffer_id)

    def occupy_data_with_reserve(
        self, 
        entry_info: EntryInfo, 
        data: Optional[DataProto]=None,
    ):
        """
        Move data to the first non-occupied entry in an appropriate buffer, occupying it.

        Args:
            entry_info (EntryInfo): The entry metadata to occupy.
            data (Optional[DataProto]): The data to occupy with. If None, will use data from data_pool.
        Raises:
            AssertionError: If entry_info is not tracked or data is missing.
        """
        assert entry_info in self.data_tracker, f"Entry info {entry_info} must have existing mapping, but {self.data_tracker=}"
        if data is None:
            # For group sampling, the rollout data is stored in the group data pool.
            assert entry_info in self.data_pool, f"Data pool must have data for entry info {entry_info}"
            data = self.data_pool.pop(entry_info)
            if data is None:
                return

        old_buffer_id, old_entry_id = self.data_tracker[entry_info]

        # Step 1: Clean up old entry (may cause entry movement)
        self.clear_reserved_entries(entry_info)

        rollout_instance_id = entry_info.rollout_instance_id
        model_version = entry_info.model_version

        # Step 2: Get all PENDING buffers within the staleness limit
        pending_buffers = self._buffer_ids_by_status[BufferStatus.PENDING]
        candidate_ids = list(pending_buffers)
        candidate_ids = [
            bid for bid in pending_buffers if bid <= model_version + self.staleness
        ]

        assert candidate_ids, f"No suitable PENDING buffer found, but at least buffer {old_buffer_id} should be available for rollout instance {rollout_instance_id}"

        # Step 3: Select the lowest PENDING buffer + EMPTY entry to insert
        # NOTE: For current implementation, if all buffers in front of buffer `old_buffer_id` don't have EMPTY entries,
        # the entry will occupy buffer `old_buffer_id`, instead of exchanging with lowest RESERVED entry.
        target_buffer_id = min(candidate_ids)
        buffer = self.buffers[target_buffer_id]
        entry_id = buffer.get_first_non_occupied()
        assert entry_id < buffer.num_entries and buffer.entries[entry_id].category == EntryCategory.EMPTY, \
            "Found non-occupied entry must be EMPTY"
        
        # Create entry info and update buffer
        buffer.insert(
            entry_id, 
            EntryCategory.OCCUPIED, 
            data=data, 
            entry_info=entry_info
        )
        self.data_tracker[entry_info] = (target_buffer_id, entry_id)
        self._update_buffer_status(target_buffer_id)
     
    def consume_buffer(
        self, 
        buffer_id: int
    ) -> DataProto:
        """
        Consume (retrieve and remove) all data from the specified buffer.

        Args:
            buffer_id (int): The ID of the buffer to consume.
        Returns:
            DataProto: The concatenated data from the buffer.
        Raises:
            AssertionError: If the buffer is not in READY state.
        """
        assert self.get_buffer_status(buffer_id) == BufferStatus.READY, \
            f"Buffer {buffer_id} must be in READY state to consume data"
        
        buffer = self.buffers[buffer_id]
        data = buffer.get_all_data()
        self.delete_buffer(buffer_id)
        return data
    
    def notify_request_retry(self, waiting_buffer_id: int):
        """
        Notify the agent loop manager to retry new requests asynchronously.
        
        Args:
            waiting_buffer_id (int): The buffer ID that is currently waiting for processing.
        """
        assert self.agent_loop_manager is not None, "Agent Loop Manager is not set."

        psrl_logger.debug(f"Notifying agent loop manager to retry new requests for buffer ID {waiting_buffer_id}")
        ray.get(self.agent_loop_manager.retry_request.remote(waiting_buffer_id))
