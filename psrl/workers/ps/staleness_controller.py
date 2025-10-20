import os
import logging
import enum
import numpy as np
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
    rollout_instance_id: Union[str, int, List[Union[str, int]]]  # The ID of the rollout instance this entry belongs to
    prompt_id: Union[str, int] # The global unique prompt ID
    # The model version when generating this entry, which should be within staleness control
    # (i.e., higher than the final occupied buffer ID minus the staleness limit)
    request_idx: Union[str, int, List[Union[str, int]]]  # The global unique request ID inside a group
    model_version: Union[int, List[int]]

    def __hash__(self):
        return hash(self.prompt_id)
    
    def __eq__(self, other):
        return (
            isinstance(other, EntryInfo) and self.prompt_id == other.prompt_id
        )
    
    def get_all_requests(self, rollout_n: int) -> List[int]:
        """Get all request IDs associated with this entry info."""
        if isinstance(self.request_idx, list):
            return [self.prompt_id * rollout_n + idx for idx in self.request_idx]
        else:
            return [self.prompt_id * rollout_n + self.request_idx]

    def get_entry_version(self) -> int:
        """Get the minimum model version associated with this entry info."""
        if isinstance(self.model_version, list):
            return min(self.model_version)
        else:
            return self.model_version

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
    entry_info: Optional[EntryInfo] = None

class BufferStatus(enum.Enum):
    """Enum for the status of a buffer.
    
    READY: All entries are OCCUPIED and the buffer is ready for training
    STUCK: Mixed OCCUPIED and RESERVED entries with no EMPTY slots  
    PENDING: Has at least one EMPTY slot and is still accepting new entries
    """
    READY = 0    # Required entries are OCCUPIED
    STUCK = 1    # Mixed OCCUPIED and RESERVED with no EMPTY slots
    PENDING = 2  # Has at least one EMPTY slot
    READY_WITH_CAPACITY = 3 # Required entries are OCCUPIED, but still has capacity for more entries

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

    def get_reserve_entry_num(self) -> int:
        """Count number of RESERVED entries in the buffer"""
        return np.sum([1 for entry in self.entries if entry.category == EntryCategory.RESERVED])

    def insert(
        self, 
        entry_id: int, 
        category: EntryCategory,
        entry_info: Optional[EntryInfo] = None,
    ):
        """
        Insert an entry at the specified position.

        Args:
            entry_id (int): The index to insert the entry at.
            category (EntryCategory): The category of the entry.
            entry_info (EntryInfo, optional): The metadata for the entry.
        Raises:
            AssertionError: If entry_id is out of range.
        """
        assert 0 <= entry_id < self.num_entries, f"Invalid entry ID: {entry_id} for bound [0, {self.num_entries})"

        self.entries[entry_id] = Entry(
            category=category,
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

        # READY state: data buffer can satisfy training requirements
        if first_non_occupied == self.ready_num_entries:
            if any(entry.category == EntryCategory.EMPTY for entry in self.entries[self.ready_num_entries:]):
                return BufferStatus.READY_WITH_CAPACITY
            else: 
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
    
    def clear(self):
        """Clear all entries in the buffer, resetting them to EMPTY."""
        self.entries = [
            Entry(category=EntryCategory.EMPTY) 
            for _ in range(self.num_entries)
        ]

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
        rollout_n: int,
    ):
        self.staleness = staleness
        self.buffer_id = 0
        self.num_entries = num_entries
        self.ready_num_entries = ready_num_entries
        self.rollout_n = rollout_n

        self.buffers: Dict[int, StalenessBuffer] = {}
        self.data_tracker: Dict[int, Tuple[int, int]] = {} # Maps entry to location (buffer_id, entry_id)
        # Status tracking for buffer IDs
        # this can reduce the need to iterate through all buffers and call `get_status` frequently
        self._buffer_ids_by_status: Dict[BufferStatus, Set[int]] = {
            status: set() for status in BufferStatus
        }

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
        psrl_logger.info(f"[Buffer Create]: buffer {buffer_id} created, current buffer IDs: {self.buffers.keys()}")
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
            entry.entry_info for entry in self.buffers[buffer_id].entries if entry.category != EntryCategory.EMPTY
        ]
        for entry_info in entries_to_remove:
            assert entry_info.prompt_id in self.data_tracker, \
                f"Entry info {entry_info} not found in data tracker"
            del self.data_tracker[entry_info.prompt_id]
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

    def get_buffers_with_capacity(self) -> Set[int]:
        """
        Get all buffer IDs that are either READY_WITH_CAPACITY or PENDING.

        Returns:
            Set[int]: A set of buffer IDs with capacity for new entries.
        """
        return (self._buffer_ids_by_status[BufferStatus.PENDING] |
                self._buffer_ids_by_status[BufferStatus.READY_WITH_CAPACITY])

    def _update_buffer_status(self, buffer_id: int) -> BufferStatus:
        """
        Update the internal status tracking for a buffer.

        Args:
            buffer_id (int): The ID of the buffer to update.
        Returns:
            BufferStatus: The new status of the buffer after update.
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
        self._buffer_ids_by_status[new_status].add(buffer_id)
        return new_status

    def min_ready_buffer_id(self) -> Optional[int]:
        """
        Get the min buffer ID that is in READY state.

        Returns:
            Optional[int]: The min READY buffer ID, or None if none exist.
        """
        ready_buffers = self.ready_buffer_ids()
        return min(ready_buffers) if ready_buffers else None

    def max_ready_buffer_id(self) -> Optional[int]:
        """
        Get the max buffer ID that is in READY state.

        Returns:
            Optional[int]: The max READY buffer ID, or None if none exist.
        """
        ready_buffers = self.ready_buffer_ids()
        return max(ready_buffers) if ready_buffers else None

    def ready_buffer_ids(self) -> Set[int]:
        """
        Get all buffer IDs that are in READY or READY_WITH_CAPACITY state.

        Returns:
            Set[int]: A set of READY buffer IDs.
        """
        return (self._buffer_ids_by_status[BufferStatus.READY] |
                self._buffer_ids_by_status[BufferStatus.READY_WITH_CAPACITY])

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

        pending_buffers = self.get_buffers_with_capacity()
        return sum(
            self.buffers[bid].get_empty_entries_num() for bid in pending_buffers if bid <= max_staleness_buffer_id
        )

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
        if entry_info.prompt_id in self.data_tracker:
            buffer_id, entry_id = self.data_tracker[entry_info.prompt_id]
            psrl_logger.debug(f"[Entry Reserve Update]: entry {entry_info} already reserved in (buffer {buffer_id}, entry {entry_id})")
            tracked_entry_info = self.buffers[buffer_id].entries[entry_id].entry_info
            if not isinstance(tracked_entry_info.request_idx, list):
                tracked_entry_info.request_idx = [tracked_entry_info.request_idx]
            entry_request_idx = entry_info.request_idx
            assert entry_request_idx not in tracked_entry_info.request_idx, \
                f"Entry info {entry_info} already reserved in (buffer {buffer_id}, entry {entry_id})"
            tracked_entry_info.request_idx.append(entry_request_idx)
            
            # Update model version
            if not isinstance(tracked_entry_info.model_version, list) and tracked_entry_info.model_version != entry_info.model_version:
                tracked_entry_info.model_version = [tracked_entry_info.model_version] * len(tracked_entry_info.request_idx)
                tracked_entry_info.model_version.append(entry_info.model_version)
            elif isinstance(tracked_entry_info.model_version, list):
                tracked_entry_info.model_version.append(entry_info.model_version)
            
            # Update rollout instance id
            if not isinstance(tracked_entry_info.rollout_instance_id, list) and tracked_entry_info.rollout_instance_id != entry_info.rollout_instance_id:
                tracked_entry_info.rollout_instance_id = [tracked_entry_info.rollout_instance_id] * (len(tracked_entry_info.request_idx) - 1)
                tracked_entry_info.rollout_instance_id.append(entry_info.rollout_instance_id)
            elif isinstance(tracked_entry_info.rollout_instance_id, list):
                tracked_entry_info.rollout_instance_id.append(entry_info.rollout_instance_id)
            
            self.buffers[buffer_id].entries[entry_id].entry_info = tracked_entry_info
            return buffer_id, entry_id

        # Ensure buffer IDs up to max_staleness_buffer_id exist
        self.ensure_buffer_exists(max_staleness_buffer_id)

        # Get all PENDING buffers within the staleness limit
        pending_buffers = self.get_buffers_with_capacity()
        candidate_ids = [
            bid for bid in pending_buffers if max_staleness_buffer_id - self.staleness <= bid <= max_staleness_buffer_id
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
            f"Found non-reserved entry must be EMPTY, but got {buffer.entries[entry_id]} in (buffer {target_buffer_id}, entry {entry_id})"

        # Create entry info and update buffer
        buffer.insert(
            entry_id, 
            EntryCategory.RESERVED, 
            entry_info=entry_info
        )
        self.data_tracker[entry_info.prompt_id] = (target_buffer_id, entry_id)
        self._update_buffer_status(target_buffer_id)
        
        psrl_logger.debug(f"[Entry Reserve]: entry {entry_info} reserved in (buffer {target_buffer_id}, entry {entry_id})")

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

        prompt_id = request_id // self.rollout_n
        request_idx = request_id % self.rollout_n
        entry_to_update = None
        if prompt_id not in self.data_tracker:
            raise AssertionError(f"Prompt ID {prompt_id} not found in data tracker")

        buffer_id, entry_id = self.data_tracker[prompt_id]
        entry_to_update = self.buffers[buffer_id].entries[entry_id].entry_info
        
        if isinstance(entry_to_update.model_version, list):
            request_idx_in_list = entry_to_update.request_idx.index(request_idx)
            entry_to_update.model_version[request_idx_in_list] = new_version_tag
        elif entry_to_update.model_version != new_version_tag:
            entry_to_update.model_version = [entry_to_update.model_version] * len(entry_to_update.request_idx)
            request_idx_in_list = entry_to_update.request_idx.index(request_idx)
            entry_to_update.model_version[request_idx_in_list] = new_version_tag
        psrl_logger.debug(f"Updated version tag of request idx {request_idx} in entry {entry_to_update} ")
        min_model_version = entry_to_update.get_entry_version()

        self.buffers[buffer_id].entries[entry_id].entry_info = entry_to_update
        
        psrl_logger.debug(f"[Entry Update]: request idx {request_idx} in entry {entry_to_update} updated from "
                          f"version tag {entry_to_update.model_version} to {new_version_tag} "
                          f"in (buffer {buffer_id}, entry {entry_id})")

    def clear_buffer(
        self,
        buffer_id: int,
    ):
        """
        Clear all entries in the buffer, resetting them to EMPTY and updating the data tracker.
        
        Args:
            buffer_id (int): The ID of the buffer to clear.
        """
        entries = self.buffers[buffer_id].entries
        entry_infos = [entry.entry_info for entry in entries]
        self.buffers[buffer_id].clear()
        for entry_info in entry_infos:
            del self.data_tracker[entry_info.prompt_id]
        self._update_buffer_status(buffer_id)

    def clear_occupied_entries(
        self,
        prompt_ids: Union[int, List[int]],
    ):
        """
        Clear OCCUPIED entries from buffers and update data tracker.
        During clearing, move the last OCCUPIED entry to fill the cleared entry to maintain contiguity.
        
        Args:
            prompt_ids (Union[int, List[int]]): The prompt IDs to clear.
        """
        if not isinstance(prompt_ids, list):
            prompt_ids = [prompt_ids]
        updated_buffer_ids = []
        for prompt_id in prompt_ids:
            assert prompt_id in self.data_tracker, f"Prompt ID {prompt_id} must be tracked to clear occupied entries"
            buffer_id, entry_id = self.data_tracker[prompt_id]
            buffer = self.buffers[buffer_id]
            last_occupied_entry_id = buffer.get_first_non_occupied() - 1
            assert 0 <= entry_id <= last_occupied_entry_id, \
                f"Entry ID {entry_id} to clear must be OCCUPIED in (buffer {buffer_id}, entry {entry_id}), " \
                f"but last occupied entry ID is {last_occupied_entry_id}"
            # Delete the entry from the buffer
            buffer.delete(entry_id)
            del self.data_tracker[prompt_id]
            psrl_logger.debug(f"[Entry Clear]: entry {entry_info} cleared from (buffer {buffer_id}, entry {entry_id})")
            
            if entry_id < last_occupied_entry_id:
                # Move the last OCCUPIED entry to the position of the deleted (i.e., EMPTY) entry
                buffer.entries[entry_id] = buffer.entries[last_occupied_entry_id]
                buffer.delete(last_occupied_entry_id)
                moved_entry_info = buffer.entries[entry_id].entry_info
                assert moved_entry_info is not None, f"Moved entry must not be None"
                # Update data tracker with the new position
                self.data_tracker[moved_entry_info.prompt_id] = (buffer_id, entry_id)
                psrl_logger.debug(f"[Entry Move]: entry {moved_entry_info} moved "
                                  f"from (buffer {buffer_id}, entry {last_occupied_entry_id}) "
                                  f"to (buffer {buffer_id}, entry {entry_id})")
            updated_buffer_ids.append(buffer_id)
        
        for buffer_id in set(updated_buffer_ids):
            self._update_buffer_status(buffer_id)

    def clear_reserved_entries(
        self,
        prompt_ids: Union[int, List[int]],
        move_across_buffer: bool = False,
    ):
        """
        Clear RESERVED entries from buffers and update data tracker.
        It may involve moving other RESERVED entries with the same model version to fill the blanks across buffers.
        
        Args:
            prompt_ids (Union[int, List[int]]): The prompt IDs to clear.
            move_across_buffer (bool): Whether to move RESERVED entries across buffers to fill cleared entries.
                Currently only used for OCCUPY operation.
        """

        if not isinstance(prompt_ids, list):
            prompt_ids = [prompt_ids]
        
        changed_buffer_ids = set()
        for prompt_id in prompt_ids:
            if prompt_id not in self.data_tracker:
                continue
            buffer_id, entry_id = self.data_tracker[prompt_id]
            buffer = self.buffers[buffer_id]
            entry_info = buffer.entries[entry_id].entry_info
            if not move_across_buffer:
                last_reserved_entry_id = buffer.get_last_non_reserved() + 1
                # Delete the entry from the buffer
                buffer.delete(entry_id)
                del self.data_tracker[prompt_id]
                psrl_logger.debug(f"[Entry Clear]: entry {entry_info} cleared from (buffer {buffer_id}, entry {entry_id})")
                changed_buffer_ids.add(buffer_id)
                # Move the last RESERVED entry to the position of the deleted (i.e., EMPTY) entry
                if entry_id > last_reserved_entry_id:
                    buffer.entries[entry_id] = buffer.entries[last_reserved_entry_id]
                    buffer.delete(last_reserved_entry_id)
                    moved_entry_info = buffer.entries[entry_id].entry_info
                    assert moved_entry_info is not None, f"Moved entry must not be None"
                    # Update data tracker with the new position
                    self.data_tracker[moved_entry_info.prompt_id] = (buffer_id, entry_id)
                    psrl_logger.debug(f"[Entry Move]: entry {moved_entry_info} moved "
                                      f"from (buffer {buffer_id}, entry {last_reserved_entry_id}) "
                                      f"to (buffer {buffer_id}, entry {entry_id})")
            else:
                # Entry movement
                # Use minimum version to represent the staleness constraint in group sampling
                model_version = entry_info.get_entry_version()
                min_buffer_id = model_version - self.staleness
                first_reserved_entry_id = None
                exchange_buffer_id = None
                # Move RESERVED entries from other buffers within staleness limit
                for bid in range(model_version, model_version + self.staleness + 1):
                    if (
                        bid not in self.buffers or
                        self.buffers[bid].get_status() in [BufferStatus.READY, BufferStatus.READY_WITH_CAPACITY]
                    ):
                        continue
                    b = self.buffers[bid]
                    for eid, entry in enumerate(b.entries):
                        if entry.category == EntryCategory.RESERVED:
                            entry_model_version = entry.get_entry_version()
                            if (
                                entry_model_version >= buffer_id - self.staleness and
                                entry_model_version <= buffer_id and
                                entry.entry_info.prompt_id not in prompt_ids
                            ):
                                first_reserved_entry_id = eid
                                exchange_buffer_id = bid
                                break
                    # Indicate that we have found the first reserved entry to exchange
                    if first_reserved_entry_id is not None and exchange_buffer_id is not None:
                        break

                if (
                    first_reserved_entry_id is not None and 
                    (exchange_buffer_id != buffer_id or first_reserved_entry_id != entry_id)
                ):
                    exchange_buffer = self.buffers[exchange_buffer_id]
                    last_reserved_entry_id = exchange_buffer.get_last_non_reserved() + 1
                    # Delete the entry from the buffer
                    buffer.delete(entry_id)
                    del self.data_tracker[prompt_id]
                    psrl_logger.debug(f"[Entry Clear]: entry {entry_info} cleared from (buffer {buffer_id}, entry {entry_id})")
                    changed_buffer_ids.add(buffer_id)
                    first_reserved_entry_info = exchange_buffer.entries[first_reserved_entry_id].entry_info
                    assert first_reserved_entry_info is not None, f"First reserved entry to move must not be None"
                    # Move the RESERVED entry to the position of the deleted (i.e., EMPTY) entry
                    buffer.entries[entry_id] = exchange_buffer.entries[first_reserved_entry_id]
                    exchange_buffer.delete(first_reserved_entry_id)
                    # Update data tracker with the new position
                    self.data_tracker[first_reserved_entry_info.prompt_id] = (buffer_id, entry_id)
                    psrl_logger.debug(f"[Entry Move]: entry {first_reserved_entry_info} moved "
                                      f"from (buffer {exchange_buffer_id}, entry {first_reserved_entry_id}) "
                                      f"to (buffer {buffer_id}, entry {entry_id})")

                    # Move the last RESERVED entry in the same buffer to the position of the deleted (i.e., EMPTY) entry
                    if first_reserved_entry_id > last_reserved_entry_id:
                        exchange_buffer.entries[first_reserved_entry_id] = exchange_buffer.entries[last_reserved_entry_id]
                        exchange_buffer.delete(last_reserved_entry_id)
                        moved_entry_info = exchange_buffer.entries[first_reserved_entry_id].entry_info
                        assert moved_entry_info is not None, f"Moved entry must not be None"
                        # Update data tracker with the new position
                        self.data_tracker[moved_entry_info.prompt_id] = (exchange_buffer_id, first_reserved_entry_id)
                        psrl_logger.debug(f"[Entry Move]: entry {moved_entry_info} moved "
                                          f"from (buffer {exchange_buffer_id}, entry {last_reserved_entry_id}) "
                                          f"to (buffer {exchange_buffer_id}, entry {first_reserved_entry_id})")
                    
                    if exchange_buffer_id not in changed_buffer_ids:
                        changed_buffer_ids.add(exchange_buffer_id)
                else:
                    # Delete the entry from the buffer
                    buffer.delete(entry_id)
                    del self.data_tracker[prompt_id]
                    psrl_logger.debug(f"[Entry Clear]: entry {entry_info} cleared from (buffer {buffer_id}, entry {entry_id})")
                    changed_buffer_ids.add(buffer_id)

        for buffer_id in changed_buffer_ids:
            self._update_buffer_status(buffer_id)

    @deprecated("This method is deprecated. Use `occupy_data_with_reserve` instead.")
    def occupy_data_without_reserve(
        self,
        entry_info: EntryInfo,
    ):
        """
        Append data to an appropriate buffer, occupying it.
        NOTE: Since we simplified the reserve process as `set_version_tag` in rollout server,
        this method is used to occupy the data in the buffer directly, such that the data tracker
        is not used.

        Args:
            entry_info (EntryInfo): The entry metadata to occupy.
        """

        # Get all PENDING buffers within the staleness limit
        pending_buffers = self.get_buffers_with_capacity()
        candidate_ids = list(pending_buffers)
        if not candidate_ids:
            buffer_id = self.buffer_id
            self.create_buffer(buffer_id)
            candidate_ids = [buffer_id]
        assert candidate_ids, f"No suitable PENDING buffer found."

        # Select the lowest PENDING buffer + EMPTY entry to insert
        target_buffer_id = min(candidate_ids)
        
        # Check staleness constraint
        if entry_info.model_version + self.staleness < target_buffer_id:
            raise ValueError(f"Entry {entry_info} is too stale for buffer {target_buffer_id} with model version {entry_info.model_version}.")

        buffer = self.buffers[target_buffer_id]
        entry_id = buffer.get_first_non_occupied()
        assert entry_id < buffer.num_entries and buffer.entries[entry_id].category == EntryCategory.EMPTY, \
            f"Found non-occupied entry must be EMPTY, but got {buffer.entries[entry_id]} in (buffer {target_buffer_id}, entry {entry_id})"
        
        # Create entry info and update buffer
        buffer.insert(
            entry_id,
            EntryCategory.OCCUPIED, 
            entry_info=entry_info
        )
        self._update_buffer_status(target_buffer_id)

    def occupy_data_with_reserve(
        self, 
        prompt_id: int,
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Move data to the first non-occupied entry in an appropriate buffer, occupying it.

        Args:
            entry_info (EntryInfo): The entry metadata to occupy.
            data (Optional[DataProto]): The data to occupy with. If None, will use data from data_pool.
        Returns:
            Tuple[Optional[int], Optional[int], Optional[int]]: The buffer ID, entry ID, and occupy number after occupying,
                or (None, None, None) if no suitable buffer/entry is available.
        """
        assert prompt_id in self.data_tracker, f"Prompt {prompt_id} must have existing mapping, but {self.data_tracker=}"

        old_buffer_id, old_entry_id = self.data_tracker[prompt_id]
        entry_info = self.buffers[old_buffer_id].entries[old_entry_id].entry_info

        model_version = entry_info.get_entry_version()
        min_buffer_id = model_version - self.staleness
        have_occupy_target = False
        for buffer_id in range(min_buffer_id, model_version + self.staleness + 1):
            if buffer_id not in self.buffers:
                continue
            buffer = self.buffers[buffer_id]
            if buffer.get_first_non_occupied() < buffer.ready_num_entries:
                have_occupy_target = True
                break

        if not have_occupy_target:
            return None, None, None

        # Step 1: Clean up old entry (may cause entry movement)
        self.clear_reserved_entries(prompt_id, move_across_buffer=True)

        model_version = entry_info.get_entry_version()

        # Step 2: Get all PENDING buffers within the staleness limit
        pending_buffers = self.get_buffers_with_capacity()
        candidate_ids = list(pending_buffers)
        candidate_ids = [
            bid for bid in pending_buffers if (
                model_version <= bid <= model_version + self.staleness and 
                self.buffers[bid].get_first_non_occupied() < self.buffers[bid].ready_num_entries
            )
        ]

        assert candidate_ids, \
            f"No suitable PENDING buffer found during occupy prompt {prompt_id}, " \
            f"which was reserved in (buffer {old_buffer_id}, entry {old_entry_id}). " \
            f"After clear, at least buffer {old_buffer_id} should be available"

        # Step 3: Select the lowest PENDING buffer + EMPTY entry to insert
        # NOTE: For current implementation, if all buffers in front of buffer `old_buffer_id` don't have EMPTY entries,
        # the entry will occupy buffer `old_buffer_id`, instead of exchanging with lowest RESERVED entry.
        target_buffer_id = min(candidate_ids)
        buffer = self.buffers[target_buffer_id]
        entry_id = buffer.get_first_non_occupied()
        assert entry_id < buffer.num_entries and buffer.entries[entry_id].category == EntryCategory.EMPTY, \
            f"Found non-occupied entry must be EMPTY, but got {buffer.entries[entry_id]} in (buffer {target_buffer_id}, entry {entry_id})"
        
        # Create entry info and update buffer
        buffer.insert(
            entry_id, 
            EntryCategory.OCCUPIED, 
            entry_info=entry_info
        )
        self.data_tracker[entry_info.prompt_id] = (target_buffer_id, entry_id)
        psrl_logger.debug(f"[Entry Occupy]: entry {entry_info} occupied in (buffer {target_buffer_id}, entry {entry_id})")
        occupy_num = buffer.get_first_non_occupied()
        buffer_status = self._update_buffer_status(target_buffer_id)
        return target_buffer_id, entry_id, occupy_num
