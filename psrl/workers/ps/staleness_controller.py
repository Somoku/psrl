import os
import logging
import enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Tuple, Set
from verl import DataProto

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

class EntryCategory(enum.Enum):
    EMPTY = 0
    RESERVED = 1
    OCCUPIED = 2


# Note:The model version is not a part of the hashing key, because model version may be updated after the reservation but before the occupation
@dataclass(frozen=True)
class EntryInfo:
    rollout_instance_id: Union[str, int]  # The ID of the rollout instance this entry belongs to
    request_id: Union[str, int] # The unique request ID for the rollout instance, used to track requests within the same instance
    model_version: int # The model version when generating this entry, which should be higher than the buffer ID minus the staleness limit
    
    def __hash__(self):
        return hash(self.request_id)
    
    def __eq__(self, other):
        return (
            isinstance(other, EntryInfo) and self.request_id == other.request_id
        )


@dataclass
class Entry:
    category: EntryCategory
    data: Optional[DataProto] = None
    entry_info: Optional[EntryInfo] = None


class BufferStatus(enum.Enum):
    READY = 0    # All entries are OCCUPIED
    STUCK = 1    # Mixed OCCUPIED and RESERVED with no EMPTY slots
    PENDING = 2  # Has at least one EMPTY slot


class StalenessBuffer:
    def __init__(self, num_entries: int):
        self.num_entries = num_entries
        self.entries: List[Entry] = [
            Entry(category=EntryCategory.EMPTY) 
            for _ in range(num_entries)
        ]

    def get_first_non_occupied(self) -> int:
        """Returns first non-occupied entry index or num_entries if none found"""
        for idx in range(self.num_entries):
            if self.entries[idx].category != EntryCategory.OCCUPIED:
                return idx
        return self.num_entries 

    def get_last_non_reserved(self) -> int:
        """Returns last non-reserved entry index or -1 if none found"""
        for idx in reversed(range(self.num_entries)):
            if self.entries[idx].category != EntryCategory.RESERVED:
                return idx
        return -1 

    def insert(
        self, 
        entry_id: int, 
        category: EntryCategory,
        data: DataProto = None,
        entry_info: EntryInfo = None
    ):
        """Insert entry at specified position"""
        assert 0 <= entry_id < self.num_entries, "Invalid entry ID"
        self.entries[entry_id] = Entry(
            category=category,
            data=data,
            entry_info=entry_info
        )

    def delete(self, entry_id: int):
        """Delete entry at specified position"""
        assert 0 <= entry_id < self.num_entries, "Invalid entry ID"
        self.entries[entry_id] = Entry(
            category=EntryCategory.EMPTY,
            data=None,
            entry_info=None
        )

    def get_status(self) -> BufferStatus:
        """Determine buffer status based on entry states"""
        first_non_occupied = self.get_first_non_occupied()
        
        # READY state: All entries are occupied
        if first_non_occupied == self.num_entries:
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
        """Get all data in the buffer"""
        # Must guarantee all entries are OCCUPIED (have data)
        data_list = []
        for entry in self.entries:
            if entry.data is None:
                raise ValueError("One or more entries have None data. All entries must be occupied.")
            data_list.append(entry.data)
        return DataProto.concat(data_list)


class StalenessInventory:
    def __init__(self, num_entries: int):
        self.num_entries = num_entries
        self.data_pool: Dict[EntryInfo, DataProto] = {}
        self.buffers: Dict[int, StalenessBuffer] = {}
        self.data_tracker: Dict[EntryInfo, Tuple[int, int]] = {}
        # Status tracking for buffer IDs
        # this can reduce the need to iterate through all buffers and call `get_status` frequently
        self._buffer_ids_by_status: Dict[BufferStatus, Set[int]] = {
            status: set() for status in BufferStatus
        }

    def create_buffer(self, buffer_id: int):
        """Create new buffer with specified ID and size"""
        assert buffer_id not in self.buffers, "Buffer already exists"
        buffer = StalenessBuffer(self.num_entries)
        self.buffers[buffer_id] = buffer
        self._update_buffer_status(buffer_id)

    def delete_buffer(self, buffer_id: int):
        """Delete buffer with specified ID"""
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
        """Get status of a specific buffer"""
        # Use cached status from _buffer_ids_by_status, rather than calling `get_status`
        if buffer_id not in self.buffers:
            raise ValueError(f"Buffer {buffer_id} does not exist")
        for status in BufferStatus:
            if buffer_id in self._buffer_ids_by_status[status]:
                return status
        raise ValueError(f"Buffer {buffer_id} has no status in inventory")

    def _update_buffer_status(self, buffer_id: int):
        """Update internal status tracking for a buffer"""
        if buffer_id not in self.buffers:
            return
            
        buffer = self.buffers[buffer_id]
        for status in BufferStatus:
            if buffer_id in self._buffer_ids_by_status[status]:
                self._buffer_ids_by_status[status].remove(buffer_id)
                break
                
        new_status = buffer.get_status()
        self._buffer_ids_by_status[new_status].add(buffer_id)
     
    def min_ready_buffer_id(self) -> Optional[int]:
        """Get smallest buffer ID that is in READY state"""
        ready_buffers = self._buffer_ids_by_status[BufferStatus.READY]
        return min(ready_buffers) if ready_buffers else None

    def min_not_ready_buffer_id(self) -> Optional[int]:
        """Get smallest buffer ID that is not in READY state"""
        non_ready = (
            self._buffer_ids_by_status[BufferStatus.STUCK] | 
            self._buffer_ids_by_status[BufferStatus.PENDING]
        )
        return min(non_ready) if non_ready else None

    def ensure_buffer_exists(self, max_staleness_buffer_id: int):
        """Ensure all buffers up to max_buffer_id exist"""
        curr_max_buffer_id = max(self.buffers.keys(), default=-1)
        for buffer_id in range(curr_max_buffer_id + 1, max_staleness_buffer_id + 1):
            self.create_buffer(buffer_id)

    def get_empty_entries_total_num(
        self,
        max_staleness_buffer_id: int
    ) -> int:
        """Check if enough buffers are available to reserve num_requests"""
        # Ensure at least num_requests EMPTY entries are available before max_staleness_buffer_id
        self.ensure_buffer_exists(max_staleness_buffer_id)
        pending_buffers = self._buffer_ids_by_status[BufferStatus.PENDING]
        return sum(
            self.buffers[bid].get_empty_entries_num() for bid in pending_buffers if bid <= max_staleness_buffer_id
        )

    def add_data(
        self,
        entry_info: EntryInfo,
        data: DataProto,
    ):
        assert entry_info not in self.data_pool, f"Data pool already has data for entry info {entry_info}"
        self.data_pool[entry_info] = data

    def remove_data(
        self,
        entry_info: EntryInfo
    ):
        """Delete data from data pool"""
        assert entry_info in self.data_pool, f"Data pool must have data for entry info {entry_info}"
        del self.data_pool[entry_info]

    def reserve_data(
        self, 
        entry_info: EntryInfo, 
        max_staleness_buffer_id: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """Reserve entry for instance in appropriate buffer"""
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

        # Select the highest buffer ID
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

    def occupy_data(
        self, 
        entry_info: EntryInfo, 
        data: Optional[DataProto]=None,
    ):
        """Move data to first non-occupied entry in appropriate buffer"""
        assert entry_info in self.data_tracker, f"Entry info {entry_info} must have existing mapping, but {self.data_tracker=}"
        if data is None:
            assert entry_info in self.data_pool, f"Data pool must have data for entry info {entry_info}"
            data = self.data_pool.pop(entry_info)

        old_buffer_id, old_entry_id = self.data_tracker[entry_info]
        old_buffer = self.buffers[old_buffer_id]

        # Step 1: Clean up old entry (may cause entry movement)
        old_buffer.delete(old_entry_id)
        del self.data_tracker[entry_info]
        # Caution: this will make an intermediate status for the buffer!
        # Need to move the smallest RESERVED entry in this buffer (if existed) to the deleted (i.e., EMPTY) entry
        # Otherwise the RESERVED and EMPTY entries will be criss-crossed in the buffer and hard to manage
        first_reserved_entry_id = None
        for entry_id, entry in enumerate(old_buffer.entries):
            if entry.category == EntryCategory.RESERVED:
                first_reserved_entry_id = entry_id
                break
        if first_reserved_entry_id != None and first_reserved_entry_id < old_entry_id:
            first_reserved_entry_info = old_buffer.entries[first_reserved_entry_id].entry_info
            # Move the RESERVED entry to the position of the deleted (i.e., EMPTY) entry
            old_buffer.entries[old_entry_id] = old_buffer.entries[first_reserved_entry_id]
            old_buffer.delete(first_reserved_entry_id)
            # Update data tracker with the new position
            self.data_tracker[first_reserved_entry_info] = (old_buffer_id, old_entry_id)
        self._update_buffer_status(old_buffer_id)

        # Step 2: Get all PENDING buffers within the staleness limit
        pending_buffers = self._buffer_ids_by_status[BufferStatus.PENDING]
        candidate_ids = [
            bid for bid in pending_buffers if bid <= old_buffer_id
        ]

        assert candidate_ids, f"No suitable PENDING buffer found, but at least buffer {old_buffer_id} should be available for rollout instance {rollout_instance_id}"

        # Step 3: Select the lowest PENDING buffer + EMPTY entry to insert
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
        """Consume data from specified buffer"""
        assert self.get_buffer_status(buffer_id) == BufferStatus.READY, \
            f"Buffer {buffer_id} must be in READY state to consume data"
        
        buffer = self.buffers[buffer_id]
        data = buffer.get_all_data()
        self.delete_buffer(buffer_id)
        return data
        