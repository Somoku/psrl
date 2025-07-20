import os
import logging
import enum
from enum import Enum, auto
from typing import Union, List, Optional, Any
from threading import Event, Lock

from ray.util.queue import Queue as RayQueue

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

class CommandType(Enum):
    """
    Enum for different command types used in the server.
    STOP: Stop the server.
    RESUME: Resume the server.
    SHUTDOWN: Shutdown the server gracefully.
    SYNC: Synchronize the server state with the latest weights.
    ABORT: Abort requests running in the server.
    CHECK_AND_SYNC: Check if the server needs to sync with the latest weights and do so if necessary.
    """
    STOP = enum.auto()
    RESUME = enum.auto()
    SHUTDOWN = enum.auto()
    SYNC = enum.auto()
    ABORT = enum.auto()
    CHECK_AND_SYNC = enum.auto()

class Command:
    """
    A command that can be sent to the server.
    
    Attributes:
        type (CommandType): The type of the command.
        _args (dict): Arguments for the command.
        _kwargs (dict): Keyword arguments for the command.
    
    Methods:
        get_args(): Returns the arguments of the command.
        get_kwargs(): Returns the keyword arguments of the command.
    """
    __slots__ = ("type", "_args", "_kwargs")

    def __init__(self, type, **kwargs):
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "_args", {})
        object.__setattr__(self, "_kwargs", {})
        
        for key, value in kwargs.items():
            if key == "_kwargs":
                if not isinstance(value, dict):
                    raise ValueError(f"_kwargs must be a dict, got {type(value)}")
                self._kwargs = value
            else:
                self._args[key] = value

    def __repr__(self):
        return f"RolloutCommand(type={self.type}, args={self._args}, kwargs={self._kwargs})"

    def get_args(self):
        """Get the arguments of the command."""
        return self._args

    def get_kwargs(self):
        """Get the keyword arguments of the command."""
        return self._kwargs

    def __getattr__(self, item):
        if item in {"type", "_args", "_kwargs"}:
            return object.__getattribute__(self, item)
        
        args = object.__getattribute__(self, "_args")
        if item in args:
            return args[item]
        
        meta = object.__getattribute__(self, "_kwargs")
        if item in meta:
            return meta[item]
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
    
    def __setattr__(self, key, value):
        if key in {"type", "_args", "_kwargs"}:
            object.__setattr__(self, key, value)
            return
        
        args = object.__getattribute__(self, "_args")
        if key in args:
            args[key] = value
        else:
            meta = object.__getattribute__(self, "_kwargs")
            meta[key] = value
    
    def __getstate__(self):
        return {
            "type": self.type,
            "_args": self._args,
            "_kwargs": self._kwargs
        }
    
    def __setstate__(self, state):
        object.__setattr__(self, "type", state["type"])
        object.__setattr__(self, "_args", state["_args"])
        object.__setattr__(self, "_kwargs", state["_kwargs"])

class CommandEvent:
    """
    A class to handle command events.
    
    Attributes:
        command_id (int): The ID of the command.
        event (Event): The event associated with the command.
    """
    def __init__(self, command_id: int, event: Optional[Event] = None):
        self.command_id = command_id
        self.event = event if event is not None else Event()

    def wait(self, timeout=None):
        """Wait for the command event to be set."""
        self.event.wait(timeout)

    def set(self):
        """Set the command event."""
        self.event.set()

    def clear(self):
        """Clear the command event."""
        self.event.clear()

    def is_set(self):
        """Check if the command event is set."""
        return self.event.is_set()

    def __repr__(self):
        return f"CommandEvent(command_id={self.command_id}, event={self.event})"

class CommandExtension:
    def __init__(self):
        """
        Initialize the CommandExtension with threading locks to handle command results and events.
        """
        self._lock = Lock()
        self._command_results = {}
        self._command_events = {}
        self._command_counter = 0
        self.command_queue = RayQueue()  # For async commands like abort

    def exec_command(self, command: Command, timeout=None, blocking=True):
        """
        Add a command to the command queue and wait for its completion within the specified timeout.
        
        Args:
            command (Command): The command to be executed.
            timeout (float, optional): The maximum time to wait for the command to complete when `blocking` is True.
                                       If None, waits indefinitely.
            blocking (bool): If True, blocks until the command is executed. If False, returns immediately.
        
        Returns:
            Union[CommandEvent, dict[int, DataProto], None]: If `blocking` is True, returns a CommandEvent or the command result.
            If `blocking` is False, returns a CommandEvent that can be used to check the command status later.
        """
        with self._lock:
            command_id = self._command_counter
            self._command_counter += 1
        
        psrl_logger.info("Executing command: %s with ID: %d", command.type, command_id)

        command_event = CommandEvent(command_id)
        with self._lock:
            self._command_events[command_id] = command_event
            self._command_results[command_id] = None
        
        command._kwargs["id"] = command_id
        psrl_logger.info("Command %s with ID %d added to command queue.", command.type, command_id)
        self.command_queue.put(command)
        
        # If not blocking, return the command event immediately
        if not blocking:
            return command_event

        # Wait for the command to complete
        if timeout is not None:
            success = command_event.wait(timeout=timeout)
            if not success:
                with self._lock:
                    if command_id in self._command_results:
                        del self._command_results[command_id]
                    if command_id in self._command_events:
                        del self._command_events[command_id]
                return None
        else:
            command_event.wait()

        with self._lock:
            result = self._command_results.get(command_id, None)
            psrl_logger.info(f"Command {command_id} completed with result: {result}")
            if command_id in self._command_results:
                del self._command_results[command_id]
            if command_id in self._command_events:
                del self._command_events[command_id]
        
        return result

    def wait_command_event(self, command_event: CommandEvent, timeout=None):
        """
        Wait for a specific command event to be set and return its result.
        
        Args:
            command_event (CommandEvent): The command event to wait for.
            timeout (float, optional): The maximum time to wait for the command event to be set.
        
        Returns:
            Any: The result of the command if it was completed, otherwise None.
        """
        if command_event is None:
            return None
        
        command_id = command_event.command_id
        
        if timeout is not None:
            success = command_event.wait(timeout=timeout)
            if not success:
                psrl_logger.warning(f"Command event {command_id} timed out after {timeout} seconds.")
                return None
        else:
            command_event.wait()
        
        with self._lock:
            result = self._command_results.get(command_id, None)
            if command_id in self._command_results:
                del self._command_results[command_id]
            if command_id in self._command_events:
                del self._command_events[command_id]

        return result
    
    def _complete_command(self, command_id: int, result: Any):
        """Set the command result, mark it as completed and notify the event waiter."""
        with self._lock:
            if command_id in self._command_results:
                self._command_results[command_id] = result
                psrl_logger.debug(f"Command ID {command_id} completed with result: {result}")
                # Set event to notify that the command has completed
                if command_id in self._command_events:
                    self._command_events[command_id].set()
            else:
                raise ValueError(f"Command ID {command_id} not found in results.")
