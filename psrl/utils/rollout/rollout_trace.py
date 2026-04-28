# Adapted from verl/verl/utils/rollout_trace.py
import contextlib
import copy
import functools
import inspect
import os
from functools import partial
from typing import Optional

from verl import DataProto


class RolloutTraceConfig:
    """Configuration for rollout tracing with various backends.

    Singleton configuration class for managing rollout trace settings across different
    tracing backends like Weave and MLflow.

    Args:
        backend (Optional[str]): Tracing backend to use ('weave', 'mlflow', or None).
        client (Optional[object]): Client instance for the selected backend.
        token2text (bool): Whether to convert tokens to text in traces. Defaults to False.
        project_name (str): Name of the project for tracing.
        experiment_name (str): Name of the experiment for tracing.
        max_samples_per_step_per_worker (Optional[int]): Maximum number of unique samples to trace
            per worker per step. If None, all samples are traced. If set, each worker will randomly
            select up to this many unique samples to trace (including all their rollouts for GRPO).
            Total traces = max_samples_per_step_per_worker * num_workers * n_rollouts_per_sample.
    """

    _instance: Optional["RolloutTraceConfig"] = None
    backend: str | None = None
    client: object | None = None
    token2text: bool = False
    _initialized: bool = False
    project_name: str = None
    experiment_name: str = None
    max_samples_per_step_per_worker: int | None = None

    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern: only one instance exists.

        Returns:
            RolloutTraceConfig: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "RolloutTraceConfig":
        """Get the singleton instance of RolloutTraceConfig.

        Returns:
            RolloutTraceConfig: The singleton configuration instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def init(
        cls,
        project_name: str,
        experiment_name: str,
        backend: str,
        token2text: bool = False,
        max_samples_per_step_per_worker: int | None = None,
    ):
        """Initialize the tracing configuration with the specified backend.

        Sets up the tracing backend (Weave or MLflow) and initializes the client.
        This method is idempotent - calling it multiple times has no effect after
        the first initialization.

        Args:
            project_name: Name of the project for organizing traces
            experiment_name: Name of the experiment within the project
            backend: Tracing backend to use ('weave', 'mlflow', or None to disable)
            token2text: Whether to convert token IDs to text in traces (default: False)
        """
        config = cls.get_instance()
        if config._initialized:
            return

        config.backend = backend
        config.token2text = token2text
        config.project_name = project_name
        config.experiment_name = experiment_name
        config.max_samples_per_step_per_worker = max_samples_per_step_per_worker

        # Initialize backend-specific client
        if backend == "weave":
            import weave

            config.client = weave.init(project_name)
        elif backend == "mlflow":
            import mlflow

            mlflow.config.enable_async_logging()
            config.client = mlflow

            # Configure MLflow tracking URI from environment or use default
            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            mlflow.set_experiment(project_name)
        else:
            # No tracing backend configured
            config.client = None

        config._initialized = True

    @classmethod
    def get_backend(cls) -> str | None:
        """Get the configured tracing backend name.

        Returns:
            str | None: Backend name ('weave', 'mlflow', or None).
        """
        return cls.get_instance().backend

    @classmethod
    def get_client(cls) -> object | None:
        """Get the tracing client instance.

        Returns:
            object | None: The client instance for the configured backend, or None.
        """
        return cls.get_instance().client

    @classmethod
    def enable_token2text(cls) -> bool | None:
        """Check if token-to-text conversion is enabled.

        Returns:
            bool | None: True if token-to-text conversion is enabled, False otherwise.
        """
        return cls.get_instance().token2text

    @classmethod
    def reset(cls):
        """Reset the singleton instance, clearing all configuration.

        Useful for testing or re-initialization scenarios.
        """
        cls._instance = None


@contextlib.contextmanager
def rollout_trace_attr(prompt_index=None, request_index=None, step=None, name="rollout_trace", validate=False):
    """Context manager for adding attributes to a trace span.

    This context manager adds metadata attributes to the current trace span,
    allowing for better organization and filtering of traces. The attributes
    are added to the configured tracing backend (Weave or MLflow).

    Args:
        prompt_index: Index of the prompt in the batch (optional)
        request_index: Index of the request being processed (optional)
        step: Current step number in the rollout (optional)
        name: Name of the trace span (default: "rollout_trace")
        validate: Whether this is a validation run (default: False)

    Yields:
        None: Context with trace attributes set

    Example:
        with rollout_trace_attr(prompt_index=0, step=3):
            # Code executed within this context will have these attributes
            result = process_step()
    """
    backend = RolloutTraceConfig.get_backend()
    attributes = {}

    # Collect trace attributes if backend is configured
    if backend:
        if prompt_index is not None:
            attributes["prompt_index"] = prompt_index
        if request_index is not None:
            attributes["request_index"] = request_index
        if step is not None:
            attributes["step"] = step
        attributes["validate"] = validate
        attributes["experiment_name"] = RolloutTraceConfig.get_instance().experiment_name

    # If no backend or no attributes, just yield without tracing
    if not attributes or backend is None:
        yield
        return

    # Add attributes to the appropriate backend
    if backend == "weave":
        import weave

        with weave.attributes(attributes):
            yield
    elif backend == "mlflow":
        import mlflow

        with mlflow.start_span(name=name) as span:
            trace_id = span.trace_id
            for key, value in attributes.items():
                mlflow.set_trace_tag(trace_id, str(key), str(value))
            yield
    else:
        yield


def rollout_trace_op(func):
    """Decorator for tracing function/method calls during rollout.

    This decorator automatically traces function calls with their inputs and outputs,
    integrating with the configured tracing backend (Weave or MLflow). It handles
    both synchronous and asynchronous functions.

    For async functions, the decorator captures:
    - Function inputs (arguments and keyword arguments)
    - Function outputs (return values)
    - Exceptions (if any occur)
    - Optional token-to-text conversion for DataProto outputs

    Args:
        func: The function or method to be traced

    Returns:
        Wrapped function with tracing capabilities

    Example:
        @rollout_trace_op
        async def process_batch(self, batch: DataProto):
            # Function implementation
            return result
    """

    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        """Async wrapper for traced functions."""
        backend = RolloutTraceConfig.get_backend()
        enable_token2text = RolloutTraceConfig.enable_token2text()
        if backend is None:
            # No tracing configured, execute function directly
            return await func(self, *args, **kwargs)

        # Extract function arguments for tracing
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        def add_token2text(self, result):
            """Convert token IDs to text in DataProto results for better readability."""

            def _process_single_item(item):
                """Process a single item, converting tokens to text if applicable."""
                if isinstance(item, DataProto) and hasattr(self, "tokenizer") and hasattr(self.tokenizer, "decode"):
                    assert len(item) == 1, "Only single item DataProto is supported for token2text conversion."
                    request_proto = item[0]
                    processed_item = {}
                    for key, value in request_proto.non_tensor_batch.items():
                        processed_item[key] = copy.deepcopy(value)
                    for key, value in request_proto.meta_info.items():
                        processed_item[key] = copy.deepcopy(value)
                    if "raw_prompt_ids" in processed_item.keys():
                        prompt_text = self.tokenizer.decode(processed_item["raw_prompt_ids"])
                        processed_item["prompt_text"] = prompt_text

                    if "raw_response_ids" in processed_item.keys():
                        response_text = self.tokenizer.decode(processed_item["raw_response_ids"])
                        processed_item["response_text"] = response_text

                    return processed_item
                return item

            if isinstance(result, list):
                _result = []
                for i, item in enumerate(result):
                    _result[i] = _process_single_item(item)
                return _result
            elif isinstance(result, tuple):
                processed_items = []
                for item in result:
                    processed_items.append(_process_single_item(item))
                return tuple(processed_items)
            elif isinstance(result, dict):
                _result = {}
                for key, value in result.items():
                    _result[key] = _process_single_item(value)
                return _result
            else:
                return _process_single_item(result)

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()
            from weave.trace.context import call_context
            from weave.trace.weave_client import _build_anonymous_op

            cur_attributes = {**call_context.call_attributes.get()}

            op = func.__qualname__
            if op not in tracer._anonymous_ops:
                tracer._anonymous_ops[op] = _build_anonymous_op(op)
            op = tracer._anonymous_ops[op]
            op.postprocess_output = partial(add_token2text, self) if enable_token2text else None
            call = tracer.create_call(op=op, inputs=inputs, attributes=cur_attributes)
            try:
                result = await func(self, *args, **kwargs)

                tracer.finish_call(call, output=result, op=op)
                return result

            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            with mlflow.start_span(name=func.__qualname__) as span:
                span.set_inputs(inputs)
                result = await func(self, *args, **kwargs)
                if enable_token2text:
                    _result = add_token2text(self, result)
                    span.set_outputs(_result)
                else:
                    span.set_outputs(result)

            return result

        else:
            return await func(self, *args, **kwargs)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        backend = RolloutTraceConfig.get_backend()
        if backend is None:
            return func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()
            from weave.trace.context import call_context

            cur_attributes = {**call_context.call_attributes.get()}
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = func(self, *args, **kwargs)
                tracer.finish_call(call, output=result)
                return result
            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            return mlflow.trace(func)(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else wrapper
