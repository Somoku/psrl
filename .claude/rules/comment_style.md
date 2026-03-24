# Comment Style Guide

This guide documents the comment conventions used in PSRL.
All new code must follow these conventions.

## 1. Annotation Markers

Use `# MARKER(author): explanation` for special-purpose comments.
No period at the end (these are labels, not sentences).

| Marker | Purpose | Example |
|--------|---------|---------|
| `NOTE(xx):` | Design decisions, non-obvious implementation choices | `# NOTE(claude): ray_worker_group_cls is used only in train side` |
| `TODO(xx):` | Future improvements or known missing features | `# TODO(claude): add timeout handling for fault tolerance` |
| `FIXME(xx):` | Known bugs or incorrect behavior that must be fixed | `# FIXME(claude): race condition when two workers push simultaneously` |
| `HACK(xx):` | Temporary workarounds; must be accompanied by why | `# HACK(claude): bypass Ray serialization bug in v2.9` |

Rules:
- Always include author initials in parentheses.
- Colon immediately after the closing paren, then one space, then the message.
- Multi-line NOTE/TODO: repeat the `#` prefix on each line, no hanging indent.

```python
# NOTE(claude): Now we use a dict to store the PS handle and merge them on the PS side.
# This is more efficient than calling transfer_train_to_gen for each key/shard,
# which would cause excessive remote calls and may crash the Ray actor.
```

## 2. Docstrings

Use **Google-style** docstrings for all public functions, methods, and classes.

```python
def wait_for_nixl_push_completion(self, timeout: float | None = None) -> bool:
    """
    Wait for the NIXL push wait thread to complete.

    Args:
        timeout (float | None): Maximum wait time in seconds.
            If None, wait indefinitely.

    Returns:
        bool: True if completed successfully, False if timed out.

    Raises:
        RuntimeError: If the push thread encountered an unrecoverable error.
    """
```

Rules:
- Opening `"""` on the same line as the function signature is **not** used; always on its own line.
- One-line summary first, then blank line, then sections.
- Section headers (`Args:`, `Returns:`, `Raises:`) followed by a colon, indented body with 4 spaces.
- Type in parentheses after arg name, using `|` union syntax: `timeout (float | None):`. Never use `optional`.
- Continuation lines for long arg descriptions indented 8 spaces (4 extra).
- End the summary sentence with a **period**; arg/return descriptions may omit it if they are fragments.
- Add a `Usage:` section with a code block when the function has non-obvious call sequencing.

## 3. Inline Comments

```python
self.ray_worker_group_cls = ray_worker_group_cls  # NOTE(claude): used only on train side
```

Rules:
- Exactly **2 spaces** between end of code and `#`.
- One space after `#`.
- Short fragment preferred; complete sentences allowed when clarity demands it.
- No period for fragments; period for complete sentences.
- Do **not** vertically align inline comments across multiple lines.

## 4. Block Comments (Standalone)

```python
# Set the validation rollout number in the config based on the global batch size.
config.actor_rollout_ref.rollout.val_rollout_n = ...
```

Rules:
- Always placed **above** the code they describe, never after.
- Indented to match the code below.
- Each line starts with `# ` (hash + one space).
- Complete sentence → period. Fragment → no period.
- Blank line above a block comment when it introduces a new logical section.

## 5. Section Separators

Use sparingly, only in long procedural files (>150 lines) to mark major phases:

```python
# --- Initialization ---
...

# --- Main Training Loop ---
...
```

Or for emphasis:
```python
# ----------------------------------------
# Phase 2: Staleness-Controlled Update
# ----------------------------------------
```

## 6. Assertions

Every `assert` **must** include a descriptive message.

```python
# Single-line
assert self.nixl_storage_client is not None, "nixl_storage_client is not initialized."

# Multi-line (condition too long, or message too long)
assert self.psrl_config.ps_mode in ("nixl_cpu", "nixl_gpu"), (
    "push_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode, "
    f"got: {self.psrl_config.ps_mode!r}."
)
```

Rules:
- Message ends with a **period**.
- Include the actual value in the message when diagnosing "wrong value" failures (`f"..., got: {val!r}."`).
- Multi-line: wrap the message string in `(...)`, continuation lines indented 4 spaces relative to `assert`.

## 7. Logging

```python
psrl_logger = logging.getLogger(__file__)

psrl_logger.debug("Getting the current PS model version...")
psrl_logger.info("[validate_config] All configuration checks passed successfully!")
psrl_logger.info(
    f"Pushing key {key} shards {shards_to_transfer} to {target_client_name} "
    f"for version {next_ps_model_version} with {len(shards_to_transfer)} shards."
)
psrl_logger.warning(f"[{log_prefix}]: failed to log {name}: {e}")
```

Rules:
- Use `psrl_logger`, never `print()`.
- Prefix with `[component_name]` for messages emitted from key entry points.
- End messages with a **period**.
- Multi-line: use implicit string concatenation inside `logger.xxx(...)`, not `\`.
- Level guidelines: `debug` for internal state, `info` for lifecycle events, `warning` for recoverable issues, `error` for failures.

## 8. Language

- **English only.** No Chinese in comments, docstrings, or log messages.
