"""vLLM sleep-mode backends contributed by the PSRL plugin.

Registered lazily with vLLM's ``SleepModeBackendFactory`` from
``vllm_patches.register_patches``; keep this module import-light so the plugin
can register a backend without importing its heavy dependencies (for example
``torch_memory_saver``).
"""
