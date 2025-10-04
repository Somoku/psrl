# VERL Version Upgrade Checklist

This document lists the key components and methods that need to be checked and verified when upgrading the VERL version.

## Train Workers

### PSRL_MegatronTrainWorker

Methods to check:

- `__init__`
- `compute_log_prob`

### PSRL_FSDPTrainWorker

Methods to check:

- `compute_log_prob`

## Generation Workers

### PSRL_GenWorker

Methods to check:

- `_build_rollout`

## Rollout Components

### PSRL_vLLMRollout

Methods to check:

- `__init__`

## Main Training Script

### main_ppo.py

- Verify main training pipeline
- Check command line argument parsing
- Confirm logging and monitoring functionality

## Ray PPO Trainer

### PSRL_RayPPOTrainer

Methods to check:

- `__init__`
- `_validate_config`
- `_save_checkpoint`
- `_load_checkpoint`
- `fit`

### utils

- `compute_advantage`

## Configuration Files

### ppo_trainer.yaml

- Verify configuration parameter compatibility
- Check new or modified configuration items
- Ensure reasonable default values

### ppo_megatron_trainer.yaml

- Check Megatron-specific configurations
- Verify parallelism strategy parameters
- Confirm model configuration correctness
