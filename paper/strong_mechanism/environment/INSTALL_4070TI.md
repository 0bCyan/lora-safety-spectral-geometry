# 4070 Ti Formal Training Environment

Paper A formal Qwen3-4B runs are locked to the recorded 4070 Ti software stack. Driver and operating-system values are recorded but not required to equal the historical report exactly; Python, CUDA-enabled PyTorch, core training libraries, GPU class, and model revision are strict.

## Recommended environment creation

On the 4070 Ti machine:

```powershell
py -3.11 -m venv .venv-paper-a
.\.venv-paper-a\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install `
  transformers==4.53.3 `
  peft==0.16.0 `
  datasets==2.21.0 `
  accelerate==1.8.1 `
  bitsandbytes==0.49.2 `
  numpy==1.26.4 `
  pandas==2.2.3 `
  safetensors sentencepiece protobuf scipy huggingface-hub
```

Verify before any formal model load:

```powershell
python paper/strong_mechanism/scripts/check_training_environment.py
```

The pinned model snapshot is resolved by the formal launcher. To prefetch it without starting training:

```powershell
python paper/strong_mechanism/scripts/prepare_runtime_config.py `
  --config paper/strong_mechanism/configs/qwen3_4b_equal_dose_temporal_schedules.json `
  --output paper/strong_mechanism/data/runs/runtime_configs/equal_dose.json `
  --verify-environment
```

For an already cached snapshot, add `--offline` to prohibit network access.

## Environment-change rule

Any strict package mismatch stops before model loading. Updating a strict version requires a versioned environment lock and a symmetric rerun of all affected principal arms. Do not mix strict environments inside a seed-pooled comparison.
