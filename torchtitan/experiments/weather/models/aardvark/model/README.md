## Aardvark Weather model (refactored)

This directory contains a cleaned, modular re-implementation of the Aardvark end‑to‑end model.

### Source and attribution
- Original research implementation: [anna-allen/aardvark-weather-public](https://github.com/anna-allen/aardvark-weather-public)
- Based on commit: `8fb35a0f5f7602eaf5127b9642c54eb052d68f56`

### What’s here
- `model.py`: `AardvarkE2E` end‑to‑end model. Non‑mutating API using tensorclass tasks, explicit normalization, configurable hyperparameters, and pluggable submodules.
- `vit.py`: Vision Transformer backbone used by encoder/processor variants.
- `unet_wrap_padding.py`: Cylindrical‑padding UNet used by the station decoder.
- `layers.py`: `ConvDeepSet` implementation for gridded↔off‑grid mapping.
- `architectures.py`: Small MLPs and residual blocks (e.g., `DownscalingMLP`).
- `aardvark_utils.py`: Tensor helpers (channel ordering, normalization shape handling).
- `tests.py`: Lightweight unit tests for math equivalence and non‑mutation semantics.

### Key differences from research code
- Non‑mutating forward pass with typed tensorclass inputs instead of in‑place dict mutation.
- Normalization factors are constructor inputs (registered as buffers) rather than loaded from disk.
- Hyperparameters (e.g., overwrite channels, grid sizes) exposed via constructor args.
- Device placement follows input tensors to minimize `.to(device)` churn.
- Dead/unused paths and ad‑hoc file I/O removed; code organized into local modules above.
- Return semantics preserved: with `return_gridded=True`, returns `(stations, forecast_grid, initial_state_grid)`.

### Tests
Run from repo root:

```bash
cd ~/default/weather-clip/torchtitan
source .venv/bin/activate
python torchtitan/experiments/weather/models/aardvark/model/tests.py
```

### State dict compatibility
We aim for compatibility with existing submodule state dicts by mirroring submodule names where possible. Loading full end‑to‑end weights may still require a key‑mapping step depending on how the original checkpoints were saved.


