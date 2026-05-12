# TinyInferEngine

A small C++ deep-learning style framework with a **dynamic computation graph** and **autograd** (backward closures on tensors). It includes an MNIST training demo and a separate inference binary that loads float32 weights from raw `.bin` files.

**Language**: [中文](README.md) | English (this file)

## Highlights

| Area | Notes |
|------|--------|
| Core | `Tensor`, `Layer`, `Sequential`, autograd via `set_auto_grad` |
| Layers | `Linear`, `Conv2D`, `MaxPool2D`, `Flatten`, `ReLU`, `SiLU`, `BatchNorm2D` |
| Training | `train_minist.cpp` (CMake target name keeps the historical typo `train_minist`) |
| Inference | `infer_engine` from `src/main.cpp` |
| Export | Root-level `export_model.py` trains a matching PyTorch CNN and exports weights for C++ |

## Dependencies

- **C++17** (see `CMakeLists.txt`).
- **OpenMP** required at **configure** time (`find_package(OpenMP REQUIRED)`).
- **Python / PyTorch** only if you use `export_model.py`; the C++ library itself does not link to PyTorch.

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

On Windows with the Visual Studio generator, binaries usually appear under `build/Release/` or `build/Debug/`.

## Repository hygiene

- Do not commit `build/`, `out/`, or IDE caches. Use the provided `.gitignore`.
- Large `weights/*.bin` files and full datasets are best kept local or distributed via LFS/releases.

## Tests

`tests/test_tensor.cpp` is an older draft and is **not** wired into CMake until it is updated to the current `Tensor` / `Layer` API.

## Last updated

2026-05-12
