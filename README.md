# Multilayer Perceptron — CUDA

GPU-accelerated Multilayer Perceptron implemented from scratch in **CUDA C++**, without any deep learning framework dependency. Trained and evaluated on the **HASYv2** handwritten mathematical symbol dataset (369 classes).

## Results

| Dataset | Classes | Test Accuracy | Early Stopping |
|---------|---------|--------------|----------------|
| HASYv2 (Fold 1) | 369 | **71.55 %** | Epoch 98 |

---

## Architecture

```
Input (1024) → Dense (512, ReLU) → Dense (256, ReLU) → Dense (128, ReLU) → Output (369, Softmax)
```

| Hyperparameter | Value |
|---|---|
| Learning Rate | 0.001 |
| Momentum (SGD) | 0.9 |
| Weight Decay (L2) | 1e-5 |
| Batch Size | 128 |
| Max Epochs | 100 |
| Early Stopping Patience | 80 |
| Seed | 42 |

---

## Project Structure

```
MLP CUDA/
├── include/
│   ├── Matrix.cuh                  # GPU matrix struct (device memory)
│   ├── Activation.cuh              # Activation function interfaces
│   ├── MultiLayerPerceptron.cuh    # MLP class declaration
│   └── DataLoader.cuh              # Dataset loading utilities
├── src/
│   ├── Matrix.cu                   # Matrix ops: dot, hadamard, transpose, etc.
│   ├── Activation.cu               # CUDA kernels: sigmoid, ReLU, softmax
│   ├── MultiLayerPerceptron.cu     # Forward, backward, train, predict kernels
│   └── DataLoader.cu               # CSV + OpenCV image loading, one-hot encoding
├── MLP_CUDA/
│   └── main.tex                    # LaTeX report
├── main.cu                         # Entry point
├── Makefile
└── README.md
```

---

## CUDA Kernels

| Kernel | Description |
|---|---|
| `bias_add_kernel` | Parallel bias addition across all activations |
| `kernel_relu_fwd / deriv` | Element-wise ReLU forward and derivative |
| `kernel_sigmoid_fwd / deriv` | Element-wise Sigmoid forward and derivative |
| `kernel_softmax` | Row-wise Softmax (one block per sample) |
| `mse_element_kernel` | Element-wise squared error |
| `mlp_reduce_sum_kernel` | Parallel reduction sum with shared memory |
| `sgd_momentun_kernel` | SGD + momentum + L2 weight update |
| `sgd_momentun_bias_kernel` | SGD + momentum bias update |

All data (weights, biases, activations) lives in **device memory** throughout training. No host↔device transfers occur during forward/backward passes.

---

## Dataset — HASYv2

[HASYv2](https://zenodo.org/record/259444) is a handwritten mathematical symbol dataset with:
- **369 classes** (LaTeX symbols)
- **32 × 32** grayscale images
- Pre-defined train/test folds (`classification-task/fold-N/`)

Place the extracted dataset at `HASYv2/` in the project root:

```
HASYv2/
├── symbols.csv
└── classification-task/
    └── fold-1/
        ├── train.csv
        └── test.csv
```

---

## Requirements

- NVIDIA GPU with CUDA support (compute capability ≥ 6.0 recommended)
- CUDA Toolkit ≥ 11.0
- OpenCV 4.x
- g++ / nvcc
- C++17

### Ubuntu / Debian

```bash
sudo apt install nvidia-cuda-toolkit libopencv-dev
```

---

## Build & Run

```bash
# Build
make

# Run with default symbols path
./mlp_cuda

# Run with custom symbols CSV
./mlp_cuda HASYv2/symbols.csv
```

---

## Training Output Example

```
=== Fold 1 ===
Loading training data...
Loading test data...
Epoch  1/100 - Train Loss: 0.0020 - Val Loss: 0.0018
Epoch 10/100 - Train Loss: 0.0010 - Val Loss: 0.0012
Epoch 20/100 - Train Loss: 0.0009 - Val Loss: 0.0011
...
Early stopping at epoch 98 (best val loss: 0.0011)
Fold 1 Test Accuracy: 71.55%

Average Test Accuracy over 1 folds: 71.55%
```

---

## License

This project is licensed under the [MIT License](LICENSE).
