#include "MultiLayerPerceptron.cuh"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <vector>

__global__ void bias_add_kernel(double* activations, const double* biases, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int c = idx % cols;
        activations[idx] += biases[c];
    }
}

__global__ void sgd_momentun_kernel(double* W, double* V, const double* gradient, int n, double learning_rate, double momentum, double weight_decay) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double grad_reg = gradient[idx] + weight_decay * W[idx];
        V[idx] = momentum * V[idx] + grad_reg;
        W[idx] -= learning_rate * V[idx];
    }
}

__global__ void sgd_momentun_bias_kernel(double* B, double* V, const double* gradient, int n, double learning_rate, double momentum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        V[idx] = momentum * V[idx] + gradient[idx];
        B[idx] -= learning_rate * V[idx];
    }
}

__global__ void cross_entropy_kernel(const double* y_prediction, const double* y_true, double* loss_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double p = y_prediction[idx];
        if (p < 1e-12) p = 1e-12;
        if (p > 1.0 - 1e-12) p = 1.0 - 1e-12;
        loss_out[idx] = -y_true[idx] * log(p);
    }
}

__global__ void mse_element_kernel(const double *y_prediction, const double *y_true, double *loss_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double error = y_prediction[idx] - y_true[idx];
        loss_out[idx] = error * error;
    }
}

__global__ void mlp_reduce_sum_kernel(const double* data, double* partial, int n) {
    extern __shared__ double s_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    s_data[tid] = (idx < n) ? data[idx];
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();    
    }

    if (tid == 0) {
        partial[blockIdx.x] = s_data[0];
    }
}

/* Helper function to calculate number of blocks */
static inline int grid1d(int n, int block = 256) {
    return (n + block - 1) / block;
}

/* Sum of vector on GPU */
static double sum_gpu(const double *d_data, int n) {
    const int block = 256;
    const int g = grid1d(n, block);
    double *d_oartial;

    CUDA_CHECK(cudaMalloc(&d_partial, g * sizeof(double)));
    mlp_reduce_sum_kernel<<<g, block, block * sizeof(double)>>> (d_data, d_partial, n);
    CUDA_CHECK(cudaGetLastError());

    vector<double> h_partial(g);
    CUDA_CHECK(cudaMemCpy(h_partial.data(), d_partial, g * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partial));

    double total = 0;
    for (double v : h_partial) {
        total += v;
    }

    return total;
}