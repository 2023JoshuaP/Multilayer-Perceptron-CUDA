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
    s_data[tid] = (idx < n) ? data[idx] : 0.0;
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
    double *d_partial;

    CUDA_CHECK(cudaMalloc(&d_partial, g * sizeof(double)));
    mlp_reduce_sum_kernel<<<g, block, block * sizeof(double)>>> (d_data, d_partial, n);
    CUDA_CHECK(cudaGetLastError());

    vector<double> h_partial(g);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, g * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partial));

    double total = 0;
    for (double v : h_partial) {
        total += v;
    }

    return total;
}

MultiLayerPerceptron::MultiLayerPerceptron(const vector<int> &layer_sizes, shared_ptr<ActivationFunction> activation, double learning_rate, double momentum, double weight_decay, int seed) : layer_sizes_(layer_sizes), activation_(activation), learning_rate_(learning_rate), momentum_(momentum), weight_decay_(weight_decay), num_layers_(static_cast<int>(layer_sizes.size())), rng_(seed) {
    mt19937 init_rng(seed);
    for (int i = 0; i < num_layers_ - 1; i++) {
        int fan_in = layer_sizes_[i];
        int fan_out = layer_sizes_[i + 1];
        double scale = sqrt(2.0 / fan_in);
        weights_.push_back(gpu_random(fan_in, fan_out, scale, init_rng));
        biases_.push_back(gpu_zeros(1, fan_out));
        vel_weights_.push_back(gpu_zeros(fan_in, fan_out));
        vel_biases_.push_back(gpu_zeros(1, fan_out));
    }
}

vector<Matrix> MultiLayerPerceptron::forward(const Matrix &input) const {
    vector<Matrix> activations;
    activations.reserve(num_layers_);
    activations.push_back(input);

    int n_weight_layers = num_layers_ - 1;
    Matrix A = input;
    for (int i = 0; i < n_weight_layers; i++) {
        Matrix Z = A.dot(weights_[i]);
        int total = Z.rows * Z.cols;

        bias_add_kernel<<<grid1d(total), 256>>> (Z.d_data, biases_[i].d_data, Z.rows, Z.cols);
        CUDA_CHECK(cudaGetLastError());

        if (i == n_weight_layers - 1) {
            A = Softmax::forward(Z);
        }
        else {
            A = activation_->forward(Z);
        }

        activations.push_back(A);
    }

    return activations;
}