#pragma once
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>

#define WIDTH 640
#define HEIGHT 480
#define ALPHA 1.0f
#define ITERATIONS 50
#define BLOCK_SIZE 16

// 🔹 CUDA Kernel optimizado para calcular los gradientes con memoria compartida
__global__ void computeGradientsShared(const float* I1, const float* I2, float* Ix, float* Iy, float* It, int width, int height) {
    __shared__ float shared_I1[BLOCK_SIZE + 1][BLOCK_SIZE + 1];
    __shared__ float shared_I2[BLOCK_SIZE + 1][BLOCK_SIZE + 1];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        shared_I1[threadIdx.y][threadIdx.x] = I1[idx];
        shared_I2[threadIdx.y][threadIdx.x] = I2[idx];

        __syncthreads();

        if (x < width - 1 && y < height - 1) {
            Ix[idx] = (shared_I1[threadIdx.y][threadIdx.x + 1] - shared_I1[threadIdx.y][threadIdx.x] +
                shared_I2[threadIdx.y][threadIdx.x + 1] - shared_I2[threadIdx.y][threadIdx.x]) * 0.5f;
            Iy[idx] = (shared_I1[threadIdx.y + 1][threadIdx.x] - shared_I1[threadIdx.y][threadIdx.x] +
                shared_I2[threadIdx.y + 1][threadIdx.x] - shared_I2[threadIdx.y][threadIdx.x]) * 0.5f;
            It[idx] = shared_I2[threadIdx.y][threadIdx.x] - shared_I1[threadIdx.y][threadIdx.x];
        }
    }
}

// 🔹 CUDA Kernel optimizado para actualizar flujo óptico
__global__ void updateFlowOptimized(const float* Ix, const float* Iy, const float* It, float* U, float* V, int width, int height, float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;
    int idx = y * width + x;

    float U_avg = (U[idx - 1] + U[idx + 1] + U[idx - width] + U[idx + width]) * 0.25f;
    float V_avg = (V[idx - 1] + V[idx + 1] + V[idx - width] + V[idx + width]) * 0.25f;

    float Ix2 = Ix[idx] * Ix[idx];
    float Iy2 = Iy[idx] * Iy[idx];

    float num = (Ix[idx] * U_avg + Iy[idx] * V_avg + It[idx]);
    float den = alpha * alpha + Ix2 + Iy2;

    if (den > 1e-5f) {
        U[idx] = U_avg - Ix[idx] * num / den;
        V[idx] = V_avg - Iy[idx] * num / den;
    }
}

// 🔹 CUDA Kernel para calcular movimiento promedio y velocidad en GPU
__global__ void computeMotionVelocity(const float* U, const float* V, float* sumX, float* sumY, float* velocity, int width, int height) {
    __shared__ float shared_sumX[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float shared_sumY[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float shared_velocity[BLOCK_SIZE * BLOCK_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (x < width && y < height) {
        shared_sumX[tid] = U[idx];
        shared_sumY[tid] = V[idx];
        shared_velocity[tid] = sqrt(U[idx] * U[idx] + V[idx] * V[idx]);
    }
    else {
        shared_sumX[tid] = 0;
        shared_sumY[tid] = 0;
        shared_velocity[tid] = 0;
    }

    __syncthreads();

    if (tid == 0) {
        float blockSumX = 0, blockSumY = 0, blockVelocity = 0;
        for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++) {
            blockSumX += shared_sumX[i];
            blockSumY += shared_sumY[i];
            blockVelocity += shared_velocity[i];
        }
        atomicAdd(sumX, blockSumX);
        atomicAdd(sumY, blockSumY);
        atomicAdd(velocity, blockVelocity);
    }
}

int hon() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: No se pudo abrir la cámara.\n";
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT);

    cv::Mat frame, gray, prevGray;
    cap >> frame;
    cv::cvtColor(frame, prevGray, cv::COLOR_BGR2GRAY);

    int size = WIDTH * HEIGHT * sizeof(float);
    float* d_I1, * d_I2, * d_Ix, * d_Iy, * d_It, * d_U, * d_V, * d_sumX, * d_sumY, * d_velocity;

    cudaMalloc(&d_I1, size);
    cudaMalloc(&d_I2, size);
    cudaMalloc(&d_Ix, size);
    cudaMalloc(&d_Iy, size);
    cudaMalloc(&d_It, size);
    cudaMalloc(&d_U, size);
    cudaMalloc(&d_V, size);
    cudaMalloc(&d_sumX, sizeof(float));
    cudaMalloc(&d_sumY, sizeof(float));
    cudaMalloc(&d_velocity, sizeof(float));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    while (true) {
        cap >> frame;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cudaMemcpy(d_I1, prevGray.data, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_I2, gray.data, size, cudaMemcpyHostToDevice);

        computeGradientsShared << <gridSize, blockSize >> > (d_I1, d_I2, d_Ix, d_Iy, d_It, WIDTH, HEIGHT);

        cudaMemset(d_U, 0, size);
        cudaMemset(d_V, 0, size);

        for (int i = 0; i < ITERATIONS; i++) {
            updateFlowOptimized << <gridSize, blockSize >> > (d_Ix, d_Iy, d_It, d_U, d_V, WIDTH, HEIGHT, ALPHA);
        }

        cudaMemset(d_sumX, 0, sizeof(float));
        cudaMemset(d_sumY, 0, sizeof(float));
        cudaMemset(d_velocity, 0, sizeof(float));

        computeMotionVelocity << <gridSize, blockSize >> > (d_U, d_V, d_sumX, d_sumY, d_velocity, WIDTH, HEIGHT);

        float h_sumX, h_sumY, h_velocity;
        cudaMemcpy(&h_sumX, d_sumX, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sumY, d_sumY, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_velocity, d_velocity, sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Movimiento X: " << h_sumX / (WIDTH * HEIGHT)
            << " | Movimiento Y: " << h_sumY / (WIDTH * HEIGHT)
            << " | Velocidad: " << h_velocity / (WIDTH * HEIGHT) << std::endl;

        prevGray = gray.clone();
        if (cv::waitKey(30) >= 0) break;
    }

    cudaFree(d_I1); cudaFree(d_I2); cudaFree(d_Ix); cudaFree(d_Iy);
    cudaFree(d_It); cudaFree(d_U); cudaFree(d_V);
    return 0;
}
