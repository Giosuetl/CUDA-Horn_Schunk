#pragma once
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define WIDTH 640
#define HEIGHT 480
#define ALPHA 1.0f
#define ITERATIONS 100

// 🔹 CUDA Kernel para calcular los gradientes Ix, Iy, It
__global__ void computeGradients(float* I1, float* I2, float* Ix, float* Iy, float* It, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width - 1 || y >= height - 1) return;

    int idx = y * width + x;

    Ix[idx] = (I1[idx + 1] - I1[idx] + I2[idx + 1] - I2[idx]) * 0.5f;
    Iy[idx] = (I1[idx + width] - I1[idx] + I2[idx + width] - I2[idx]) * 0.5f;
    It[idx] = I2[idx] - I1[idx];
}

// 🔹 CUDA Kernel para actualizar el flujo óptico U y V
__global__ void updateFlow(float* Ix, float* Iy, float* It, float* U, float* V, int width, int height, float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    int idx = y * width + x;

    float U_avg = (U[idx - 1] + U[idx + 1] + U[idx - width] + U[idx + width]) / 4.0f;
    float V_avg = (V[idx - 1] + V[idx + 1] + V[idx - width] + V[idx + width]) / 4.0f;

    float Ix2 = Ix[idx] * Ix[idx];
    float Iy2 = Iy[idx] * Iy[idx];

    float num = (Ix[idx] * U_avg + Iy[idx] * V_avg + It[idx]);
    float den = alpha * alpha + Ix2 + Iy2;

    if (den > 1e-5f) {
        U[idx] = U_avg - Ix[idx] * num / den;
        V[idx] = V_avg - Iy[idx] * num / den;
    }
}

// 🔹 Convertir imagen OpenCV a float y subirla a CUDA
void uploadImageToCuda(const cv::Mat& img, float* d_img) {
    int size = img.cols * img.rows * sizeof(float);
    float* h_img = new float[img.cols * img.rows];

    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            h_img[i * img.cols + j] = img.at<uchar>(i, j) / 255.0f;

    cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice);
    delete[] h_img;
}

int hon() {
    cv::VideoCapture cap(1);
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
    float* d_I1, * d_I2, * d_Ix, * d_Iy, * d_It, * d_U, * d_V;

    cudaMalloc(&d_I1, size);
    cudaMalloc(&d_I2, size);
    cudaMalloc(&d_Ix, size);
    cudaMalloc(&d_Iy, size);
    cudaMalloc(&d_It, size);
    cudaMalloc(&d_U, size);
    cudaMalloc(&d_V, size);

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    while (true) {
        cap >> frame;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        uploadImageToCuda(prevGray, d_I1);
        uploadImageToCuda(gray, d_I2);

        computeGradients << <gridSize, blockSize >> > (d_I1, d_I2, d_Ix, d_Iy, d_It, WIDTH, HEIGHT);

        cudaMemset(d_U, 0, size);
        cudaMemset(d_V, 0, size);

        for (int i = 0; i < ITERATIONS; i++) {
            updateFlow << <gridSize, blockSize >> > (d_Ix, d_Iy, d_It, d_U, d_V, WIDTH, HEIGHT, ALPHA);
        }

        float* h_U = new float[WIDTH * HEIGHT];
        float* h_V = new float[WIDTH * HEIGHT];

        cudaMemcpy(h_U, d_U, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_V, d_V, size, cudaMemcpyDeviceToHost);

        // 🔹 Calcular movimiento promedio en X, Y y velocidad
        float sumX = 0.0f, sumY = 0.0f, velocity = 0.0f;
        int count = 0;

        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; j++) {
                int idx = i * WIDTH + j;
                sumX += h_U[idx];
                sumY += h_V[idx];
                velocity += sqrt(h_U[idx] * h_U[idx] + h_V[idx] * h_V[idx]);
                count++;
            }
        }

        float avgX = sumX / count;
        float avgY = sumY / count;
        float avgVelocity = velocity / count;

        std::cout << "Movimiento X: " << avgX << " | Movimiento Y: " << avgY
            << " | Velocidad: " << avgVelocity << std::endl;

        prevGray = gray.clone();
        delete[] h_U;
        delete[] h_V;

        if (cv::waitKey(30) >= 0) break;
    }

    cudaFree(d_I1); cudaFree(d_I2); cudaFree(d_Ix); cudaFree(d_Iy);
    cudaFree(d_It); cudaFree(d_U); cudaFree(d_V);

    return 0;
}
