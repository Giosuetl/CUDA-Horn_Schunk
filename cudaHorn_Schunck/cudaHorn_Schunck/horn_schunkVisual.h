#pragma once

#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define WIDTH 640
#define HEIGHT 480
#define ALPHA 1.0f   // Regularización
#define ITERATIONS 100  // Iteraciones de Horn-Schunck
#define BLOCK_SIZE 16

// 🔹 CUDA Kernel para calcular los gradientes Ix, Iy, It
__global__ void computeGradients(const float* I1, const float* I2, float* Ix, float* Iy, float* It, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width - 1 || y >= height - 1) return;

    int idx = y * width + x;
    Ix[idx] = (I1[idx + 1] - I1[idx] + I2[idx + 1] - I2[idx]) * 0.5f;
    Iy[idx] = (I1[idx + width] - I1[idx] + I2[idx + width] - I2[idx]) * 0.5f;
    It[idx] = I2[idx] - I1[idx];
}

// 🔹 CUDA Kernel para actualizar el flujo óptico U y V con Horn-Schunck
__global__ void updateFlow(const float* Ix, const float* Iy, const float* It, float* U, float* V, int width, int height, float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    int idx = y * width + x;
    float U_avg = (U[idx - 1] + U[idx + 1] + U[idx - width] + U[idx + width]) * 0.25f;
    float V_avg = (V[idx - 1] + V[idx + 1] + V[idx - width] + V[idx + width]) * 0.25f;

    float Ix2 = Ix[idx] * Ix[idx];
    float Iy2 = Iy[idx] * Iy[idx];
    float num = Ix[idx] * U_avg + Iy[idx] * V_avg + It[idx];
    float den = alpha * alpha + Ix2 + Iy2;

    if (den > 1e-5f) {
        U[idx] = U_avg - Ix[idx] * num / den;
        V[idx] = V_avg - Iy[idx] * num / den;
    }
}

void uploadImageToCuda(const cv::Mat& img, float* d_img) {
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32F, 1.0 / 255.0);
    cudaMemcpy(d_img, imgFloat.ptr<float>(), imgFloat.total() * sizeof(float), cudaMemcpyHostToDevice);
}

int hsVisual() {
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

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

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

        std::vector<float> h_U(WIDTH * HEIGHT), h_V(WIDTH * HEIGHT);
        cudaMemcpy(h_U.data(), d_U, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_V.data(), d_V, size, cudaMemcpyDeviceToHost);

        cv::Mat flowU(HEIGHT, WIDTH, CV_32F, h_U.data());
        cv::Mat flowV(HEIGHT, WIDTH, CV_32F, h_V.data());

        cv::Mat hsv[3], hsvImage, colorFlow;
        cv::cartToPolar(flowU, flowV, hsv[1], hsv[0]);
        hsv[0] *= 180 / CV_PI;
        hsv[2] = cv::Mat::ones(HEIGHT, WIDTH, CV_32F);
        cv::merge(hsv, 3, hsvImage);
        hsvImage.convertTo(hsvImage, CV_8U, 255);
        cv::cvtColor(hsvImage, colorFlow, cv::COLOR_HSV2BGR);

        cv::imshow("Flujo Óptico Horn–Schunck", colorFlow);

        if (cv::waitKey(30) >= 0) break;
        prevGray = gray.clone();
    }

    cudaFree(d_I1); cudaFree(d_I2); cudaFree(d_Ix); cudaFree(d_Iy);
    cudaFree(d_It); cudaFree(d_U); cudaFree(d_V);
    return 0;
}
