
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


//#include "Horn-Schunck.h"
#include "horn_schunkVisual.h"
#include <iostream>


/*
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
Multiprocesadores (SM): 16
Mßximo hilos por bloque: 1024
Mßximo hilos por multiprocesador: 1536
Mßximo bloques por multiprocesador: 16
Tama±o mßximo de grid (dim3): (2147483647, 65535, 65535)
*/

int main()
{
    cudaDeviceProp prop;
    int device;

    // Obtener el dispositivo actual
    cudaGetDevice(&device);
    // Obtener propiedades de la GPU
    cudaGetDeviceProperties(&prop, device);

    // Información relevante
    printf("GPU: %s\n", prop.name);
    printf("Multiprocesadores (SM): %d\n", prop.multiProcessorCount);
    printf("Máximo hilos por bloque: %d\n", prop.maxThreadsPerBlock);
    printf("Máximo hilos por multiprocesador: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Máximo bloques por multiprocesador: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Tamaño máximo de grid (dim3): (%d, %d, %d)\n\n\n\n",
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    //hon();
    hsVisual();
    return 0;
}

