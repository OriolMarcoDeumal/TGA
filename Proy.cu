#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <sys/times.h>
#include <sys/resource.h>

using namespace std;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void histogram_kernel(unsigned char *input_ptr, int *histogram, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height * 3) {
        int Y = (int) (16 + 0.25679890625 * input_ptr[idx + 0] + 0.50412890625 * input_ptr[idx + 1] + 0.09790625 * input_ptr[idx + 2]);
        atomicAdd(&histogram[Y], 1);
    }
}

__global__ void equalize_kernel(unsigned char *input_ptr, int *histogram_equalized, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height * 3) {
        int value_before = input_ptr[idx];
        int value_after = histogram_equalized[value_before];
        input_ptr[idx] = value_after;
    }
}

__global__ void ycbcr_kernel(unsigned char *input_ptr, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height * 3) {
        int r = input_ptr[idx + 0];
        int g = input_ptr[idx + 1];
        int b = input_ptr[idx + 2];

        int Y = (int) (16 + 0.25679890625 * r + 0.50412890625 * g + 0.09790625 * b);
        int Cb = (int) (128 - 0.168736 * r - 0.331264 * g + 0.5 * b);
        int Cr = (int) (128 + 0.5 * r - 0.418688 * g - 0.081312 * b);

        input_ptr[idx + 0] = Y;
        input_ptr[idx + 1] = Cb;
        input_ptr[idx + 2] = Cr;

        int R = max(0, min(255, (int) (Y + 1.402 * (Cr - 128))));
        int G = max(0, min(255, (int) (Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128))));
        int B = max(0, min(255, (int) (Y + 1.772 * (Cb - 128))));

        input_ptr[idx + 0] = R;
        input_ptr[idx + 1] = G;
        input_ptr[idx + 2] = B;
    }
}

void CheckCudaError(char sms[], int line) {
    cudaError_t error;
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s at line %d\n", sms, line);
        exit(-1);
    }
}
int loadImg(char* fileIN, char* fileOUT)
{
  printf("Reading image...\n");
  int channels;
  image = stbi_load(fileIN, &width, &height, &channels, 0);
  if (!image) {
    fprintf(stderr, "Couldn't load image.\n");
    return (-1);
  }
  printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,channels);

  printf("Filtrando\n");
  // Transferir la imagen desde la memoria del sistema a la memoria de la GPU
  unsigned char *d_image;
  cudaMalloc((void **)&d_image, width * height * channels * sizeof(unsigned char));
  cudaMemcpy(d_image, image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

  // Ejecutar el kernel para aplicar el filtro
  eq_GPU(d_image, width, height, channels);

  // Transferir la imagen resultante desde la memoria de la GPU a la memoria del sistema
  cudaMemcpy(image, d_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  printf("Escribiendo\n");
  // ESCRITURA DE LA IMAGEN EN SECUENCIAL
  stbi_write_png(fileOUT, width, height, channels, image, 0);

  // Liberar la memoria de la GPU
  cudaFree(d_image);

  return(0);
}

__global__ void eq_GPU(unsigned char *input_ptr)
{
    shared int histogram[256];

int tx = threadIdx.x;

// Inicialización del histograma
if(tx < 256)
{
    histogram[tx] = 0;
}

__syncthreads();

// Cálculo del histograma
int i = tx + blockIdx.x * blockDim.x;
while(i < height*width*3)
{
    int r = input_ptr[i+0];
    int g = input_ptr[i+1];
    int b = input_ptr[i+2];

    int Y = (int) (16 + 0.25679890625*r + 0.50412890625*g + 0.09790625*b);
    int Cb = (int) (128 - 0.168736*r - 0.331264*g +0.5*b);
    int Cr = (int) (128 + 0.5*r - 0.418688*g - 0.081312*b);

    input_ptr[i+0] = Y;
    input_ptr[i+1] = Cb;
    input_ptr[i+2] = Cr;

    atomicAdd(&histogram[Y], 1);

    i += blockDim.x * gridDim.x;
}

__syncthreads();

// Cálculo de la ecualización del histograma
int sum = 0;
int histogram_equalized[256] = {0};
for(int i = 0; i < 256; i++){
    sum += histogram[i];
    histogram_equalized[i] = (int) (((((float)sum - histogram[0]))/(((float)width*height - 1)))*255);
}

__syncthreads();

// Ecualización del histograma
i = tx + blockIdx.x * blockDim.x;
while(i < height*width*3)
{
    int Y = input_ptr[i];
    int Cb = input_ptr[i+1];
    int Cr = input_ptr[i+2];

    int R = max(0, min(255, (int) (Y + 1.402*(Cr-128))));
    int G = max(0, min(255, (int) (Y - 0.344136*(Cb-128) - 0.714136*(Cr-128))));
    int B = max(0, min(255, (int) (Y + 1.772*(Cb- 128))));

    input_ptr[i+0] = histogram_equalized[Y];
    input_ptr[i+1] = Cb;
    input_ptr[i+2] = Cr;

    input_ptr[i+0] = R;
    input_ptr[i+1] = G;
    input_ptr[i+2] = B;

    i += blockDim.x * gridDim.x * 3;
}
}
int main()
{
    // Definir el tamaño de la imagen
    int height = 512;
    int width = 512;

    // Cargar la imagen desde un archivo
    unsigned char* input_data = load_image("input.jpg", height, width);

    // Calcular el tamaño de la memoria necesaria para la imagen en la GPU
    int input_size = height * width * 3 * sizeof(unsigned char);

    // Reservar memoria en la GPU para la imagen de entrada y copiarla desde el host
    unsigned char* d_input_data;
    cudaMalloc((void**) &d_input_data, input_size);
    cudaMemcpy(d_input_data, input_data, input_size, cudaMemcpyHostToDevice);

    // Calcular el número de bloques y threads por bloque
    int num_threads = 256;
    int num_blocks = (height * width * 3 + num_threads - 1) / num_threads;

    // Llamar a la función eq_GPU en la GPU
    eq_GPU<<<num_blocks, num_threads>>>(d_input_data);

    // Copiar la imagen procesada de la GPU al host
    cudaMemcpy(input_data, d_input_data, input_size, cudaMemcpyDeviceToHost);

    // Guardar la imagen procesada en un archivo
    save_image("output.jpg", input_data, height, width);

    // Liberar la memoria en la GPU y el host
    cudaFree(d_input_data);
    free(input_data);

    return 0;
}

