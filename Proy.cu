#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <sys/times.h>
#include <sys/resource.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

// Funciones del kernel
__global__ void histogram_kernel(unsigned char *input_ptr, int *histogram, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int px = idx / 3;
    int channel = idx % 3;
if (px < width * height && channel == 0) {
    int Y = (int)(16 + 0.25679890625 * input_ptr[px * 3 + 0] + 0.50412890625 * input_ptr[px * 3 + 1] + 0.09790625 * input_ptr[px * 3 + 2]);
    atomicAdd(&histogram[Y], 1);
}
}


__global__ void equalize_kernel(unsigned char *input_ptr, int *histogram_equalized, int width, int height) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int px = idx / 3;
    int channel = idx % 3;
if (px < width * height && channel == 0) {
    int value_before = input_ptr[idx];
    int value_after = histogram_equalized[value_before];
    input_ptr[idx] = value_after;
}
}

__global__ void ycbcr_kernel(unsigned char *input_ptr, int width, int height, bool toYCbCr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height * 3) {
        int r = input_ptr[idx + 0];
        int g = input_ptr[idx + 1];
        int b = input_ptr[idx + 2];

        if (toYCbCr) {
            int Y = (int) (16 + 0.25679890625 * r + 0.50412890625 * g + 0.09790625 * b);
            int Cb = (int) (128 - 0.168736 * r - 0.331264 * g + 0.5 * b);
            int Cr = (int) (128 + 0.5 * r - 0.418688 * g - 0.081312 * b);

            input_ptr[idx + 0] = Y;
            input_ptr[idx + 1] = Cb;
            input_ptr[idx + 2] = Cr;
        } else {
            int Y = r;
            int Cb = g;
            int Cr = b;

            int R = max(0, min(255, (int) (Y + 1.402 * (Cr - 128))));
            int G = max(0, min(255, (int) (Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128))));
            int B = max(0, min(255, (int) (Y + 1.772 * (Cb - 128))));

            input_ptr[idx + 0] = R;
            input_ptr[idx + 1] = G;
            input_ptr[idx + 2] = B;
        }
   

}}

// Función para verificar los errores de CUDA
void CheckCudaError(char sms[], int line) {
    cudaError_t error;
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s at line %d\n", sms, line);
        exit(-1);
    }
}

int loadImg(char* fileIN, char* fileOUT) {
  printf("Reading image...\n");
  int channels;
  unsigned char *image = stbi_load(fileIN, &width, &height, &channels, 0);
  if (!image) {
    fprintf(stderr, "Couldn't load image.\n");
    return (-1);
  }
  printf("Image Read. Width : %d, Height : %d, nComp: %d\n", width, height, channels);

  printf("Filtrando\n");
  // Transferir la imagen desde la memoria del sistema a la memoria de la GPU
  unsigned char *d_image;
  cudaMalloc((void **)&d_image, width * height * channels * sizeof(unsigned char));
  cudaMemcpy(d_image, image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

  // Ejecutar el kernel para aplicar el filtro
  dim3 block_dim(256, 1, 1);
  dim3 grid_dim((width * height * channels + block_dim.x - 1) / block_dim.x, 1, 1);
  eq_GPU<<<grid_dim, block_dim>>>(d_image, width, height, channels);

  // Transferir la imagen resultante desde la memoria de la GPU a la memoria del sistema
  cudaMemcpy(image, d_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  printf("Escribiendo\n");
  // ESCRITURA DE LA IMAGEN EN SECUENCIAL
  stbi_write_png(fileOUT, width, height, channels, image, 0);

  // Liberar la memoria de la GPU
  cudaFree(d_image);

  // Liberar la memoria de la imagen
  stbi_image_free(image);

  return (0);
}

__global__ void eq_GPU(unsigned char *input_ptr, int width, int height, int channels)
{
    // Se define el histograma como una variable compartida entre los hilos de un bloque
    __shared__ int histogram[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Se inicializa el histograma en 0
    if (threadIdx.x < 256) {
        histogram[threadIdx.x] = 0;
    }

    __syncthreads();

    // Cada hilo procesa un pixel de la imagen
    if (idx < width * height * channels) {
        int Y = (int) (16 + 0.25679890625 * input_ptr[idx + 0] + 0.50412890625 * input_ptr[idx + 1] + 0.09790625 * input_ptr[idx + 2]);

        // Cada hilo aumenta en 1 el valor del histograma en la posición correspondiente
        atomicAdd(&histogram[Y], 1);
    }

    __syncthreads();

    // Se calcula el histograma acumulado
    for (int i = 1; i < 256; i++) {
        histogram[i] += histogram[i - 1];
    }

    __syncthreads();

    // Se normaliza el histograma acumulado
    float normalization_factor = 255.0f / (width * height);
    for (int i = 0; i < 256; i++) {
        histogram[i] = (int)(histogram[i] * normalization_factor + 0.5f);
        histogram[i] = min(histogram[i], 255);
    }

    __syncthreads();

    // Se aplica la ecualización del histograma a la imagen
    if (idx < width * height * channels) {
        int value_before = input_ptr[idx];
        int value_after = histogram[value_before];
        input_ptr[idx] = value_after;
    }
}

  int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <input_file> <output_file>" << endl;
        return (-1);
    }

    char* fileIN = argv[1];
    char* fileOUT = argv[2];

    // Cargar imagen
    int width, height, channels;
    unsigned char* image = stbi_load(fileIN, &width, &height, &channels, 0);
    if (!image) {
        fprintf(stderr, "Couldn't load image.\n");
        return (-1);
    }

    // Reservar memoria en la GPU
    unsigned char* d_image;
    cudaMalloc((void **)&d_image, width * height * channels * sizeof(unsigned char));
    cudaMemcpy(d_image, image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Definir dimensiones del grid y del bloque
    dim3 block_dim(256, 1, 1);
    dim3 grid_dim((width * height * channels + block_dim.x - 1) / block_dim.x, 1, 1);

    // Crear histograma
    int* histogram;
    cudaMallocManaged(&histogram, 256 * sizeof(int));
    cudaMemset(histogram, 0, 256 * sizeof(int));

    // Cambios: Convertir la imagen de RGB a YCbCr
    ycbcr_kernel<<<grid_dim, block_dim>>>(d_image, width, height, /*toYCbCr=*/true);

    // Ejecutar kernel para crear histograma
    histogram_kernel<<<grid_dim, block_dim>>>(d_image, histogram, width, height);

    // Verificar errores de CUDA
    CheckCudaError((char *)"Error creando histograma", __LINE__);

    // Calcular el histograma acumulado
    int* histogram_accumulated;
    cudaMallocManaged(&histogram_accumulated, 256 * sizeof(int));
    cudaMemset(histogram_accumulated, 0, 256 * sizeof(int));
    int sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += histogram[i];
        histogram_accumulated[i] = sum;
    }

    // Verificar errores de CUDA
    CheckCudaError((char *)"Error calculando histograma acumulado", __LINE__);

    // Crear arreglo de histograma equalizado
    int* histogram_equalized;
    cudaMallocManaged(&histogram_equalized, 256 * sizeof(int));
    cudaMemset(histogram_equalized, 0, 256 * sizeof(int
    for (int i = 0; i < 256; i++) {
        histogram_equalized[i] = (int) (255.0f * histogram_accumulated[i] / (width * height));
    }

    // Verificar errores de CUDA
    CheckCudaError((char *)"Error creando histograma equalizado", __LINE__);

    // Ejecutar kernel para equalizar la imagen
    equalize_kernel<<<grid_dim, block_dim>>>(d_image, histogram_equalized, width, height);

    // Verificar errores de CUDA
    CheckCudaError((char *)"Error al ejecutar kernel de equalización", __LINE__);

    // Cambios: Convertir la imagen de YCbCr a RGB
    ycbcr_kernel<<<grid_dim, block_dim>>>(d_image, width, height, /*toYCbCr=*/false);

    // Verificar errores de CUDA
    CheckCudaError((char *)"Error al convertir la imagen a RGB", __LINE__);

    // Transferir la imagen de la GPU al CPU
    cudaMemcpy(image, d_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // Guardar la imagen resultante
    stbi_write_png(fileOUT, width, height, channels, image, width * channels);

    // Liberar memoria de la GPU y CPU
    cudaFree(d_image);
    cudaFree(histogram);
    cudaFree(histogram_accumulated);
    cudaFree(histogram_equalized);
    stbi_image_free(image);

    return 0;
}


