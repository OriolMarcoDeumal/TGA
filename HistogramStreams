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

// Kernel functions
__global__ void histogram_kernel(unsigned char *input_ptr, int *histogram, int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int px = idx * channels;
    if (idx < width * height) {
        int Y = input_ptr[px];
        atomicAdd(&histogram[Y], 1);
    }
}

__global__ void equalize_kernel(unsigned char *input_ptr, int *histogram_equalized, int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int px = idx * channels;
    if (idx < width * height) {
        int value_before = input_ptr[px];
        int value_after = histogram_equalized[value_before];
        input_ptr[px] = value_after;
    }
}

__global__ void ycbcr_kernel(unsigned char *input_ptr, int width, int height, int channels, bool toYCbCr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int px = idx * channels;

    if (idx < width * height) {
        int r = input_ptr[px];
        int g = input_ptr[px + 1];
        int b = input_ptr[px + 2];
    
        if (toYCbCr) {
            int Y = (int) (16 + 0.25679890625 * r + 0.50412890625 * g + 0.09790625 * b);
            int Cb = (int) (128 - 0.168736 * r - 0.331264 * g + 0.5 * b);
            int Cr = (int) (128 + 0.5 * r - 0.418688 * g - 0.081312 * b);

            input_ptr[px + 0] = Y;
            input_ptr[px + 1] = Cb;
            input_ptr[px + 2] = Cr;
        } else {
            int Y = r;
            int Cb = g;
            int Cr = b;

            int R = max(0, min(255, (int) (Y + 1.402 * (Cr - 128))));
            int G = max(0, min(255, (int) (Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128))));
            int B = max(0, min(255, (int) (Y + 1.772 * (Cb - 128))));

            input_ptr[px + 0] = R;
            input_ptr[px + 1] = G;
            input_ptr[px + 2] = B;
        }
    }
}

// Function to check CUDA errors
void CheckCudaError(char sms[], int line) {
    cudaError_t error;
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s at line %d\n", sms, line);
        exit(-1);
    }
}


int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <input_file> <output_file>" << endl;
        return (-1);
    }

    char* fileIN = argv[1];
    char* fileOUT = argv[2];

    // Load image
    int width, height, channels;
    unsigned char* image = stbi_load(fileIN, &width, &height, &channels, 0);
    if (!image) {
        fprintf(stderr, "Couldn't load image.\n");
        return (-1);
    }


    // Allocate memory on the GPU
    unsigned char* d_image;
    cudaMalloc((void **)&d_image, width * height * channels * sizeof(unsigned char));

    // Define grid and block dimensions
    dim3 block_dim(256, 1, 1);
    dim3 grid_dim((width * height * channels + block_dim.x - 1) / block_dim.x, 1, 1);

    // Create histogram
    int* histogram;
    cudaMallocManaged(&histogram, 256 * sizeof(int));

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Asynchronously copy image to GPU and initialize histogram with zeros
    cudaMemcpyAsync(d_image, image, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(histogram, 0, 256 * sizeof(int), stream);

    // Convert the image from RGB to YCbCr
    ycbcr_kernel<<<grid_dim, block_dim, 0, stream>>>(d_image, width, height, channels, /*toYCbCr=*/true);

    // Execute kernel to create histogram
    histogram_kernel<<<grid_dim, block_dim, 0, stream>>>(d_image, histogram, width, height, channels);

    // Wait for the operations in the stream to finish
    cudaStreamSynchronize(stream);
    CheckCudaError((char *)"Error creating histogram", __LINE__);

     int* histogram_accumulated;
    cudaMallocManaged(&histogram_accumulated, 256 * sizeof(int));
    cudaMemset(histogram_accumulated, 0, 256 * sizeof(int));
    int sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += histogram[i];
        histogram_accumulated[i] = sum;
    }

    CheckCudaError((char *)"Error calculating accumulated histogram", __LINE__);

    // Create equalized histogram array
    int* histogram_equalized;
    cudaMallocManaged(&histogram_equalized, 256 * sizeof(int));
    cudaMemset(histogram_equalized, 0, 256 * sizeof(int));
    for (int i = 0; i < 256; i++) {
        histogram_equalized[i] = (int) (255.0f * histogram_accumulated[i] / (width * height));
    }

    CheckCudaError((char *)"Error creating equalized histogram", __LINE__);


    // Execute kernel to equalize image
    equalize_kernel<<<grid_dim, block_dim, 0, stream>>>(d_image, histogram_equalized, width, height, channels);

    // Convert the image from YCbCr to RGB
    ycbcr_kernel<<<grid_dim, block_dim, 0, stream>>>(d_image, width, height, channels, /*toYCbCr=*/false);

    // Asynchronously copy image from GPU to CPU
    cudaMemcpyAsync(image, d_image, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);

    // Wait for the operations in the stream to finish before saving the image
    cudaStreamSynchronize(stream);
    CheckCudaError((char *)"Error converting image to RGB", __LINE__);

    // Save resulting image
    stbi_write_png(fileOUT, width, height, channels, image, width * channels);

    // Free GPU and CPU memory
    cudaFree(d_image);
    cudaFree(histogram);
    cudaFree(histogram_accumulated);
    cudaFree(histogram_equalized);
    stbi_image_free(image);

    // Destroy CUDA stream
    cudaStreamDestroy(stream);

    return 0;
}
