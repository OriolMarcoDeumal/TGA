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
    int px = idx * 3;
    if (idx < width * height) {
        int Y = (int)(16 + 0.25679890625 * input_ptr[px] + 0.50412890625 * input_ptr[px + 1] + 0.09790625 * input_ptr[px + 2]);
	int Cb = (int) (128 - 0.168736*input_ptr[px] - 0.331264*input_ptr[px+1] +0.5*input_ptr[px+2]);
	int Cr = (int) (128 + 0.5*input_ptr[px] - 0.418688*input_ptr[px+1] - 0.081312*input_ptr[px+2]);

        input_ptr[px]=Y;
    	input_ptr[px+1] = Cb;
	input_ptr[px+2] = Cr;


    atomicAdd(&histogram[Y], 1);
}
}


__global__ void equalize_kernel(unsigned char *input_ptr, int *histogram_equalized, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int px  = idx *3;
    if (idx < width * height) {
    	int value_before = input_ptr[px];
    	int value_after = histogram_equalized[value_before];
    	input_ptr[px] = value_after;
    }
}

__global__ void ycbcr_kernel(unsigned char *input_ptr, int width, int height, bool toYCbCr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int px = idx * 3;

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

void CheckCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(error), file, line);
        exit(-1);
    }
}





int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return -1;
    }

    // Load the input image.
    int width, height, channels;
    unsigned char* image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!image) {
        fprintf(stderr, "Couldn't load image: %s\n", argv[1]);
        return -1;
    }

    // Convert the image to YCbCr.
    unsigned char* ycbcr_image = new unsigned char[width * height * 3];
    ycbcr_kernel<<<dim3(width / 16, height / 16), dim3(16, 16), 0>>>(image, width, height, true);
    cudaDeviceSynchronize();

    // Equalize the image.
    unsigned char* equalized_image = new unsigned char[width * height * 3];
    equalize_kernel<<<dim3(width / 16, height / 16), dim3(16, 16), 0>>>(ycbcr_image, width, height);
    cudaDeviceSynchronize();

    // Convert the image back to RGB.
    unsigned char* output_image = new unsigned char[width * height * 3];
    ycbcr_kernel<<<dim3(width / 16, height / 16), dim3(16, 16), 0>>>(equalized_image, width, height, false);
    cudaDeviceSynchronize();

    // Save the output image.
    stbi_write_png(argv[2], width, height, channels, output_image, width * channels);

    // Free the memory.
    delete[] image;
    delete[] ycbcr_image;
    delete[] equalized_image;
    delete[] output_image;

    return 0;
}
