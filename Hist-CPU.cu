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

char *fileIN, *fileOUT;
unsigned char *image;
int width, height, pixelWidth; //meta info de la imagen

void CheckCudaError(char sms[], int line);


int loadImg(char* fileIN, char* fileOUT)
{
  printf("Reading image...\n");
  image = stbi_load(fileIN, &width, &height, &pixelWidth, 0);
  if (!image) {
    fprintf(stderr, "Couldn't load image.\n");
    return (-1);
  }
  printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);

  printf("Filtrando\n");
  //SECUENCIAL BLANCO Y NEGRO:
  for(int i=0;i<width*height*3;i=i+3){
    image[i]=image[i];
    image[i+1]=image[i+1];
    image[i+2]=image[i+2];
  }
  printf("Escribiendo\n");
  //ESCRITURA DE LA IMAGEN EN SECUENCIAL
  stbi_write_png(fileOUT,width,height,pixelWidth,image,0);
  return(0);

}


//Histogram Equalization with CPU
static int eq_CPU(unsigned char *input_ptr){

    int histogram[256] = {0};

    for (int i = 0; i< height*width*3; i+=3){
        int r = input_ptr[i+0];
        int g = input_ptr[i+1];
        int b = input_ptr[i+2];
        
        int Y = (int) (16 + 0.25679890625*r + 0.50412890625*g + 0.09790625*b);
        int Cb = (int) (128 - 0.168736*r - 0.331264*g +0.5*b);
        int Cr = (int) (128 + 0.5*r - 0.418688*g - 0.081312*b);

        input_ptr[i+0] = Y;
        input_ptr[i+1] = Cb;
        input_ptr[i+2] = Cr;

        histogram[Y] += 1;
    }

    int sum = 0;
    int histogram_equalized[256] = {0};
    for(int i = 0; i < 256; i++){
        sum += histogram[i];
        histogram_equalized[i] = (int) (((((float)sum - histogram[0]))/(((float)width*height - 1)))*255);

    }
    for (int i = 0; i< height*width*3; i+=3){
        int value_before = input_ptr[i];
        int value_after = histogram_equalized[value_before];

        input_ptr[i] = value_after;

        int y = input_ptr[i+0];
        int cb = input_ptr[i+1];
        int cr = input_ptr[i+2];

        int R = max(0, min(255, (int) (y + 1.402*(cr-128))));
        int G = max(0, min(255, (int) (y - 0.344136*(cb-128) - 0.714136*(cr-128))));
        int B = max(0, min(255, (int) (y + 1.772*(cb- 128))));

        input_ptr[i+0] = R;
        input_ptr[i+1] = G;
        input_ptr[i+2] = B;
    }

    return 0;
}

int main(int argc, char** argv)
{
    char input[] = "./IMG/IMG00.jpg";
    char output[] = "out.jpg";
    
    if (loadImg(input, output) == 0);
    else
        return (-1);

    printf("Starting to process.. \n");
        
    eq_CPU(image);

    string output_name;
    //Save the image
    printf("Saving output image..\n");
    output_name = "output_pixel_CPU.jpg";

    const char* name2 = output_name.c_str();
    stbi_write_png(name2 ,width,height,pixelWidth,image,0);

    printf("Image saved!\n");

    return 0;
}
