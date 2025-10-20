#ifndef CACTUS_STB_IMAGE_IMPL_H
#define CACTUS_STB_IMAGE_IMPL_H

#define STB_IMAGE_IMPLEMENTATION

#define STBI_NO_BMP
#define STBI_NO_PSD
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_TGA

/* Optionally disable SIMD for smaller generated code (slower).
   #define STBI_NO_SIMD
*/

#include "stb_image.h"

#endif
