/*
 * INF560
 *
 * Image Filtering Project
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
#include <omp.h>
#include <mpi.h>

#include "gif_lib.h"

 /* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0
#define TEST_DEBUG 0
#define FULL_VERBOSE 0

#define DURATION(t1, t2) ((t2).tv_sec - (t1).tv_sec) + (((t2).tv_usec - (t1).tv_usec) / 1e6)

// uncomment this for round-robin mode (images will be treated by 0 1 2 3 0 1 2 3)
// if not defined, images will be treated in blocks (0 0 1 1 2 2 3 3), which allows transferring data with a single gather

// #define ROUND_ROBIN
#define FAST_BLUR

// defines the images through which rank mpi_rank should go
#ifdef ROUND_ROBIN
#define LOOP_IMAGE i = mpi_rank; i < image->n_images; i += mpi_size
#else
#define LOOP_IMAGE i = mpi_rank * image->n_images / mpi_size; i < (mpi_rank + 1) * image->n_images / mpi_size; i++
#endif

/* Represent one pixel from the image */
typedef struct pixel
{
    int r; /* Red */
    int g; /* Green */
    int b; /* Blue */
} pixel;

/* Represent one GIF image (animated or not */
typedef struct animated_gif
{
    int n_images; /* Number of images */
    int* width; /* Width of each image */
    int* height; /* Height of each image */
    pixel** p; /* Pixels of each image */
    uint8_t** output;   /* compressed output: i-th pixel of output image k goes into output[k][i/8] & (1 << (i%8))  (ie. we simply use one bit per output pixel, since they are black or white) */
    GifFileType* g; /* Internal representation.
                         DO NOT MODIFY */
} animated_gif;

int mpi_rank, mpi_size;

void run_kernel();

// Load a GIF image from a file and return a structure of type animated_gif.
animated_gif* load_pixels(char* filename)
{
    GifFileType* g;
    ColorMapObject* colmap;
    int error;
    int n_images;
    int* width;
    int* height;
    pixel** p;
    uint8_t** output;
    int i;
    animated_gif* image;

    /* Open the GIF image (read mode) */
    g = DGifOpenFileName(filename, &error);
    if (g == NULL)
    {
        fprintf(stderr, "Error DGifOpenFileName %s\n", filename);
        return NULL;
    }

    /* Read the GIF image */
    error = DGifSlurp(g);
    if (error != GIF_OK)
    {
        fprintf(stderr,
            "Error DGifSlurp: %d <%s>\n", error, GifErrorString(g->Error));
        return NULL;
    }

    /* Grab the number of images and the size of each image */
    n_images = g->ImageCount;

    width = (int*) malloc(n_images * sizeof(int));
    if (width == NULL)
    {
        fprintf(stderr, "Unable to allocate width of size %d\n",
            n_images);
        return 0;
    }

    height = (int*) malloc(n_images * sizeof(int));
    if (height == NULL)
    {
        fprintf(stderr, "Unable to allocate height of size %d\n",
            n_images);
        return 0;
    }

    /* Fill the width and height */
    for (i = 0; i < n_images; i++)
    {
        width[i] = g->SavedImages[i].ImageDesc.Width;
        height[i] = g->SavedImages[i].ImageDesc.Height;

#if SOBELF_DEBUG
        printf("Image %d: l:%d t:%d w:%d h:%d interlace:%d localCM:%p\n",
            i,
            g->SavedImages[i].ImageDesc.Left,
            g->SavedImages[i].ImageDesc.Top,
            g->SavedImages[i].ImageDesc.Width,
            g->SavedImages[i].ImageDesc.Height,
            g->SavedImages[i].ImageDesc.Interlace,
            g->SavedImages[i].ImageDesc.ColorMap
        );
#endif
    }


    /* Get the global colormap */
    colmap = g->SColorMap;
    if (colmap == NULL)
    {
        fprintf(stderr, "Error global colormap is NULL\n");
        return NULL;
    }

#if SOBELF_DEBUG
    printf("Global color map: count:%d bpp:%d sort:%d\n",
        g->SColorMap->ColorCount,
        g->SColorMap->BitsPerPixel,
        g->SColorMap->SortFlag
    );
#endif

    /* Allocate the array of pixels to be returned */
    p = (pixel**) malloc(n_images * sizeof(pixel*));
    if (p == NULL)
    {
        fprintf(stderr, "Unable to allocate array of %d images\n",
            n_images);
        return NULL;
    }

    // for (i = 0; i < n_images; i++)
    // {
    //     p[i] = (pixel*) malloc(width[i] * height[i] * sizeof(pixel));
    //     if (p[i] == NULL)
    //     {
    //         fprintf(stderr, "Unable to allocate %d-th array of %d pixels\n",
    //             i, width[i] * height[i]);
    //         return NULL;
    //     }
    // }

    output = (uint8_t**) malloc(n_images * sizeof(uint8_t*));

    // we allocate one big array per mpi rank to be able to send it all at once
    for (int r = 0; r < mpi_size; r++)
    {
        // this rank will handle images from r * n_images / size to (r+1) * n_images / size
        int nPixels = 0;
        for (int i = r * n_images / mpi_size; i < (r + 1) * n_images / mpi_size; i++)
            nPixels += width[i] * height[i];

        // allocate one big array for the whole rank
        p[r * n_images / mpi_size] = (pixel*) malloc(nPixels * sizeof(pixel));
        output[r * n_images / mpi_size] = (uint8_t*) calloc((nPixels + 7) / 8 + n_images, 1);   // we initialize it to zero

        // update pointers for consecutive images
        for (int i = r * n_images / mpi_size; i < (r + 1) * n_images / mpi_size - 1; i++)
        {
            p[i + 1] = p[i] + width[i] * height[i];
            output[i + 1] = output[i] + (width[i] * height[i] + 7) / 8;
        }
    }

    /* Fill pixels */

    /* For each image */
    for (i = 0; i < n_images; i++)
    {
        int j;

        /* Get the local colormap if needed */
        if (g->SavedImages[i].ImageDesc.ColorMap)
        {

            // MODIFICATION: we now support local color map (yay!)
            // fprintf(stderr, "Error: application does not support local colormap\n");
            // return NULL;

            colmap = g->SavedImages[i].ImageDesc.ColorMap;
        }

        /* Traverse the image and fill pixels */
        for (j = 0; j < width[i] * height[i]; j++)
        {
            int c;

            c = g->SavedImages[i].RasterBits[j];

            // MODIFICATION: use last frame color for SBackGroundColor
            if (c == g->SBackGroundColor)
            {
                if (i == 0)
                    p[i][j] = (pixel){ 0, 0, 0 };
                else
                    p[i][j] = p[i - 1][j];
            }

            p[i][j].r = colmap->Colors[c].Red;
            p[i][j].g = colmap->Colors[c].Green;
            p[i][j].b = colmap->Colors[c].Blue;
        }
    }

    /* Allocate image info */
    image = (animated_gif*) malloc(sizeof(animated_gif));
    if (image == NULL)
    {
        fprintf(stderr, "Unable to allocate memory for animated_gif\n");
        return NULL;
    }

    /* Fill image fields */
    image->n_images = n_images;
    image->width = width;
    image->height = height;
    image->p = p;
    image->g = g;
    image->output = output;

#if SOBELF_DEBUG
    printf("-> GIF w/ %d image(s) with first image of size %d x %d\n",
        image->n_images, image->width[0], image->height[0]);
#endif

    return image;
}

int output_modified_read_gif(char* filename, GifFileType* g)
{
    GifFileType* g2;
    int error2;

#if SOBELF_DEBUG
    printf("Starting output to file %s\n", filename);
#endif

    g2 = EGifOpenFileName(filename, false, &error2);
    if (g2 == NULL)
    {
        fprintf(stderr, "Error EGifOpenFileName %s\n",
            filename);
        return 0;
    }

    g2->SWidth = g->SWidth;
    g2->SHeight = g->SHeight;
    g2->SColorResolution = g->SColorResolution;
    g2->SBackGroundColor = g->SBackGroundColor;
    g2->AspectByte = g->AspectByte;
    g2->SColorMap = g->SColorMap;
    g2->ImageCount = g->ImageCount;
    g2->SavedImages = g->SavedImages;
    g2->ExtensionBlockCount = g->ExtensionBlockCount;
    g2->ExtensionBlocks = g->ExtensionBlocks;

    error2 = EGifSpew(g2);
    if (error2 != GIF_OK)
    {
        fprintf(stderr, "Error after writing g2: %d <%s>\n",
            error2, GifErrorString(g2->Error));
        return 0;
    }

    return 1;
}

int store_pixels(char* filename, animated_gif* image)
{
    printf("storing pixels\n");
    int n_colors = 0;
    pixel** p;
    int i, j, k;
    GifColorType* colormap;

    /* Initialize the new set of colors */
    colormap = (GifColorType*) malloc(256 * sizeof(GifColorType));
    if (colormap == NULL)
    {
        fprintf(stderr,
            "Unable to allocate 256 colors\n");
        return 0;
    }

    /* Everything is white by default */
    for (i = 0; i < 256; i++)
    {
        colormap[i].Red = 255;
        colormap[i].Green = 255;
        colormap[i].Blue = 255;
    }

    /* Change the background color and store it */
    int moy;
    moy = (
        image->g->SColorMap->Colors[image->g->SBackGroundColor].Red
        +
        image->g->SColorMap->Colors[image->g->SBackGroundColor].Green
        +
        image->g->SColorMap->Colors[image->g->SBackGroundColor].Blue
        ) / 3;
    if (moy < 0) moy = 0;
    if (moy > 255) moy = 255;

#if SOBELF_DEBUG
    printf("[DEBUG] Background color (%d,%d,%d) -> (%d,%d,%d)\n",
        image->g->SColorMap->Colors[image->g->SBackGroundColor].Red,
        image->g->SColorMap->Colors[image->g->SBackGroundColor].Green,
        image->g->SColorMap->Colors[image->g->SBackGroundColor].Blue,
        moy, moy, moy);
#endif

    colormap[0].Red = moy;
    colormap[0].Green = moy;
    colormap[0].Blue = moy;

    image->g->SBackGroundColor = 0;

    n_colors++;

    /* Process extension blocks in main structure */
    for (j = 0; j < image->g->ExtensionBlockCount; j++)
    {
        int f;

        f = image->g->ExtensionBlocks[j].Function;
        if (f == GRAPHICS_EXT_FUNC_CODE)
        {
            int tr_color = image->g->ExtensionBlocks[j].Bytes[3];

            if (tr_color >= 0 &&
                tr_color < 255)
            {

                int found = -1;

                moy =
                    (
                        image->g->SColorMap->Colors[tr_color].Red
                        +
                        image->g->SColorMap->Colors[tr_color].Green
                        +
                        image->g->SColorMap->Colors[tr_color].Blue
                        ) / 3;
                if (moy < 0) moy = 0;
                if (moy > 255) moy = 255;

#if SOBELF_DEBUG
                printf("[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                    i,
                    image->g->SColorMap->Colors[tr_color].Red,
                    image->g->SColorMap->Colors[tr_color].Green,
                    image->g->SColorMap->Colors[tr_color].Blue,
                    moy, moy, moy);
#endif

                for (k = 0; k < n_colors; k++)
                {
                    if (
                        moy == colormap[k].Red
                        &&
                        moy == colormap[k].Green
                        &&
                        moy == colormap[k].Blue
                        )
                    {
                        found = k;
                    }
                }
                if (found == -1)
                {
                    if (n_colors >= 256)
                    {
                        fprintf(stderr,
                            "Error: Found too many colors inside the image\n"
                        );
                        return 0;
                    }

#if SOBELF_DEBUG
                    printf("[DEBUG]\tNew color %d\n",
                        n_colors);
#endif

                    colormap[n_colors].Red = moy;
                    colormap[n_colors].Green = moy;
                    colormap[n_colors].Blue = moy;


                    image->g->ExtensionBlocks[j].Bytes[3] = n_colors;

                    n_colors++;
                }
                else
                {
#if SOBELF_DEBUG
                    printf("[DEBUG]\tFound existing color %d\n",
                        found);
#endif
                    image->g->ExtensionBlocks[j].Bytes[3] = found;
                }
            }
        }
    }

    for (i = 0; i < image->n_images; i++)
    {
        for (j = 0; j < image->g->SavedImages[i].ExtensionBlockCount; j++)
        {
            int f;

            f = image->g->SavedImages[i].ExtensionBlocks[j].Function;
            if (f == GRAPHICS_EXT_FUNC_CODE)
            {
                int tr_color = image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3];

                if (tr_color >= 0 &&
                    tr_color < 255)
                {

                    int found = -1;

                    moy =
                        (
                            image->g->SColorMap->Colors[tr_color].Red
                            +
                            image->g->SColorMap->Colors[tr_color].Green
                            +
                            image->g->SColorMap->Colors[tr_color].Blue
                            ) / 3;
                    if (moy < 0) moy = 0;
                    if (moy > 255) moy = 255;

#if SOBELF_DEBUG
                    printf("[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                        i,
                        image->g->SColorMap->Colors[tr_color].Red,
                        image->g->SColorMap->Colors[tr_color].Green,
                        image->g->SColorMap->Colors[tr_color].Blue,
                        moy, moy, moy);
#endif

                    for (k = 0; k < n_colors; k++)
                    {
                        if (
                            moy == colormap[k].Red
                            &&
                            moy == colormap[k].Green
                            &&
                            moy == colormap[k].Blue
                            )
                        {
                            found = k;
                        }
                    }
                    if (found == -1)
                    {
                        if (n_colors >= 256)
                        {
                            fprintf(stderr,
                                "Error: Found too many colors inside the image\n"
                            );
                            return 0;
                        }

#if SOBELF_DEBUG
                        printf("[DEBUG]\tNew color %d\n",
                            n_colors);
#endif

                        colormap[n_colors].Red = moy;
                        colormap[n_colors].Green = moy;
                        colormap[n_colors].Blue = moy;


                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = n_colors;

                        n_colors++;
                    }
                    else
                    {
#if SOBELF_DEBUG
                        printf("[DEBUG]\tFound existing color %d\n",
                            found);
#endif
                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = found;
                    }
                }
            }
        }
    }

#if SOBELF_DEBUG
    printf("[DEBUG] Number of colors after background and transparency: %d\n",
        n_colors);
#endif

    p = image->p;

    /* Find the number of colors inside the image */
    for (i = 0; i < image->n_images; i++)
    {

#if SOBELF_DEBUG
        printf("OUTPUT: Processing image %d (total of %d images) -> %d x %d\n",
            i, image->n_images, image->width[i], image->height[i]);
#endif

        for (j = 0; j < image->width[i] * image->height[i]; j++)
        {
            int found = 0;
            int value = image->output[i][j / 8] & ((uint8_t) 1 << (j % 8)) ? 255 : 0;

            for (k = 0; k < n_colors; k++)
            {
                // if ((p[i][j].r == colormap[k].Red && p[i][j].g == colormap[k].Green && p[i][j].b == colormap[k].Blue) != (value == colormap[k].Red && value == colormap[k].Green && value == colormap[k].Blue))
                //     printf("oopsie i j %d %d values %d %d %d %d\n", i, j, p[i][j].r, p[i][j].g, p[i][j].b, value);
                // if (p[i][j].r == colormap[k].Red &&
                //     p[i][j].g == colormap[k].Green &&
                //     p[i][j].b == colormap[k].Blue)
                if (value == colormap[k].Red && value == colormap[k].Green && value == colormap[k].Blue)
                    found = 1;
            }

            if (found == 0)
            {
                if (n_colors >= 256)
                {
                    fprintf(stderr,
                        "Error: Found too many colors inside the image\n"
                    );
                    return 0;
                }

#if SOBELF_DEBUG
                printf("[DEBUG] Found new %d color (%d,%d,%d)\n",
                    n_colors, p[i][j].r, p[i][j].g, p[i][j].b);
#endif

                // colormap[n_colors].Red = p[i][j].r;
                // colormap[n_colors].Green = p[i][j].g;
                // colormap[n_colors].Blue = p[i][j].b;
                colormap[n_colors].Red = value;
                colormap[n_colors].Green = value;
                colormap[n_colors].Blue = value;
                n_colors++;
            }
        }
    }

#if SOBELF_DEBUG
    printf("OUTPUT: found %d color(s)\n", n_colors);
#endif


    /* Round up to a power of 2 */
    if (n_colors != (1 << GifBitSize(n_colors)))
    {
        n_colors = (1 << GifBitSize(n_colors));
    }

#if SOBELF_DEBUG
    printf("OUTPUT: Rounding up to %d color(s)\n", n_colors);
#endif

    /* Change the color map inside the animated gif */
    ColorMapObject* cmo;

    cmo = GifMakeMapObject(n_colors, colormap);
    if (cmo == NULL)
    {
        fprintf(stderr, "Error while creating a ColorMapObject w/ %d color(s)\n",
            n_colors);
        return 0;
    }

    image->g->SColorMap = cmo;

    /* Update the raster bits according to color map */
    for (i = 0; i < image->n_images; i++)
    {
        // MODIFICATION: remove local color map for each image
        // otherwise, we use the local color map of the corresponding image in the original gif

        image->g->SavedImages[i].ImageDesc.ColorMap = NULL;

        for (j = 0; j < image->width[i] * image->height[i]; j++)
        {
            int value = image->output[i][j / 8] & ((uint8_t) 1 << (j % 8)) ? 255 : 0;
            int found_index = -1;
            for (k = 0; k < n_colors; k++)
                //     if (p[i][j].r == image->g->SColorMap->Colors[k].Red &&
                //         p[i][j].g == image->g->SColorMap->Colors[k].Green &&
                //         p[i][j].b == image->g->SColorMap->Colors[k].Blue)
                if (value == image->g->SColorMap->Colors[k].Red &&
                    value == image->g->SColorMap->Colors[k].Green &&
                    value == image->g->SColorMap->Colors[k].Blue)
                    found_index = k;

            if (found_index == -1)
            {
                fprintf(stderr,
                    "Error: Unable to find a pixel in the color map\n");
                return 0;
            }

            image->g->SavedImages[i].RasterBits[j] = found_index;
        }
    }


    /* Write the final image */
    if (!output_modified_read_gif(filename, image->g)) { return 0; }

    return 1;
}

void apply_gray_filter(animated_gif* image)
{
    int i, j;
    pixel** p;

    p = image->p;

#pragma omp parallel private(i,j)
    {
#if TEST_DEBUG
        printf("Gray filter applied by thread %d/%d\n", omp_get_thread_num(), omp_get_num_threads());
#endif
#pragma omp for schedule(static)

        for (LOOP_IMAGE)
        {
            for (j = 0; j < image->width[i] * image->height[i]; j++)
            {
                int moy;

                moy = (p[i][j].r + p[i][j].g + p[i][j].b) / 3;
                if (moy < 0) moy = 0;
                if (moy > 255) moy = 255;

                // p[i][j].r = moy;
                // p[i][j].g = moy;
                p[i][j].b = moy;
            }
        }
    }
}

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

void apply_gray_line(animated_gif* image)
{
    int i, j, k;
    pixel** p;

    p = image->p;

    for (LOOP_IMAGE)
    {
        for (j = 0; j < 10; j++)
        {
            for (k = image->width[i] / 2; k < image->width[i]; k++)
            {
                p[i][CONV(j, k, image->width[i])].r = 0;
                p[i][CONV(j, k, image->width[i])].g = 0;
                p[i][CONV(j, k, image->width[i])].b = 0;
            }
        }
    }
}

void apply_blur_filter(animated_gif* image, int size, int threshold)
{
    int i, j, k;
    int width, height;
    int end = 0;
    int n_iter = 0;

    pixel** p;
    pixel* new;

    printf("size = %d threshold = %d\n", size, threshold);

    /* Get the pixels of all images */
    p = image->p;

    /* Process all images */
#pragma omp parallel private(i,j,k,width,height, end, n_iter, new)
    {
#if TEST_DEBUG
        printf("Blur filter applied by thread %d/%d\n", omp_get_thread_num(), omp_get_num_threads());
#endif
#pragma omp for schedule(static)
        for (LOOP_IMAGE)
        {
            n_iter = 0;
            width = image->width[i];
            height = image->height[i];

            /* Allocate array of new pixels */
            new = (pixel*) malloc(width * height * sizeof(pixel));

            // pixel* neww = (pixel*) malloc(width * height * sizeof(pixel));
            // for (int j = 0; j < height; j++)
            //     for (int k = 0; k < width; k++)
            //         neww[CONV(j, k, width)].b = p[i][CONV(j, k, width)].b;

            // sum[k] is the sum of p[i][j] for k-size <= j <= k+size (ie. partial sums over rows), if k is not on a border
            int* sum = (int*) malloc(width * height * sizeof(int));

#ifndef FAST_BLUR
            /* Perform at least one blur iteration */
            do
            {
                end = 1;
                n_iter++;

                for (j = 0; j < height - 1; j++)
                {
                    for (k = 0; k < width - 1; k++)
                    {
                        // new[CONV(j, k, width)].r = p[i][CONV(j, k, width)].r;
                        // new[CONV(j, k, width)].g = p[i][CONV(j, k, width)].g;
                        new[CONV(j, k, width)].b = p[i][CONV(j, k, width)].b;
                    }
                }

                /* Apply blur on top part of image (10%) */
                for (j = size; j < height / 10 - size; j++)
                {
                    for (k = size; k < width - size; k++)
                    {
                        int stencil_j, stencil_k;
                        // int t_r = 0;
                        // int t_g = 0;
                        int t_b = 0;

                        for (stencil_j = -size; stencil_j <= size; stencil_j++)
                        {
                            for (stencil_k = -size; stencil_k <= size; stencil_k++)
                            {
                                // t_r += p[i][CONV(j + stencil_j, k + stencil_k, width)].r;
                                // t_g += p[i][CONV(j + stencil_j, k + stencil_k, width)].g;
                                t_b += p[i][CONV(j + stencil_j, k + stencil_k, width)].b;
                            }
                        }

                        // new[CONV(j, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
                        // new[CONV(j, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
                        new[CONV(j, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
                    }
                }

                /* Copy the middle part of the image */
                // for (j = height / 10 - size; j < height * 0.9 + size; j++)
                // {
                //     for (k = size; k < width - size; k++)
                //     {
                //         new[CONV(j, k, width)].r = p[i][CONV(j, k, width)].r;
                //         new[CONV(j, k, width)].g = p[i][CONV(j, k, width)].g;
                //         new[CONV(j, k, width)].b = p[i][CONV(j, k, width)].b;
                //     }
                // }

                /* Apply blur on the bottom part of the image (10%) */
                for (j = height * 0.9 + size; j < height - size; j++)
                {
                    for (k = size; k < width - size; k++)
                    {
                        int stencil_j, stencil_k;
                        // int t_r = 0;
                        // int t_g = 0;
                        int t_b = 0;

                        for (stencil_j = -size; stencil_j <= size; stencil_j++)
                        {
                            for (stencil_k = -size; stencil_k <= size; stencil_k++)
                            {
                                // t_r += p[i][CONV(j + stencil_j, k + stencil_k, width)].r;
                                // t_g += p[i][CONV(j + stencil_j, k + stencil_k, width)].g;
                                t_b += p[i][CONV(j + stencil_j, k + stencil_k, width)].b;
                            }
                        }

                        // new[CONV(j, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
                        // new[CONV(j, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
                        new[CONV(j, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
                    }
                }

                for (j = 1; j < height - 1; j++)
                {
                    for (k = 1; k < width - 1; k++)
                    {

                        // float diff_r;
                        // float diff_g;
                        float diff_b;

                        // diff_r = (new[CONV(j, k, width)].r - p[i][CONV(j, k, width)].r);
                        // diff_g = (new[CONV(j, k, width)].g - p[i][CONV(j, k, width)].g);
                        diff_b = (new[CONV(j, k, width)].b - p[i][CONV(j, k, width)].b);

                        if (
                            // diff_r > threshold || -diff_r > threshold
                            // ||
                            // diff_g > threshold || -diff_g > threshold
                            // ||
                            diff_b > threshold || -diff_b > threshold
                            )
                        {
                            end = 0;
                        }

                        // p[i][CONV(j, k, width)].r = new[CONV(j, k, width)].r;
                        // p[i][CONV(j, k, width)].g = new[CONV(j, k, width)].g;
                        p[i][CONV(j, k, width)].b = new[CONV(j, k, width)].b;
                    }
                }

            } while (threshold > 0 && !end);
#else

            n_iter = 0;
            do
            {
                end = 1;
                n_iter++;

                // compute sums over rows
                for (j = 0; j < height; j++)
                {
                    int index_deb = j * width;

                    // index 0 is the sum from 0 to size
                    sum[index_deb] = 0;
                    for (int k = 0; k <= size; k++)
                        sum[index_deb] += p[i][index_deb + k].b;

                    // from 1 to size, we add a new value
                    for (int k = 1; k <= size; k++)
                        sum[index_deb + k] = sum[index_deb + k - 1] + p[i][index_deb + k + size].b;

                    // from size+1 to width-size-1, we add a new value and remove one
                    for (int k = size + 1; k <= width - size - 1; k++)
                        sum[index_deb + k] = sum[index_deb + k - 1] + p[i][index_deb + k + size].b - p[i][index_deb + k - size - 1].b;

                    // from width-size to width-1, we remove one value
                    for (int k = width - size; k <= width - 1; k++)
                        sum[index_deb + k] = sum[index_deb + k - 1] - p[i][index_deb + k - size - 1].b;
                }

                // test
                for (int j = 0; j < height; j++)
                    for (int k = 0; k < width; k++)
                    {
                        int ss = 0;
                        for (int l = k >= size ? k - size : 0; l <= k + size && l < width; l++)
                            ss += p[i][CONV(j, l, width)].b;
                        if (ss != sum[CONV(j, k, width)])
                            printf("BAD SUM at j %d k %d\n", j, k);
                    }

                // update pixels value (over columns)
                for (k = size; k < width - size; k++)
                {
                    // top part of the image
                    int total_sum = 0;
                    for (int j = 0; j <= 2 * size; j++)
                        total_sum += sum[CONV(j, k, width)];

                    for (int j = size; j < height / 10 - size; j++)
                    {
                        int new_val = total_sum / ((2 * size + 1) * (2 * size + 1));
                        if (p[i][CONV(j, k, width)].b > new_val + threshold || p[i][CONV(j, k, width)].b < new_val - threshold)
                            end = 0;

                        p[i][CONV(j, k, width)].b = new_val;

                        total_sum += sum[CONV(j + size + 1, k, width)] - sum[CONV(j - size, k, width)];
                    }

                    // bottom part of the image
                    total_sum = 0;
                    for (int j = height - 1; j >= height - 2 * size - 1; j--)
                        total_sum += sum[CONV(j, k, width)];

                    for (int j = height - 1 - size; j >= height * 0.9 + size; j--)
                    {
                        int new_val = total_sum / ((2 * size + 1) * (2 * size + 1));
                        if (p[i][CONV(j, k, width)].b > new_val + threshold || p[i][CONV(j, k, width)].b < new_val - threshold)
                            end = 0;

                        p[i][CONV(j, k, width)].b = new_val;

                        total_sum += sum[CONV(j - size - 1, k, width)] - sum[CONV(j + size, k, width)];
                    }
                }
            } while (threshold > 0 && !end);

#endif

#if SOBELF_DEBUG
            printf("BLUR: number of iterations for image %d\n", n_iter);
#endif

            free(new);
        }
    }
}

void apply_sobel_filter(animated_gif* image)
{
    int i, j, k;
    int width, height;

    pixel** p;

    p = image->p;

#pragma omp parallel private(i,j,k,width,height)
    {
#if TEST_DEBUG
        printf("Sobel filter applied by thread %d/%d\n", omp_get_thread_num(), omp_get_num_threads());
#endif
#pragma omp for schedule(static)
        for (LOOP_IMAGE)
        {
            width = image->width[i];
            height = image->height[i];

            pixel* sobel;

            sobel = (pixel*) malloc(width * height * sizeof(pixel));

            for (j = 1; j < height - 1; j++)
            {
                for (k = 1; k < width - 1; k++)
                {
                    int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
                    int pixel_blue_so, pixel_blue_s, pixel_blue_se;
                    int pixel_blue_o, pixel_blue, pixel_blue_e;

                    float deltaX_blue;
                    float deltaY_blue;
                    float val_blue;

                    pixel_blue_no = p[i][CONV(j - 1, k - 1, width)].b;
                    pixel_blue_n = p[i][CONV(j - 1, k, width)].b;
                    pixel_blue_ne = p[i][CONV(j - 1, k + 1, width)].b;
                    pixel_blue_so = p[i][CONV(j + 1, k - 1, width)].b;
                    pixel_blue_s = p[i][CONV(j + 1, k, width)].b;
                    pixel_blue_se = p[i][CONV(j + 1, k + 1, width)].b;
                    pixel_blue_o = p[i][CONV(j, k - 1, width)].b;
                    pixel_blue = p[i][CONV(j, k, width)].b;
                    pixel_blue_e = p[i][CONV(j, k + 1, width)].b;

                    deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2 * pixel_blue_o + 2 * pixel_blue_e - pixel_blue_so + pixel_blue_se;

                    deltaY_blue = pixel_blue_se + 2 * pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2 * pixel_blue_n - pixel_blue_no;

                    // val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4;
                    val_blue = deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue;

                    int index = CONV(j, k, width);

                    // it's faster to avoid sqrt (sqrt(val_blue) / 4 > 50 <=> val_blue > (4 * 50)^2 = 40000)
                    // if (val_blue > 50)
                    if (val_blue > 40000)
                    {
                        // sobel[CONV(j, k, width)].r = 255;
                        // sobel[CONV(j, k, width)].g = 255;
                        // sobel[CONV(j, k, width)].b = 255;

                        // set corresponding bit to 1
                        image->output[i][index / 8] |= (uint8_t) 1 << (index % 8);
                    }
                    // output is already initialized to 0
                    // else
                    // {
                    //     sobel[CONV(j, k, width)].r = 0;
                    //     sobel[CONV(j, k, width)].g = 0;
                    //     sobel[CONV(j, k, width)].b = 0;

                    //     // set corresponding bit to 0
                    //     image->output[i][index / 8] &= ~((uint8_t) 1 << (index % 8));
                    // }
                }
            }

            // for (j = 1; j < height - 1; j++)
            // {
            //     for (k = 1; k < width - 1; k++)
            //     {
            //         p[i][CONV(j, k, width)].r = sobel[CONV(j, k, width)].r;
            //         p[i][CONV(j, k, width)].g = sobel[CONV(j, k, width)].g;
            //         p[i][CONV(j, k, width)].b = sobel[CONV(j, k, width)].b;
            //     }
            // }

            free(sobel);
        }
    }

    printf("sobelf done\n");
}

/*
 * Main entry point
 */
int main(int argc, char** argv)
{
    int mpi_thread_provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &mpi_thread_provided);

    if (mpi_thread_provided != MPI_THREAD_SINGLE)
        fprintf(stderr, "got MPI thread mode %d provided instead of MPI_THREAD_SINGLE\n", mpi_thread_provided);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    printf("MPI rank %d size %d\n", mpi_rank, mpi_size);

    char* input_filename;
    char* output_filename;
    animated_gif* image;
    struct timeval t1, t2, t3, t4, t5, t6;
    double duration;

    /* Check command-line arguments */
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s input.gif output.gif \n", argv[0]);
        return 1;
    }

    input_filename = argv[1];
    output_filename = argv[2];

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    image = load_pixels(input_filename);
    if (image == NULL) { return 1; }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = DURATION(t1, t2);

#if FULL_VERBOSE
    printf("GIF loaded from file %s with %d image(s) in %lf s\n",
        input_filename, image->n_images, duration);
#else
    printf("%s ; ", input_filename);
#endif

    /* FILTER Timer start */
    gettimeofday(&t1, NULL);

    /* Convert the pixels into grayscale */
    apply_gray_filter(image);

    gettimeofday(&t4, NULL);

    /* Apply blur filter with convergence value */
    apply_blur_filter(image, 5, 20);

    gettimeofday(&t5, NULL);

    /* Apply sobel filter on pixels */
    apply_sobel_filter(image);

    // SOBELF timer stop
    gettimeofday(&t2, NULL);

    duration = DURATION(t1, t2);

    printf("gray %lf blur %lf sobelf %lf total %lf\n", DURATION(t1, t4), DURATION(t4, t5), DURATION(t5, t2), DURATION(t1, t2));

    // now, we need to gather all images (there are ceil(n_images / mpi_size) per node at most)

    // int* tt1 = malloc(50 * sizeof(int)), * tt2 = malloc(100 * sizeof(int)), * t3 = malloc(100 * sizeof(int));
    // int displs[2] = { 0, (t3 + 4) - tt1 };
    // int recvcounts[2] = { 1, 1 };
    // int val = mpi_rank ? 12 : 9;
    // if (mpi_rank == 0)
    // {
    //     MPI_Gatherv(&val, 1, MPI_INT, tt1, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    //     printf("%d %d\n", tt1[0], t3[4]);
    // }
    // else
    //     MPI_Gatherv(&val, 1, MPI_INT, NULL, NULL, NULL, NULL, 0, MPI_COMM_WORLD);

#ifdef ROUND_ROBIN
    // the images are treated in a round-robin fashion: for 8 images and 3 nodes, it will be 0 1 2 0 1 2 0 1
    // this means that there should be N//S common GatherV with displacements 0,1,..,S-1 ; S,S+1,..,2S-1 ; etc
    // then (if N mod S != 0) a last GatherV with displacements S * (N // S) ; ... ; N-1 and the last S - (N mod S) nodes should not send anything

    // since we cannot specify arbitrary receiving adresses but only displacements, we use somewhat cheesy pointer arithmetic
    // being cautious about the fact that a pixel is (exactly) 3 ints
    // ie. the 1st image treated by rank 1 should be copied from p[0] with displacement 3 * (int)(p[1] - p[0])
    // it does not matter if recvbuffer is actually not a real buffer, we just need to specify displacements

    int* displ = malloc(mpi_size * sizeof(int));
    int* recvcount = malloc(mpi_size * sizeof(int));

    // int* recv = malloc(image->n_images * sizeof(int));

    for (int i = 0; i < image->n_images / mpi_size + (image->n_images % mpi_size != 0); i++)
    {
        // populate displ and recvcount
        for (int k = 0; k < mpi_size; k++)
        {
            if (i * mpi_size + k < image->n_images)
            {
                // displ[k] = i * mpi_size + k;
                // recvcount[k] = 1;
                int image_no = i * mpi_size + k;
                displ[k] = 3 * (int) (image->p[image_no] - image->p[0]);
                recvcount[k] = 3 * image->width[image_no] * image->height[image_no];
            }
            else
                displ[k] = recvcount[k] = 0;        // the last nodes shouldn't send anything at the last receive
        }

        void* source = i * mpi_size + mpi_rank < image->n_images ? (void*) (image->p[i * mpi_size + mpi_rank]) : NULL;

        if (mpi_rank == 0)      // rank 0 receive and sends nothing (it already has the correct images)
            MPI_Gatherv(NULL, 0, MPI_INT, image->p[0], recvcount, displ, MPI_INT, 0, MPI_COMM_WORLD);
        else
            MPI_Gatherv(source, recvcount[mpi_rank], MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
    }
#else
    // the images are treated 0 0 1 1 2 2 3 3
    // rank r gets images from r * n / s to (r+1) * n / s, and we know that these images are allocated consecutively: we only need a single gather
    // we use pointer arithmetic for displ, as we do not place everything in a single buffer
    int* displ = malloc(mpi_size * sizeof(int));
    int* recvcount = malloc(mpi_size * sizeof(int));

    displ[0] = recvcount[0] = 0;
    for (int k = 1; k < mpi_size; k++)
    {
        // displ[k] = 3 * (int) (image->p[k * image->n_images / mpi_size] - image->p[0]);
        displ[k] = (int) (image->output[k * image->n_images / mpi_size] - image->output[0]);

        int nPixels = 0;
        for (int i = k * image->n_images / mpi_size; i < (k + 1) * image->n_images / mpi_size; i++)
            nPixels += image->width[i] * image->height[i];
        recvcount[k] = (nPixels + 7) / 8 + image->n_images;
    }

    if (mpi_rank == 0)
        // we should receive a single image array from all other ranks, since we know that images are consecutive for a single rank
        MPI_Gatherv(NULL, 0, MPI_UNSIGNED_CHAR, image->output[0], recvcount, displ, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    else
        MPI_Gatherv(image->output[mpi_rank * image->n_images / mpi_size], recvcount[mpi_rank], MPI_UNSIGNED_CHAR, NULL, NULL, NULL, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
#endif

    /* COMM Timer stop */
    gettimeofday(&t3, NULL);

    double duration2 = DURATION(t2, t3);

    printf("process %d: duration1 duration2 %lf %lf\n", mpi_rank, duration, duration2);

    if (mpi_rank == 0)
    {
#if FULL_VERBOSE
        printf("SOBEL done in %lf; COMMUNICATION done in %lf\n", duration, duration2);
#else
        printf("%lf %lf\n", duration, duration2);
#endif
    }

    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Store file from array of pixels to GIF file only for rank 0 */
    if (mpi_rank == 0)
        if (!store_pixels(output_filename, image)) { return 1; }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = DURATION(t1, t2);

    run_kernel();

    MPI_Finalize();

#if FULL_VERBOSE
    printf("Export done in %lf s in file %s\n", duration, output_filename);
#endif

    return 0;
}
