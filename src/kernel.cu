#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../include/helper_image.h"
#include "../include/td1.h"
#include <iostream>
#include "helper_cuda.h"
// #include <windows.h>                // for Windows APIs
#define KERNEL_SIZE 7
#define EDGE 0xFFFF
#define NON_EDGE 0x0
#define EDGE_V 255
#define REPETITIONS 1

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void populate_blur_kernel(double out_kernel[KERNEL_SIZE][KERNEL_SIZE])
{
	double scaleVal = 1;
	double stDev = (double)KERNEL_SIZE / 3;

	for (int i = 0; i < KERNEL_SIZE; ++i) {
		for (int j = 0; j < KERNEL_SIZE; ++j) {
			double xComp = pow((i - KERNEL_SIZE / 2), 2);
			double yComp = pow((j - KERNEL_SIZE / 2), 2);

			double stDevSq = pow(stDev, 2);
			double pi = 3.14159;

			//calculate the value at each index of the Kernel
			double kernelVal = exp(-(((xComp)+(yComp)) / (2 * stDevSq)));
			kernelVal = (1 / (sqrt(2 * pi)*stDev)) * kernelVal;

			//populate Kernel
			out_kernel[i][j] = kernelVal;

			if (i == 0 && j == 0)
			{
				scaleVal = out_kernel[0][0];
			}

			//normalize Kernel
			out_kernel[i][j] = out_kernel[i][j] / scaleVal;
		}
	}
}

void apply_gaussian_filter(unsigned char *out_pixels, unsigned char *in_pixels, double kernel[KERNEL_SIZE][KERNEL_SIZE], int rows, int cols)
{
	double kernelSum;
	double redPixelVal;

	for (int pixNum = 0; pixNum < rows * cols; ++pixNum) {

		for (int i = 0; i < KERNEL_SIZE; ++i) {
			for (int j = 0; j < KERNEL_SIZE; ++j) {

				//check edge cases, if within bounds, apply filter
				if (((pixNum + ((i - ((KERNEL_SIZE - 1) / 2))*cols) + j - ((KERNEL_SIZE - 1) / 2)) >= 0)
					&& ((pixNum + ((i - ((KERNEL_SIZE - 1) / 2))*cols) + j - ((KERNEL_SIZE - 1) / 2)) <= rows * cols - 1)
					&& (((pixNum % cols) + j - ((KERNEL_SIZE - 1) / 2)) >= 0)
					&& (((pixNum % cols) + j - ((KERNEL_SIZE - 1) / 2)) <= (cols - 1))) {

					redPixelVal += kernel[i][j] * in_pixels[pixNum + ((i - ((KERNEL_SIZE - 1) / 2))*cols) + j - ((KERNEL_SIZE - 1) / 2)];
					kernelSum += kernel[i][j];
				}
			}
		}
		out_pixels[pixNum] = redPixelVal / kernelSum;

		redPixelVal = 0;
		kernelSum = 0;
	}
}

__global__ void apply_gaussian_filter_gpu(unsigned char *out_pixels, unsigned char *in_pixels, double *kernel, int rows, int cols)
{
	double kernelSum = 0;
	double redPixelVal = 0;
	int pixNum;
	int pixNum_filter;
	int pixNum_kernel;

	// just copy the input to the output
	/*for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		out_pixels[blockIdx.x*cols + i] = in_pixels[blockIdx.x*cols + i];
	}*/

	//for (int pixNum = 0; pixNum < rows * cols; ++pixNum) {

	for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		pixNum = blockIdx.x * cols + i;
		kernelSum = 0;
		redPixelVal = 0;

		for (int ki = 0; ki < KERNEL_SIZE; ++ki) {
			for (int kj = 0; kj < KERNEL_SIZE; ++kj) {

				//check edge cases, if within bounds, apply filter
				if (((pixNum + ((ki - ((KERNEL_SIZE - 1) / 2))*cols) + kj - ((KERNEL_SIZE - 1) / 2)) >= 0)
					&& ((pixNum + ((ki - ((KERNEL_SIZE - 1) / 2))*cols) + kj - ((KERNEL_SIZE - 1) / 2)) <= rows * cols - 1)
					&& (((pixNum % cols) + kj - ((KERNEL_SIZE - 1) / 2)) >= 0)
					&& (((pixNum % cols) + kj - ((KERNEL_SIZE - 1) / 2)) <= (cols - 1))) {

					pixNum_filter = pixNum + ((ki - ((KERNEL_SIZE - 1) / 2))*cols) + kj - ((KERNEL_SIZE - 1) / 2);
					pixNum_kernel = ki*KERNEL_SIZE+kj;

							if (threadIdx.x == 64 && blockIdx.x == 256) {
								printf("pn: %d\tpnf: %d\t pnk: %d\n", pixNum, pixNum_filter, pixNum_kernel);
							}

					redPixelVal += kernel[pixNum_kernel] * in_pixels[pixNum_filter];
					kernelSum   += kernel[pixNum_kernel];
				}
			}
		}
		out_pixels[pixNum] = redPixelVal / kernelSum;
	}

	//	redPixelVal = 0;
	//	kernelSum = 0;
	//}
}

void compute_intensity_gradient(unsigned char *in_pixels, int16_t *deltaX, int16_t *deltaY, int rows, int cols)
{
	unsigned int idx;

	// compute delta X ***************************
	// deltaX = f(x+1) - f(x-1)
	for (unsigned int i = 0; i < rows; ++i)
	{
		idx = cols * i; // current position X per line

		// gradient at the first pixel of each line
		// note: the edge,pix[idx-1] is NOT exsit
		deltaX[idx] = (int16_t)(in_pixels[idx + 1] - in_pixels[idx]);

		// gradients where NOT edge
		for (unsigned int j = 1; j < cols - 1; ++j)
		{
			idx++;
			deltaX[idx] = (int16_t)(in_pixels[idx + 1] - in_pixels[idx - 1]);
		}

		// gradient at the last pixel of each line
		idx++;
		deltaX[idx] = (int16_t)(in_pixels[idx] - in_pixels[idx - 1]);

	}

	// compute delta Y ***************************
	// deltaY = f(y+1) - f(y-1)
	for (unsigned int j = 0; j < cols; ++j)
	{
		idx = j;    // current Y position per column
		// gradient at the first pixel
		deltaY[idx] = (int16_t)(in_pixels[idx + cols] - in_pixels[idx]);

		// gradients for NOT edge pixels
		for (unsigned int i = 1; i < rows - 1; ++i)
		{
			idx += cols;
			deltaY[idx] = (int16_t)(in_pixels[idx + cols] - in_pixels[idx - cols]);
		}

		// gradient at the last pixel of each column
		idx += cols;
		deltaY[idx] = (int16_t)(in_pixels[idx] - in_pixels[idx - cols]);
	}
}

void magnitude(int16_t *deltaX, int16_t *deltaY, int16_t *img_magn, int rows, int cols)
{
	unsigned int idx;
	idx = 0;
	for (unsigned int i = 0; i < rows; ++i)
		for (unsigned int j = 0; j < cols; ++j, ++idx)
		{
			img_magn[idx] = sqrt((double)deltaX[idx] * deltaX[idx] + (double)deltaY[idx] * deltaY[idx]);
		}
}

void conversion_u8(short *in, unsigned char *out, int rows, int cols) {
	unsigned int idx = 0;

	for (unsigned int i = 0; i < rows; ++i)
		for (unsigned int j = 0; j < cols; ++j, ++idx) 
			out[idx] = (unsigned char) (in[idx] / 2);
		
}

void suppress_non_max(int16_t *mag, int16_t *deltaX, int16_t *deltaY, int16_t *nms, int rows, int cols)
{
	unsigned t = 0;
	float alpha;
	float mag1, mag2;
	const int16_t SUPPRESSED = 0;

	// put zero all boundaries of image
	// TOP edge line of the image
	for (unsigned j = 0; j < rows; ++j)
		nms[j] = 0;

	// BOTTOM edge line of image
	t = (cols - 1)*rows;
	for (unsigned j = 0; j < rows; ++j, ++t)
		nms[t] = 0;

	// LEFT & RIGHT edge line
	t = rows;
	for (unsigned i = 1; i < cols; ++i, t += rows)
	{
		nms[t] = 0;
		nms[t + rows - 1] = 0;
	}

	t = rows + 1;  // skip boundaries of image
	// start and stop 1 pixel inner pixels from boundaries
	for (unsigned i = 1; i < cols - 1; i++, t += 2)
	{
		for (unsigned j = 1; j < rows - 1; j++, t++)
		{
			// if magnitude = 0, no edge
			if (mag[t] == 0) nms[t] = SUPPRESSED;
			else {
				if (deltaX[t] >= 0)
				{
					if (deltaY[t] >= 0)  // dx >= 0, dy >= 0
					{
						if ((deltaX[t] - deltaY[t]) >= 0)       // direction 1 (SEE, South-East-East)
						{
							alpha = (float)deltaY[t] / deltaX[t];
							mag1 = (1 - alpha)*mag[t + 1] + alpha * mag[t + rows + 1];
							mag2 = (1 - alpha)*mag[t - 1] + alpha * mag[t - rows - 1];
						}
						else                                // direction 2 (SSE)
						{
							alpha = (float)deltaX[t] / deltaY[t];
							mag1 = (1 - alpha)*mag[t + rows] + alpha * mag[t + rows + 1];
							mag2 = (1 - alpha)*mag[t - rows] + alpha * mag[t - rows - 1];
						}
					}

					else  // dx >= 0, dy < 0
					{
						if ((deltaX[t] + deltaY[t]) >= 0)    // direction 8 (NEE)
						{
							alpha = (float)-deltaY[t] / deltaX[t];
							mag1 = (1 - alpha)*mag[t + 1] + alpha * mag[t - rows + 1];
							mag2 = (1 - alpha)*mag[t - 1] + alpha * mag[t + rows - 1];
						}
						else                                // direction 7 (NNE)
						{
							alpha = (float)deltaX[t] / -deltaY[t];
							mag1 = (1 - alpha)*mag[t + rows] + alpha * mag[t + rows - 1];
							mag2 = (1 - alpha)*mag[t - rows] + alpha * mag[t - rows + 1];
						}
					}
				}

				else
				{
					if (deltaY[t] >= 0) // dx < 0, dy >= 0
					{
						if ((deltaX[t] + deltaY[t]) >= 0)    // direction 3 (SSW)
						{
							alpha = (float)-deltaX[t] / deltaY[t];
							mag1 = (1 - alpha)*mag[t + rows] + alpha * mag[t + rows - 1];
							mag2 = (1 - alpha)*mag[t - rows] + alpha * mag[t - rows + 1];
						}
						else                                // direction 4 (SWW)
						{
							alpha = (float)deltaY[t] / -deltaX[t];
							mag1 = (1 - alpha)*mag[t - 1] + alpha * mag[t + rows - 1];
							mag2 = (1 - alpha)*mag[t + 1] + alpha * mag[t - rows + 1];
						}
					}

					else // dx < 0, dy < 0
					{
						if ((-deltaX[t] + deltaY[t]) >= 0)   // direction 5 (NWW)
						{
							alpha = (float)deltaY[t] / deltaX[t];
							mag1 = (1 - alpha)*mag[t - 1] + alpha * mag[t - rows - 1];
							mag2 = (1 - alpha)*mag[t + 1] + alpha * mag[t + rows + 1];
						}
						else                                // direction 6 (NNW)
						{
							alpha = (float)deltaX[t] / deltaY[t];
							mag1 = (1 - alpha)*mag[t - rows] + alpha * mag[t - rows - 1];
							mag2 = (1 - alpha)*mag[t + rows] + alpha * mag[t + rows + 1];
						}
					}
				}

				// non-maximal suppression
				// compare mag1, mag2 and mag[t]
				// if mag[t] is smaller than one of the neighbours then suppress it
				if ((mag[t] < mag1) || (mag[t] < mag2))
					nms[t] = SUPPRESSED;
				else
				{
					nms[t] = mag[t];
				}

			}
		}
	}
}

void trace_immed_neighbors(int16_t *out_pixels, int16_t *in_pixels, unsigned int idx, int16_t t_low, int rows, int cols)
{

	/* directions representing indices of neighbors */
	unsigned n, s, e, w;
	unsigned nw, ne, sw, se;

	/* get indices */
	n = idx - cols;
	nw = n - 1;
	ne = n + 1;
	s = idx + cols;
	sw = s - 1;
	se = s + 1;
	w = idx - 1;
	e = idx + 1;

	if ((in_pixels[nw] >= t_low) && (out_pixels[nw] != EDGE_V)) {
		out_pixels[nw] = EDGE_V;
	}
	if ((in_pixels[n] >= t_low) && (out_pixels[n] != EDGE_V)) {
		out_pixels[n] = EDGE_V;
	}
	if ((in_pixels[ne] >= t_low) && (out_pixels[ne] != EDGE_V)) {
		out_pixels[ne] = EDGE_V;
	}
	if ((in_pixels[w] >= t_low) && (out_pixels[w] != EDGE_V)) {
		out_pixels[w] = EDGE_V;
	}
	if ((in_pixels[e] >= t_low) && (out_pixels[e] != EDGE_V)) {
		out_pixels[e] = EDGE_V;
	}
	if ((in_pixels[sw] >= t_low) && (out_pixels[sw] != EDGE_V)) {
		out_pixels[sw] = EDGE_V;
	}
	if ((in_pixels[s] >= t_low) && (out_pixels[s] != EDGE_V)) {
		out_pixels[s] = EDGE_V;
	}
	if ((in_pixels[se] >= t_low) && (out_pixels[se] != EDGE_V)) {
		out_pixels[se] = EDGE_V;
	}
}

void apply_hysteresis(int16_t *out_pixels, int16_t *in_pixels, int16_t t_high, int16_t t_low, int rows, int cols)
{
	for (uint16_t i = 1; i < rows - 1; i++) {
		for (uint16_t j = 1; j < cols - 1; j++) {
			unsigned int t = (cols * i) + j;
			/* if our input is above the high threshold and the output hasn't already marked it as an edge */
			if (out_pixels[t] != EDGE_V) {
				if (in_pixels[t] > t_high) {
					/* mark as strong edge */
					out_pixels[t] = EDGE_V;

					/* check 8 immediately surrounding neighbors
					 * if any of the neighbors are above the low threshold, preserve edge */
					trace_immed_neighbors(out_pixels, in_pixels, t, t_low, rows, cols);
				}
				else {
					out_pixels[t] = NON_EDGE;
				}
			}
		}
	}
}

int main_function(int argc, char **argv)
{
	printf("Test \n");

	unsigned char *lena = NULL;
	unsigned char *lena_dev;
	unsigned int w = 512, h = 512;
	unsigned char *img_gauss;
	unsigned char *img_gauss_dev;
	short *img_deltaX;
	short *img_deltaY;
	short *img_magn;
	short *img_magn_nms;
	short *img_magn_hys;

	img_gauss = new unsigned char[w*h];
	img_deltaX = new short[w*h];
	img_deltaY = new short[w*h];
	img_magn = new short[w*h];
	img_magn_nms = new short[w*h];
	img_magn_hys = new short[w*h];

	double kernel[KERNEL_SIZE][KERNEL_SIZE];
	double *kernel_dev;
	populate_blur_kernel(kernel);

	// Pour sauvegarde
	unsigned char *img_deltaX_u8;
	unsigned char *img_deltaY_u8;
	unsigned char *img_magn_u8;
	unsigned char *img_magn_nms_u8;
	unsigned char *img_magn_hys_u8;
	
	// timing
	timespec start, stop, t1, t2, t3, t4;           // ticks
	double elapsedTotal_ms = 0;
	double elapsedDelta1_ms = 0;
	double elapsedDelta2_ms = 0;
	double elapsedDelta3_ms = 0;
	double elapsedDelta4_ms = 0;
	double elapsedDelta5_ms = 0;

	img_deltaX_u8 = new unsigned char[w*h];
	img_deltaY_u8 = new unsigned char[w*h];
	img_magn_u8 = new unsigned char[w*h];
	img_magn_nms_u8 = new unsigned char[w*h];
	img_magn_hys_u8 = new unsigned char[w*h];

	// if (!sdkLoadPGM("C:\\Users\\fmoranc\\Desktop\\A4\\ProjetVS_SETI\\Release\\lena.pgm", &lena, &w, &h)) fprintf(stderr, "Failed to load lena\n");
	if (!sdkLoadPGM("lena.pgm", &lena, &w, &h)) fprintf(stderr, "Failed to load lena\n");

	gpuErrchk( cudaMalloc((void**)&kernel_dev, KERNEL_SIZE*KERNEL_SIZE * sizeof(double)) );
	gpuErrchk( cudaMemcpy(kernel_dev, kernel, KERNEL_SIZE*KERNEL_SIZE * sizeof(double), cudaMemcpyHostToDevice) );

	gpuErrchk( cudaMalloc((void**)&lena_dev, w*h*sizeof(unsigned char)) );
	gpuErrchk( cudaMalloc((void**)&img_gauss_dev, w*h*sizeof(unsigned char)) );
	

	for (int iRepetition = 0; iRepetition < REPETITIONS; ++iRepetition) {
		start = timespec_now();

		// copy image to device - TODO
		gpuErrchk( cudaMemcpy(lena_dev, lena, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice) );

		apply_gaussian_filter_gpu<<<512, 64>>>(img_gauss_dev, lena_dev, kernel_dev, h, w);
		// apply_gaussian_filter(img_gauss, lena, kernel, h, w);
		gpuErrchk( cudaDeviceSynchronize() );

		// copy image from device to host
		gpuErrchk( cudaMemcpy(img_gauss, img_gauss_dev, w*h*sizeof(unsigned char), cudaMemcpyDeviceToHost) );

		gpuErrchk( cudaDeviceSynchronize() );

		t1 = timespec_now();

		compute_intensity_gradient(img_gauss, img_deltaX, img_deltaY, h, w);
		t2 = timespec_now();

		magnitude(img_deltaX, img_deltaY, img_magn, h, w);
		t3 = timespec_now();

		suppress_non_max(img_magn, img_deltaX, img_deltaY, img_magn_nms, h, w);
		t4 = timespec_now();

		apply_hysteresis(img_magn_hys, img_magn_nms, 10, 5, h, w);
		stop = timespec_now();

		elapsedDelta1_ms += timespec_to_ms(t1 - start);
		elapsedDelta2_ms += timespec_to_ms(t2 - t1);
		elapsedDelta3_ms += timespec_to_ms(t3 - t2);
		elapsedDelta4_ms += timespec_to_ms(t4 - t3);
		elapsedDelta5_ms += timespec_to_ms(stop - t4);
		elapsedTotal_ms += timespec_to_ms(stop - start);
	}

	fprintf(stderr, "Elapsed delta 1 ms: %f gauss\n", elapsedDelta1_ms / REPETITIONS);
	fprintf(stderr, "Elapsed delta 2 ms: %f gradient\n", elapsedDelta2_ms / REPETITIONS);
	fprintf(stderr, "Elapsed delta 3 ms: %f magnitude\n", elapsedDelta3_ms / REPETITIONS);
	fprintf(stderr, "Elapsed delta 4 ms: %f non max\n", elapsedDelta4_ms / REPETITIONS);
	fprintf(stderr, "Elapsed delta 5 ms: %f hysteresis\n", elapsedDelta5_ms / REPETITIONS);
	fprintf(stderr, "Elapsed total ms: %f\n", elapsedTotal_ms/REPETITIONS);


	conversion_u8(img_deltaX, img_deltaX_u8, h, w);
	conversion_u8(img_deltaY, img_deltaY_u8, h, w);
	conversion_u8(img_magn, img_magn_u8, h, w);
	conversion_u8(img_magn_nms, img_magn_nms_u8, h, w);
	conversion_u8(img_magn_hys, img_magn_hys_u8, h, w);


	
	// Sauvegarde l'image
	sdkSavePGM("out/img_lena.pgm", lena, w, h);
	sdkSavePGM("out/img_gauss.pgm", img_gauss, w, h);
	sdkSavePGM("out/img_deltaX.pgm", img_deltaX_u8, w, h);
	sdkSavePGM("out/img_deltaY.pgm", img_deltaY_u8, w, h);
	sdkSavePGM("out/img_magn_nms.pgm", img_magn_nms_u8, w, h);
	sdkSavePGM("out/img_magn_hys.pgm", img_magn_hys_u8, w, h); 

	return 0;
}
