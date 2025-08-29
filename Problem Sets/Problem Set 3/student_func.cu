#include <float.h>

/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__global__
void maximum_luminance(const float* const d_logLuminance,
	   	       float* const d_min_logLum,
		       float* const d_max_logLum,
		       const size_t numRows,
		       const size_t numCols)
{
    extern __shared__ float s[];
    int2 pos_2d = make_int2(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y);
    int  pos_1d = pos_2d.y * numCols + pos_2d.x;
    int  block_size = blockDim.x * blockDim.y;

    float *s_min = s;
    float *s_max = s + block_size;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // Cooperatively read the block of data from global to shared memory
    if (pos_2d.x >= numCols || pos_2d.y >= numRows) {
       s_min[tid] = FLT_MAX;
       s_max[tid] = -FLT_MAX;
    } else {
      s_min[tid] = s_max[tid] = d_logLuminance[pos_1d];
    }
    __syncthreads();

    for (int step = 1; step < block_size; step *= 2) {
    	if ((tid % (step * 2) == 0) && (tid + step < block_size)) {
	   s_min[tid] = min(s_min[tid], s_min[tid + step]);
	   s_max[tid] = max(s_max[tid], s_max[tid + step]);
	}
	__syncthreads();
    }
    if (tid == 0) {
       d_min_logLum[gridDim.x * blockIdx.y + blockIdx.x] = s_min[tid];
       d_max_logLum[gridDim.x * blockIdx.y + blockIdx.x] = s_max[tid];
//       printf("(%f, %f)", s_min[tid], s_max[tid]);
    }
}

// Given two input arrays of floats, find the min of the first one,
// and the max of the second one. Assume a 1D thread block.
__global__
void maximum_reduce(const float* const d_data_for_min,
     		    const float* const d_data_for_max,
	            float* const d_data_min,
	            float* const d_data_max,
	            const size_t data_size)
{
    extern __shared__ float s[];
    int  block_size = blockDim.x;

    assert(data_size <= block_size);

    float *s_min = s;
    float *s_max = s + data_size;

    int tid = threadIdx.x;
    // Cooperatively read the block of data from global to shared memory
    if (tid >= data_size) {
       s_min[tid] = FLT_MAX;
       s_max[tid] = -FLT_MAX;
    } else {
      s_min[tid] = d_data_for_min[tid];
      s_max[tid] = d_data_for_max[tid];
    }
    __syncthreads();

    for (int step = 1; step < data_size; step *= 2) {
    	if ((tid % (step * 2) == 0) && (tid + step < data_size)) {
	   s_min[tid] = min(s_min[tid], s_min[tid + step]);
	   s_max[tid] = max(s_max[tid], s_max[tid + step]);
	}
	__syncthreads();
    }
    
    if (tid == 0) {
       *d_data_min = s_min[tid];
       *d_data_max = s_max[tid];
//       printf("d_data_min = %f, d_data_max = %f", *d_data_min, *d_data_max);
    }
}

__global__
void compute_histogram_array(const float* const d_logLuminance,
  		       unsigned int* const d_histograms,
		       const size_t numRows,
		       const size_t numCols,
		       const float lumMin,
		       const float lumRange,
		       const size_t numBins)
{    
    // x and y of the pixel
    int2 pos_2d = make_int2(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y);
    
    // position of the pixel in the image array.
    int  pos_1d = pos_2d.y * numCols + pos_2d.x;

    int num_blocks = gridDim.x * gridDim.y;

    // Position of the thread block within the grid array. 
    int block_1d = gridDim.x * blockIdx.y + blockIdx.x;

    if (pos_2d.x < numCols && pos_2d.y < numRows) {
       unsigned int bin = (unsigned int)((d_logLuminance[pos_1d] - lumMin) * (numBins - 1) / lumRange);
       atomicAdd(&d_histograms[bin * num_blocks + block_1d], 1);
    }
}


__global__
void compute_histogram_array_shared(const float* const d_logLuminance,
  		       unsigned int* const d_histograms,
		       const size_t numRows,
		       const size_t numCols,
		       const float lumMin,
		       const float lumRange,
		       const size_t numBins)
{
    extern __shared__ unsigned int s_histogram[];
    
    // x and y of the pixel
    int2 pos_2d = make_int2(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y);
    
    // position of the pixel in the image array.
    int  pos_1d = pos_2d.y * numCols + pos_2d.x;

    // Size of the thread block
    int  block_size = blockDim.x * blockDim.y;

    // Position of the thread block within the grid array. 
    int block_1d = gridDim.x * blockIdx.y + blockIdx.x;

    // position of the thread within the block thread array.
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // initialize the shared_memory
    for (unsigned int bin = tid; bin < numBins; bin += block_size) 
    	s_histogram[bin] = 0;
    __syncthreads();
    
    assert(lumRange != 0.0);
    if (pos_2d.x < numCols && pos_2d.y < numRows) {
       unsigned int bin = (unsigned int)((d_logLuminance[pos_1d] - lumMin) * (numBins - 1) / lumRange);
       atomicAdd(&s_histogram[bin], 1);
    }
    __syncthreads();

    int num_blocks = gridDim.x * gridDim.y;

    // write the local s_histogram into columns of the global array of histograms, d_histograms
    // the d_histograms has dimensions numBins x num_blocks
    for (int row = tid; row < numBins; row += block_size) {
    	d_histograms[row * num_blocks + block_1d] = s_histogram[row];
    }	
    __syncthreads();
}

__global__
void compute_histogram(unsigned int* const d_histograms,
		       unsigned int* const d_histogram,
		       const size_t numBins,
		       const size_t items_per_bin)
{
    extern __shared__ unsigned int s_histogram[];

    // Size of the thread block
    int  block_size = blockDim.x; //  * blockDim.y;
    assert(block_size >= items_per_bin);

    // Position of the thread block within the grid array. 
    int block_1d = blockIdx.x; // + gridDim.x * blockIdx.y;

    // position of the thread within the block thread array.
    int tid = threadIdx.x; // + threadIdx.y * blockDim.x;
    
    int num_blocks = gridDim.x; // * gridDim.y;

    // Now, each row of the global array of histograms corresponds to a single bin.
    // Now, each thread block needs to reduce rows (bin numbers) that are a multiple of
    // num_blocks.
    // We will have the threads in this thread_block reduce
    // a single row at a time.
    
    for (int row = block_1d; row < numBins; row += num_blocks) {

        // First, read the row-wise bin-arrays to the shared_memory
	if (tid < items_per_bin) {
            s_histogram[tid] = d_histograms[row * items_per_bin + tid];
	}	
    	__syncthreads();

   	for (int step = 1; step < items_per_bin; step *= 2) {
    	     if ((tid % (step * 2) == 0) && (tid + step < items_per_bin)) {
	   	s_histogram[tid] += s_histogram[tid + step];
		//printf("h ");
	     }		
	     __syncthreads();
    	}
	if (tid == 0)
	   d_histogram[row] = s_histogram[tid];
    }
}


__global__
void compute_cdf(const unsigned int* const d_histogram,
     		 unsigned int* d_cdf,
		 const size_t numItems)
{

    // Size of the thread block
    int  block_size = blockDim.x;
    assert(block_size >= numItems);

    // position of the thread within the block thread array.
    int tid = threadIdx.x; 
    
   // First, read the row-wise bin-arrays to the shared_memory
   if (tid < numItems)
      d_cdf[tid] = d_histogram[tid];

   // reverse tid
   int rid = numItems - 1 - tid;

   int max_step = 0;
   // The reduction phase of Blelloch algorithm
   for (int step = 1; step < numItems; step *= 2) {
       max_step = step;
       if ((rid % (step * 2) == 0) && (rid + step < numItems)) {
       	  d_cdf[tid] += d_cdf[tid - step];
       }
       __syncthreads();
   }

   // The sweep phase of Blelloch algorithm
   if (rid == 0)
      d_cdf[tid] = 0;
   int temp;
   for (int step = max_step; step >= 1; step /= 2) {
       if ((rid % (step * 2) == 0) && (rid + step < numItems)) {
          temp = d_cdf[tid];
       	  d_cdf[tid] += d_cdf[tid - step];
	  d_cdf[tid - step] = temp;
       }  
       __syncthreads();
   }
	
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{

// 1) find the minimum and maximum value in the input logLuminance channel
//    store in min_logLum and max_logLum

//  Set reasonable block size (i.e., number of threads per block)
//  int y_dim = (int)(32 * sqrt((float)numRows / numCols)); // integer divided by integer makes integer!
//  int x_dim = (int)(32 * 32 / y_dim);
//  const dim3 blockSize(32, 32);
//  dim3 blockSize(x_dim, y_dim);

  // Because numBins is 1024, we choose the block size to be 32*32, so that the threads
  // in a block would first cooperatively compute a histogram in shared memory, then
  // write it to global memory, then read the columns of the histogram back into
  // shared memory, then do a reduce-sum on them, and finally write the reduced histogram
  // to global memory.
  dim3 blockSize(32, 32);

  // Compute correct grid size (i.e., number of blocks per kernel launch)
  // from the image size and and block size.
  dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x, (numRows + blockSize.y - 1) / blockSize.y);

  printf("numBins = %ld, numCols = %ld, numRows = %ld\n", numBins, numCols, numRows);
  printf("blockDim.y = %u, blockDim.x = %u \n", blockSize.y, blockSize.x);
  printf("gridSize.x = %u, gridSize.y = %u \n", gridSize.x, gridSize.y);

  float *d_min_logLum;
  float *d_max_logLum;
  int num_blocks = gridSize.x * gridSize.y;

  checkCudaErrors(cudaMalloc(&d_min_logLum, sizeof(float) * num_blocks));
  checkCudaErrors(cudaMalloc(&d_max_logLum, sizeof(float) * num_blocks));

  int shared_memory_size = 2 * blockSize.x * blockSize.y * sizeof(float);
  maximum_luminance<<<gridSize, blockSize, shared_memory_size>>>(d_logLuminance,
								 d_min_logLum,
								 d_max_logLum,
			     					 numRows,
			     					 numCols);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  gridSize = dim3(1);
  blockSize = dim3(num_blocks);

  float *d_logLum;
  float h_logLum[2];
  
  checkCudaErrors(cudaMalloc(&d_logLum, sizeof(float) * 2));

  maximum_reduce<<<gridSize, blockSize, num_blocks * 2 * sizeof(float)>>>(d_min_logLum, d_max_logLum,
	   		  &d_logLum[0],
			  &d_logLum[1],
			  num_blocks);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(h_logLum, d_logLum, 2 * sizeof(float), cudaMemcpyDeviceToHost));
  min_logLum = h_logLum[0];
  max_logLum = h_logLum[1];
//  printf("min_logLum = %f, max_logLum = %f", min_logLum, max_logLum);

  checkCudaErrors(cudaFree(d_min_logLum));
  checkCudaErrors(cudaFree(d_max_logLum));
  checkCudaErrors(cudaFree(d_logLum));

  // 2) subtract them to find the range
  float lumRange = max_logLum - min_logLum;
  assert(lumRange != 0.0);

  printf("maxBin = %u\n", (unsigned int)((max_logLum - min_logLum) * (numBins - 1) / lumRange)); 
  
  // 3) generate a histogram of all the values in the logLuminance channel using
  //    the formula: bin = (lum[i] - lumMin) / lumRange * numBins

  blockSize = dim3(32, 32);
  gridSize = dim3((numCols + blockSize.x - 1) / blockSize.x, (numRows + blockSize.y - 1) / blockSize.y);
  num_blocks = gridSize.x * gridSize.y;

  unsigned int *d_histograms;
  unsigned int *d_histogram;
  checkCudaErrors(cudaMalloc(&d_histograms, sizeof(unsigned int) * num_blocks * numBins));
  checkCudaErrors(cudaMalloc(&d_histogram, sizeof(unsigned int) * numBins));

  checkCudaErrors(cudaMemset(d_histogram, 0, numBins * sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_histograms, 0, numBins * num_blocks * sizeof(unsigned int)));
  
  compute_histogram_array_shared<<<gridSize, blockSize, sizeof(unsigned int) * numBins>>>(d_logLuminance,
  				d_histograms, numRows, numCols, min_logLum, lumRange, numBins);

//  compute_histogram_array<<<gridSize, blockSize>>>(d_logLuminance,
//  				d_histograms, numRows, numCols, min_logLum, lumRange, numBins);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

/*
  unsigned int h_histograms[numBins * num_blocks];
  checkCudaErrors(cudaMemcpy(h_histograms, d_histograms, sizeof(unsigned int) * numBins * num_blocks, cudaMemcpyDeviceToHost));
  
  printf("\n\n");
  int sum = 0;
  for (unsigned int i = 0; i < numBins; i++) {
      sum += h_histograms[i * num_blocks];
      printf("%u ", h_histograms[i * num_blocks]);
  }

  printf("Total of first block = %u\n\n", sum);
*/


  blockSize = dim3(gridSize.x * gridSize.y);
  gridSize = dim3(numBins);

  assert(blockSize.x <= 1024); 

  compute_histogram<<<gridSize, blockSize, sizeof(unsigned int) * blockSize.x>>>(
  				d_histograms, d_histogram, numBins, blockSize.x);
 
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

/*
  unsigned int h_histogram[numBins];
  checkCudaErrors(cudaMemcpy(h_histogram, d_histogram, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));
  
  for (unsigned int i = 0; i < numBins; i++)
      printf("%u ", h_histogram[i]);
  printf("\n");
*/ 
  checkCudaErrors(cudaFree(d_histograms));
  
  /*
   4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  blockSize = dim3(1024);
  gridSize = dim3(1 + (numBins - 1) / blockSize.x);
  assert(numBins <= 1024);
  compute_cdf<<<gridSize, blockSize, sizeof(unsigned int) * blockSize.x>>>(
  				d_histogram, d_cdf, numBins);
 
  checkCudaErrors(cudaFree(d_histogram));
/*
  unsigned int h_cdf[numBins];
  checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));
  
  for (unsigned int i = 0; i < numBins; i++)
      printf("%u ", h_cdf[i]);
  printf("\n");
*/ 
}
