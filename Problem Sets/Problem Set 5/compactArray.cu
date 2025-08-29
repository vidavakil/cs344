//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

// This implementation of radix sort uses shared memory

static const unsigned int numThreadsPerSM = 1024;

__global__
void computeLowestBlockPrefixSum(unsigned int* const d_inputVals,
                      unsigned int* const d_onesPrefixSum,
                      const size_t        numElems,
                      bool                isInitialized,
                      bool                ignoreFirstElement)
{
    extern __shared__ unsigned int s_onesPrefixSum[];

    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int localBlockSize = (((blockIdx.x + 1) * blockDim.x) <= numElems) ? blockDim.x : (numElems % blockDim.x);

    if (not isInitialized) {
        if (tid < numElems) {
            s_onesPrefixSum[threadIdx.x] = (d_inputVals[tid] != 0) ? 1 : 0;
            if (tid == 0 and ignoreFirstElement)
                s_onesPrefixSum[threadIdx.x] = 1;
        }
        __syncthreads();
    } else {
        if (tid < numElems) {
            s_onesPrefixSum[threadIdx.x] = d_onesPrefixSum[tid];
        } 
        __syncthreads();   
    }

    // reverse tid
    int rid = localBlockSize - 1 - threadIdx.x;

    unsigned int max_step = 0;
    // The reduction phase of Blelloch algorithm
    for (unsigned int step = 1; step < localBlockSize; step *= 2) {
        max_step = step;
        if (rid >= 0 && (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize)) {
            s_onesPrefixSum[threadIdx.x] += s_onesPrefixSum[threadIdx.x - step];
        }
        __syncthreads();
    }

    unsigned int totalOnes = 0;

    // The sweep phase of Blelloch algorithm

    // Before overriding the last element of the counts array,
    // store it away.
    if (rid == 0) {
        totalOnes = s_onesPrefixSum[threadIdx.x];
        s_onesPrefixSum[threadIdx.x] = 0;
    }

    __syncthreads();

    unsigned int temp;
    for (unsigned int step = max_step; step >= 1; step /= 2) {
        if (rid >= 0 && (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize)) {
            temp = s_onesPrefixSum[threadIdx.x];
            s_onesPrefixSum[threadIdx.x] += s_onesPrefixSum[threadIdx.x - step];
            s_onesPrefixSum[threadIdx.x - step] = temp;
        }  
        __syncthreads();
    }

    // Store the totals in the first element of the array. These special
    // elements constitute the histogram of zeros and ones in each thread
    // block.
    if (rid == 0) 
        s_onesPrefixSum[0] = totalOnes;
    __syncthreads();

    if (tid < numElems) {
        d_onesPrefixSum[tid] = s_onesPrefixSum[threadIdx.x];
    }
}                      


__global__
void computeStridedPrefixSum(unsigned int* const d_onesPrefixSum, 
                      const size_t numElems,
                      unsigned int stride) 
{
    extern __shared__ unsigned int s_onesPrefixSum[];

    unsigned int localBlockSize = (((blockIdx.x + 1) * blockDim.x) <= numElems)
                                ? blockDim.x 
                                : (numElems % blockDim.x);
    unsigned int pos_1d = (blockDim.x * blockIdx.x + threadIdx.x) * stride;

    if (threadIdx.x < localBlockSize) {
        s_onesPrefixSum[threadIdx.x] = d_onesPrefixSum[pos_1d];
    }
    __syncthreads();

    // reverse tid
    int rid = localBlockSize - 1 - threadIdx.x;
 
    unsigned int max_step = 0;
    // The reduction phase of Blelloch algorithm
    for (unsigned int step = 1; step < localBlockSize; step *= 2) {
        max_step = step;
        if (rid >= 0 && (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize))
            s_onesPrefixSum[threadIdx.x] += s_onesPrefixSum[threadIdx.x - step];
        __syncthreads();
    }

    // The sweep phase of Blelloch algorithm

    // Before overriding the last element of the counts array,
    // store it away.
    unsigned int totalOnes = 0;
    if (rid == 0) {
        totalOnes = s_onesPrefixSum[threadIdx.x];
        s_onesPrefixSum[threadIdx.x] = 0;
    }

    __syncthreads();

    unsigned int temp;
    for (unsigned int step = max_step; step >= 1; step /= 2) {
        if (rid >= 0 && (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize)) {
            temp = s_onesPrefixSum[threadIdx.x];
            s_onesPrefixSum[threadIdx.x] += s_onesPrefixSum[threadIdx.x - step];
            s_onesPrefixSum[threadIdx.x - step] = temp;
        }  
        __syncthreads();
    }

    // Store totalZeros in the first element of the arrays
    if (rid == 0)
        s_onesPrefixSum[0] = totalOnes;
    __syncthreads();

    if (threadIdx.x < localBlockSize) {
        d_onesPrefixSum[pos_1d] = s_onesPrefixSum[threadIdx.x];        
    }    
}

__global__
void moveElements(unsigned int* const d_inputVals,
                  unsigned int* const d_outputVals, 
                  unsigned int* const d_onesPrefixSum, 
                  const size_t numElems,
                  const unsigned int stride,
                  const unsigned int strideLevels,
                  const unsigned int startStridalAddress, // stride ^ strideLevels
                  bool               ignoreFirstElement)
{
    unsigned int pos_1d = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int addressTobeWrittenTo = 0;
    unsigned int stridalAddress = startStridalAddress;    
    const unsigned int* lookUpAddress = d_onesPrefixSum;

    unsigned int  roundedAddress = 0;    

    assert(blockDim.x == stride);
    assert(stridalAddress != 0);
    
    if (pos_1d > 0 and pos_1d < numElems and stridalAddress >= 1) {
        while (pos_1d % stridalAddress != 0) {
            roundedAddress =  stridalAddress * (pos_1d / stridalAddress);
            if (roundedAddress != 0 && roundedAddress % (stridalAddress * stride) != 0)
                addressTobeWrittenTo += lookUpAddress[roundedAddress];
            stridalAddress /= stride;
            assert(addressTobeWrittenTo < numElems);
        }
        assert(addressTobeWrittenTo < numElems);
        addressTobeWrittenTo += lookUpAddress[pos_1d];
    }
    if (pos_1d < numElems && (d_inputVals[pos_1d] != 0 || (pos_1d == 0 and ignoreFirstElement)))
        d_outputVals[addressTobeWrittenTo] = d_inputVals[pos_1d];   
}

// Because the size of input may be too big, a single thread block may not be
// able to compute the scatter addresses for each step of the compaction or
// expansion. Thus, we break up the task over many thread blocks. A first 
// kernel is called to have the thread blocks compute local exclusive prefix
// sums at any given position. They will write the total sum in the 0th
// location of the prefix sum that is always 0. A second kernel is recursively
// called to perform prefix-sums at ever increasing granularity. A third kernel
// is then called that will use the prefix sums, and the offsets stored in
// multiples of each level of granularity to compute the scatter addresses, and
// then performs the writing.
void compactArray(unsigned int* const d_inputVals,   // some are zeros
                  unsigned int*      &d_outputVals,  // compacted array to be allocated
                  unsigned int       &compactedSize, // size of the compacted array
                  const size_t        numElems,
                  bool                ignoreFirstElement)
{ 
    dim3 blockDim(numThreadsPerSM);
    dim3 gridDim((numElems + blockDim.x - 1) / blockDim.x);
    dim3 gridOverGridDim((gridDim.x + blockDim.x - 1) / blockDim.x);

/*
    printf("compactArray: numElems = %lu, gridDim.x = %u, blockDim.x = %u, gridOverGridDim = %u\n", 
           numElems, gridDim.x, blockDim.x, gridOverGridDim.x);
*/

    // Allocate memory for the scatter addresses
    unsigned int *d_onesHistogram;

    checkCudaErrors(cudaMalloc(&d_onesHistogram, sizeof(unsigned int) * numElems));

    unsigned int stride = numThreadsPerSM;
    bool alreadyInitialized = false;

    computeLowestBlockPrefixSum<<<gridDim, blockDim, sizeof(unsigned int) * numThreadsPerSM>>>(
                    d_inputVals, d_onesHistogram, numElems, alreadyInitialized, ignoreFirstElement);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    unsigned int strideLevels = 0;
    unsigned int currentStride = stride;
    unsigned int numBlocks = 0;

    // Step 2: Compute prefix sum (CDF) of the histogram
    while (numElems > currentStride) {
        gridOverGridDim.x = (numElems + currentStride * stride - 1) / (currentStride * stride);
        numBlocks = (numElems + currentStride - 1) / currentStride;
        computeStridedPrefixSum<<<gridOverGridDim, blockDim, sizeof(unsigned int) * numThreadsPerSM>>>(
            d_onesHistogram, numBlocks, currentStride);
        strideLevels++;
        currentStride *= stride;
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());        
    }

    checkCudaErrors(cudaMemcpy(&compactedSize, d_onesHistogram, 
                               sizeof(unsigned int), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMalloc(&d_outputVals, sizeof(int) * compactedSize));

    // Step 3: Combine the results of the first two kernels to compute the location each
    // element must be moved to and then move them.
    moveElements<<<gridDim, blockDim>>>(d_inputVals,
                                        d_outputVals,
                                        d_onesHistogram, 
                                        numElems, 
                                        stride, 
                                        strideLevels,
                                        currentStride / stride,
                                        ignoreFirstElement);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_onesHistogram));
}


void expandArray(      unsigned int* const d_compactedBinNumBlocksPrefixSum,
                       unsigned int       &numExpandedBins,  // to be overwriten
                 const unsigned int        numCompactedBins, // compactedSize
                 const unsigned int        blockSize,
                       unsigned int        stride,
                       unsigned int       &stridalAddress) 
{ 
    assert(blockSize == numThreadsPerSM);

    dim3 blockDim(blockSize);
    dim3 gridDim((numCompactedBins + blockDim.x - 1) / blockDim.x);
    dim3 gridOverGridDim((gridDim.x + blockDim.x - 1) / blockDim.x);

/*
    printf("expandArray: numCompactedBins = %u, gridDim.x = %u, blockDim.x = %u, gridOverGridDim = %u\n\n", 
           numCompactedBins, gridDim.x, blockDim.x, gridOverGridDim.x);
*/
    assert(stride == blockSize);

    computeLowestBlockPrefixSum<<<gridDim, blockDim, sizeof(unsigned int) * numThreadsPerSM>>>(
        NULL, d_compactedBinNumBlocksPrefixSum, numCompactedBins, true, false);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    unsigned int strideLevels = 0;
    unsigned int currentStride = stride;
    unsigned int numBlocks = 0;

    // Step 2: Compute prefix sum (CDF) of the histogram
    while (numCompactedBins > currentStride) {
        gridOverGridDim.x = (numCompactedBins + currentStride * stride - 1) / (currentStride * stride);
        numBlocks = (numCompactedBins + currentStride - 1) / currentStride;
        computeStridedPrefixSum<<<gridOverGridDim, blockDim, sizeof(unsigned int) * numThreadsPerSM>>>(
            d_compactedBinNumBlocksPrefixSum, numBlocks, currentStride);
        strideLevels++;
        currentStride *= stride;
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }

    stridalAddress = currentStride/stride;

    // Store in the output argument numExpandedBins 
    checkCudaErrors(cudaMemcpy(&numExpandedBins, d_compactedBinNumBlocksPrefixSum, 
                               sizeof(unsigned int), cudaMemcpyDeviceToHost));

//    printf("numExpandedBins = %u\n", numExpandedBins);
}
