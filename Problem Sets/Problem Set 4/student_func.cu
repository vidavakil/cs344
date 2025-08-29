//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>

static const unsigned int numThreadsPerSM = 1024;

__global__
void computeLowestBlockPrefixSum(      unsigned int* const d_inputVals,
                                       unsigned int* const d_zerosPrefixSum,
                                       unsigned int* const d_onesPrefixSum,
                                 const size_t              numElems,
                                       unsigned int        bitPosition)
{
    extern __shared__ unsigned int s_prefixSum[];

    unsigned int* s_zerosPrefixSum = s_prefixSum;
    unsigned int* s_onesPrefixSum = &s_prefixSum[blockDim.x];

    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int localBlockSize = (((blockIdx.x + 1) * blockDim.x) <= numElems)
                                 ? blockDim.x 
                                 : (numElems % blockDim.x);                                 

    if (tid < numElems) 
    {
        unsigned int bitPositionIsZero = ((d_inputVals[tid] & (1 << bitPosition)) == 0) ? 1 : 0;
        s_zerosPrefixSum[threadIdx.x] = bitPositionIsZero;
        s_onesPrefixSum[threadIdx.x] = 1 - bitPositionIsZero;
    }
    __syncthreads();

    // reverse tid
    int rid = localBlockSize - 1 - threadIdx.x;

    unsigned int max_step = 0;
    // The reduction phase of Blelloch algorithm
    for (unsigned int step = 1; step < localBlockSize; step *= 2) {
        max_step = step;
        if (rid >= 0 && (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize)) {
            s_zerosPrefixSum[threadIdx.x] += s_zerosPrefixSum[threadIdx.x - step];
            s_onesPrefixSum[threadIdx.x] += s_onesPrefixSum[threadIdx.x - step];
        }
        __syncthreads();
    }

    unsigned int totalZeros = 0, totalOnes = 0;

    // The sweep phase of Blelloch algorithm

    // Before overriding the last element of the counts array,
    // store it away.
    if (rid == 0) {
        totalZeros = s_zerosPrefixSum[threadIdx.x];
        s_zerosPrefixSum[threadIdx.x] = 0;
        totalOnes = s_onesPrefixSum[threadIdx.x];
        s_onesPrefixSum[threadIdx.x] = 0;
    }
    __syncthreads();
    
    unsigned int temp;
    for (unsigned int step = max_step; step >= 1; step /= 2) {
        if (rid >= 0 && (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize)) {
            temp = s_zerosPrefixSum[threadIdx.x];
            s_zerosPrefixSum[threadIdx.x] += s_zerosPrefixSum[threadIdx.x - step];
            s_zerosPrefixSum[threadIdx.x - step] = temp;

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
    {
        s_zerosPrefixSum[0] = totalZeros;
        s_onesPrefixSum[0] = totalOnes;
    }
    __syncthreads();

    if (tid < numElems) 
    {
        d_zerosPrefixSum[tid] = s_zerosPrefixSum[threadIdx.x];
        d_onesPrefixSum[tid] = s_onesPrefixSum[threadIdx.x];
    }
}                      


__global__
void computeStridedPrefixSum(      unsigned int* const d_zerosPrefixSum, 
                                   unsigned int* const d_onesPrefixSum, 
                             const size_t              numElems,
                                   unsigned int        stride) 
{
    extern __shared__ unsigned int s_prefixSum[];

    unsigned int* s_zerosPrefixSum = s_prefixSum;
    unsigned int* s_onesPrefixSum = &s_prefixSum[blockDim.x];

    unsigned int localBlockSize = (((blockIdx.x + 1) * blockDim.x) <= numElems) 
                                  ? blockDim.x 
                                  : (numElems % blockDim.x);

    unsigned int pos_1d = (blockDim.x * blockIdx.x + threadIdx.x) * stride;

    if (threadIdx.x < localBlockSize) {
        s_zerosPrefixSum[threadIdx.x] = d_zerosPrefixSum[pos_1d];
        s_onesPrefixSum[threadIdx.x] = d_onesPrefixSum[pos_1d];
    }
    __syncthreads();

    // reverse tid
    int rid = localBlockSize - 1 - threadIdx.x;  // could be negative
 
    unsigned int max_step = 0;
    // The reduction phase of Blelloch algorithm
    for (unsigned int step = 1; step < localBlockSize; step *= 2) {
        max_step = step;
        if (rid >= 0 && (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize)) {
            s_zerosPrefixSum[threadIdx.x] += s_zerosPrefixSum[threadIdx.x - step];
            s_onesPrefixSum[threadIdx.x] += s_onesPrefixSum[threadIdx.x - step];
        }
        __syncthreads();
    }

    // The sweep phase of Blelloch algorithm

    // Before overriding the last element of the counts array,
    // store it away.
    unsigned int totalZeros = 0;
    unsigned int totalOnes = 0;
    if (rid == 0) {
        totalZeros = s_zerosPrefixSum[threadIdx.x];
        totalOnes = s_onesPrefixSum[threadIdx.x];
        s_zerosPrefixSum[threadIdx.x] = 0;
        s_onesPrefixSum[threadIdx.x] = 0;
    }

    __syncthreads();

    unsigned int temp;
    for (unsigned int step = max_step; step >= 1; step /= 2) {
        if (rid >= 0 && (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize)) {
            temp = s_zerosPrefixSum[threadIdx.x];
            s_zerosPrefixSum[threadIdx.x] += s_zerosPrefixSum[threadIdx.x - step];
            s_zerosPrefixSum[threadIdx.x - step] = temp;

            temp = s_onesPrefixSum[threadIdx.x];
            s_onesPrefixSum[threadIdx.x] += s_onesPrefixSum[threadIdx.x - step];
            s_onesPrefixSum[threadIdx.x - step] = temp;
        }  
        __syncthreads();
    }

    // Store totalZeros in the first element of the arrays
    if (rid == 0) {
       s_zerosPrefixSum[0] = totalZeros;
       s_onesPrefixSum[0] = totalOnes;
    }
    __syncthreads();

    if (threadIdx.x < localBlockSize) {
        d_zerosPrefixSum[pos_1d] = s_zerosPrefixSum[threadIdx.x];
        d_onesPrefixSum[pos_1d] = s_onesPrefixSum[threadIdx.x];        
    }
}

__global__
void moveElements(const unsigned int* const d_inputVals,
                  const unsigned int* const d_inputPos,
                        unsigned int* const d_outputVals, 
                        unsigned int* const d_outputPos,
                  const unsigned int* const d_zerosPrefixSum, 
                  const unsigned int* const d_onesPrefixSum, 
                  const size_t              numElems, 
                  const size_t              bitPosition,
                  const unsigned int        stride,
                  const unsigned int        strideLevels,
                  const unsigned int        startStridalAddress) // stride ^ strideLevels
{
    unsigned int pos_1d = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int bitPositionIsZero = ((d_inputVals[pos_1d] & (1 << bitPosition)) == 0);
    unsigned int addressTobeWrittenTo = 0;
    unsigned int stridalAddress = startStridalAddress;
    const unsigned int* lookUpAddress = d_zerosPrefixSum;

    unsigned int  roundedAddress = 0;

    assert(blockDim.x == stride);
    assert(stridalAddress != 0);

    if (not bitPositionIsZero) {
        addressTobeWrittenTo = d_zerosPrefixSum[0];
        lookUpAddress = d_onesPrefixSum;
    }

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
    if (pos_1d < numElems) {
        d_outputVals[addressTobeWrittenTo] = d_inputVals[pos_1d];
        d_outputPos[addressTobeWrittenTo] = d_inputPos[pos_1d];
    }
}

// Because the size of input may be too big, a single thread block may not
// be able to compute the scatter addresses for each step of the radix sort.
// Thus, we break up the task over many thread blocks. A first kernel is
// called to have the thread blocks compute local exclusive prefix sums for both 
// 0 and 1 occurances at a given bit position. They will write the prefix sum
// results in the output array passed to them. They will write the total
// number of 0's or 1's in the 0th location of the prefix sum that is always
// 0. A second kernel is called to perform a prefix-sum over the two 
// histograms of 0's and 1's, formed by the first elements of the original
// prefix sums. A third kernel is then called that will use the original 
// prefix sums, and the offsets stored in their 0th location to computer
// the location where each value/pos in their chunk has to be written into
// the ouput arrays, and then performs the writing.
// The above steps are repeated for each bit position, and each time the
// input and output buffers are swapped.
void radixSort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems,
               unsigned int radixLow,
               unsigned int radixHigh)
{ 
    assert (radixHigh >= radixLow);
    unsigned int numIterations = radixHigh - radixLow + 1;

    dim3 blockDim(numThreadsPerSM);
    dim3 gridDim((numElems + blockDim.x - 1) / blockDim.x);
    dim3 gridOverGridDim((gridDim.x + blockDim.x - 1) / blockDim.x);

/*
    printf("radixSort: numElems = %lu, gridDim.x = %u, blockDim.x = %u, gridOverGridDim = %u\n", 
           numElems, gridDim.x, blockDim.x, gridOverGridDim.x);
*/

    // For now, let's only support small enough data sizes
    // assert(gridOverGridDim.x == 1);

    // Allocate memory for the scatter addresses for zeros and ones
    unsigned int *d_zerosHistogram, *d_onesHistogram;

    checkCudaErrors(cudaMalloc(&d_zerosHistogram, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMalloc(&d_onesHistogram, sizeof(unsigned int) * numElems));

    unsigned int stride = numThreadsPerSM;

    for (unsigned int i = 0; i < numIterations; i++)
    {
        if (i % 2 == 0) 
            computeLowestBlockPrefixSum<<<gridDim, blockDim, sizeof(unsigned int) * 2 * numThreadsPerSM>>>(
                                                    d_inputVals, 
                                                    d_zerosHistogram, 
                                                    d_onesHistogram, 
                                                    numElems, 
                                                    i + radixLow);
        else
            computeLowestBlockPrefixSum<<<gridDim, blockDim, sizeof(unsigned int) * 2 * numThreadsPerSM>>>(
                                                    d_outputVals, 
                                                    d_zerosHistogram, 
                                                    d_onesHistogram, 
                                                    numElems, 
                                                    i + radixLow);

        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        unsigned int strideLevels = 0;
        unsigned int currentStride = stride;
        unsigned int numBlocks = 0;

        // Step 2: Compute prefix sum (CDF) of the histogram
        while (numElems > currentStride) {
            gridOverGridDim.x = (numElems + currentStride * stride - 1) / (currentStride * stride);
            numBlocks = (numElems + currentStride - 1) / currentStride;
            computeStridedPrefixSum<<<gridOverGridDim, blockDim, sizeof(unsigned int) * 2 * numThreadsPerSM>>>(
                                                            d_zerosHistogram, 
                                                            d_onesHistogram, 
                                                            numBlocks, 
                                                            currentStride);
            strideLevels++;
            currentStride *= stride;
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        }

        // Step 3: Combine the results of the first two kernels to compute the location each
        // element must be moved to and then move them.
        // Step 2: Compute prefix sum (CDF) of the histogram

        if (i % 2 == 0) 
            moveElements<<<gridDim, blockDim>>>(d_inputVals,
                                                d_inputPos,
                                                d_outputVals,
                                                d_outputPos,
                                                d_zerosHistogram, 
                                                d_onesHistogram, 
                                                numElems, 
                                                i + radixLow, 
                                                stride, 
                                                strideLevels, 
                                                currentStride/stride);
        else
            moveElements<<<gridDim, blockDim>>>(d_outputVals,
                                                d_outputPos,
                                                d_inputVals,
                                                d_inputPos,
                                                d_zerosHistogram, 
                                                d_onesHistogram, 
                                                numElems, 
                                                i + radixLow, 
                                                stride, 
                                                strideLevels, 
                                                currentStride/stride);

        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }

    if (numIterations % 2 == 0) {
        checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, 
                               numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, 
                               numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    }

    checkCudaErrors(cudaFree(d_zerosHistogram));
    checkCudaErrors(cudaFree(d_onesHistogram));
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    radixSort(d_inputVals,
              d_inputPos,
              d_outputVals,
              d_outputPos,
              numElems,
              0,
              31);
    checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, 
                    numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

}