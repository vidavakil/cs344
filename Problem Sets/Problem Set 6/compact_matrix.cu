//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

static const unsigned int numThreadsPerSM = 1024;

__global__
void computeLowestBlockPrefixSum(const unsigned int* const d_inputVals,
                                       unsigned int* const d_prefixSum,
                                 const size_t              numColumns, // per row
                                 const size_t              numRows,
                                 const bool                isInitialized,
                                 const bool                ignoreFirstElement)
{
    unsigned int localBlockSize = (((blockIdx.x + 1) * blockDim.x) <= numColumns) ? blockDim.x : (numColumns % blockDim.x);
    unsigned int rowId = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int columnId = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int pos_1d = rowId * numColumns + columnId; // position in input and output arrays

    if (not isInitialized) {
        if (columnId < numColumns and rowId < numRows) {
            d_prefixSum[pos_1d] = (d_inputVals[pos_1d] != 0) ? 1 : 0;
            if (columnId == 0 and ignoreFirstElement)
                d_prefixSum[pos_1d] = 1;
        }
        __syncthreads();
    }

    // reverse columnId
    int rid = localBlockSize - 1 - threadIdx.x;

    unsigned int max_step = 0;
    // The reduction phase of Blelloch algorithm
    for (unsigned int step = 1; step < localBlockSize; step *= 2) {
        max_step = step;
        if (rowId < numRows && rid >= 0 && (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize)) {
            d_prefixSum[pos_1d] += d_prefixSum[pos_1d - step];
        }
        __syncthreads();
    }

    unsigned int totalOnes = 0;

    // The sweep phase of Blelloch algorithm

    // Before overriding the last element of the counts array,
    // store it away.
    if (rowId < numRows && rid == 0) {
        totalOnes = d_prefixSum[pos_1d];
        d_prefixSum[pos_1d] = 0;
    }

    __syncthreads();

    unsigned int temp;
    for (unsigned int step = max_step; step >= 1; step /= 2) {
        if (rowId < numRows && rid >= 0 && (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize)) {
            temp = d_prefixSum[pos_1d];
            d_prefixSum[pos_1d] += d_prefixSum[pos_1d - step];
            d_prefixSum[pos_1d - step] = temp;
        }  
        __syncthreads();
    }

    // Store the totals in the first element of the array. These special
    // elements constitute the histogram of zeros and ones in each thread
    // block.
    if (rowId < numRows && rid == 0) 
        d_prefixSum[rowId * numColumns + blockDim.x * blockIdx.x] = totalOnes;
}                      


__global__
void computeStridedPrefixSum(      unsigned int* const d_prefixSum, 
                             const size_t              numColumns,
                             const size_t              numRows,
                             const unsigned int        stride) 
{
    unsigned int localBlockSize = (((blockIdx.x + 1) * blockDim.x) <= numColumns) ? blockDim.x : (numColumns % blockDim.x);
    unsigned int rowId = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int columnId = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int pos_1d = rowId * numColumns + columnId * stride;

    // reverse tid
    int rid = localBlockSize - 1 - threadIdx.x;
 
    unsigned int max_step = 0;
    // The reduction phase of Blelloch algorithm
    for (unsigned int step = 1; step < localBlockSize; step *= 2) {
        max_step = step;
        if (rowId < numRows && rid >= 0 && (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize))
            d_prefixSum[pos_1d] += d_prefixSum[pos_1d - step * stride];
        __syncthreads();
    }

    // The sweep phase of Blelloch algorithm

    // Before overriding the last element of the counts array,
    // store it away.
    unsigned int totalOnes = 0;
    if (rowId < numRows && rid == 0) {
        totalOnes = d_prefixSum[pos_1d];
        d_prefixSum[pos_1d] = 0;
    }

    __syncthreads();

    unsigned int temp;
    for (unsigned int step = max_step; step >= 1; step /= 2) {
        if (rowId < numRows && rid >= 0 && (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize)) {
            temp = d_prefixSum[pos_1d];
            d_prefixSum[pos_1d] += d_prefixSum[pos_1d - step * stride];
            d_prefixSum[pos_1d - step * stride] = temp;
        }  
        __syncthreads();
    }

    // Store totalZeros in the first element of the arrays
    if (rowId < numRows && rid == 0)
        d_prefixSum[rowId * numColumns + blockDim.x * blockIdx.x * stride] = totalOnes;
}

__global__
void moveElements(const unsigned int* const d_inputVals,
                        unsigned int* const d_outputVals, 
                  const unsigned int* const d_prefixSum, 
                  const size_t              numColumns,
                  const size_t              numRows,
                  const unsigned int        stride,
                  const unsigned int        strideLevels,
                  const unsigned int        startStridalAddress, // stride ^ strideLevels
                  const bool                ignoreFirstElement)
{
    unsigned int rowId = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int columnId = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int pos_1d = rowId * numColumns + columnId;
    unsigned int addressTobeWrittenTo = rowId * numColumns;
    unsigned int stridalAddress = startStridalAddress;    
    const unsigned int* lookUpAddress = &d_prefixSum[rowId * numColumns];

    unsigned int  roundedAddress = 0;    

    assert(blockDim.x == stride);
    assert(stridalAddress != 0);
    
    if (rowId < numRows && columnId > 0 && columnId < numColumns && stridalAddress >= 1) {
        while (columnId % stridalAddress != 0) {
            roundedAddress =  stridalAddress * (columnId / stridalAddress);
            if (roundedAddress != 0 && roundedAddress % (stridalAddress * stride) != 0)
                addressTobeWrittenTo += lookUpAddress[roundedAddress];
            stridalAddress /= stride;
            assert(addressTobeWrittenTo < numColumns);
        }
        assert(addressTobeWrittenTo < numColumns);
        addressTobeWrittenTo += lookUpAddress[columnId];
    }
    if (rowId < numRows && columnId < numColumns && (d_inputVals[pos_1d] != 0 || (columnId == 0 && ignoreFirstElement)))
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
void compactMatrix(const unsigned int* const  d_inputVals,      // some are zeros
                         unsigned int*       &d_outputVals,     // compacted array, preallocated, same size as d_inputVals
//                         unsigned int* const  compactedSize,    // an array of size numRows, containing the size of each compacted row.
//                         unsigned int        &maxCompactedSize, // maximum size of the compacted rows 
                   const size_t               numColumns,         // number of elements per row
                   const size_t               numRows,          // number of rows in the input matrix
                   const bool                 ignoreFirstElement)
{ 
    dim3 blockDim(numThreadsPerSM);
    dim3 gridDim(numRows, (numColumns + blockDim.x - 1) / blockDim.x);
    dim3 gridOverGridDim(numRows, (gridDim.x + blockDim.x - 1) / blockDim.x);

/*
    printf("compactArray: numColumns = %lu, gridDim.x = %u, blockDim.x = %u, gridOverGridDim = %u\n", 
           numColumns, gridDim.x, blockDim.x, gridOverGridDim.x);
*/

    // Allocate memory for the scatter addresses
    unsigned int *d_prefixSum;
    checkCudaErrors(cudaMalloc(&d_prefixSum, sizeof(unsigned int) * numColumns * nomRows));

    unsigned int stride = numThreadsPerSM;
    bool alreadyInitialized = false;

    computeLowestBlockPrefixSum<<<gridDim, blockDim>>>(d_inputVals, d_prefixSum, numColumns, numRows, alreadyInitialized, ignoreFirstElement);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    unsigned int strideLevels = 0;
    unsigned int currentStride = stride;
    unsigned int numBlocks = 0;

    // Step 2: Compute prefix sum (CDF) of the histogram
    while (numColumns > currentStride) {
        gridOverGridDim.x = (numColumns + currentStride * stride - 1) / (currentStride * stride);
        numBlocks = (numColumns + currentStride - 1) / currentStride;
        computeStridedPrefixSum<<<gridOverGridDim, blockDim>>>(d_prefixSum, numBlocks, numRows, currentStride);
        strideLevels++;
        currentStride *= stride;
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());        
    }
/*
    unsigned int* h_onesHistogram;
    checkCudaErrors(cudaMalloc(&h_onesHistogram, sizeof(unsigned int) * numColumns * nomRows));
    checkCudaErrors(cudaMemcpy(&h_onesHistogram, d_prefixSum, 
                               sizeof(unsigned int) * numColumns * nomRows, cudaMemcpyDeviceToHost));
    maxCompactedSize = 0;
    for (unsigned int i = 0; i < numRows; i++) {
        compactedSize[i] = h_onesHistogram[numColumns * i];
        if (compactedSize[i] > maxCompactedSize)
            maxCompactedSize = compactedSize[i];
    }

    checkCudaErrors(cudaMalloc(&d_outputVals, sizeof(int) * maxCompactedSize * nomRows));
*/
    // Step 3: Combine the results of the first two kernels to compute the location each
    // element must be moved to and then move them.
    moveElements<<<gridDim, blockDim>>>(d_inputVals,
                                        d_outputVals,
                                        d_prefixSum, 
                                        numColumns,
                                        numRows, 
                                        stride, 
                                        strideLevels,
                                        currentStride / stride,
                                        ignoreFirstElement);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_prefixSum));
    checkCudaErrors(cudaFree(h_onesHistogram));
}


void expandMatrix(     unsigned int* const d_compactedBinNumBlocksPrefixSum, // to be filled
                       unsigned int* const numExpandedBins,    // an array of size numRows, containing the size of each compacted row. to be filled
                       unsigned int       &maxNumExpandedBins, // to be filled with the maximum element in numExpandedBins
                 const unsigned int        numCompactedBins,   // compactedSize
                 const unsigned int        numRows,
                 const unsigned int        blockSize,
                 const unsigned int        stride,
                       unsigned int       &stridalAddress)     // to be filled
{ 
    assert(blockSize == numThreadsPerSM);

    dim3 blockDim(blockSize);
    dim3 gridDim(numRows, (numCompactedBins + blockDim.x - 1) / blockDim.x);
    dim3 gridOverGridDim(numRows, (gridDim.x + blockDim.x - 1) / blockDim.x);

/*
    printf("expandArray: numCompactedBins = %u, gridDim.x = %u, blockDim.x = %u, gridOverGridDim = %u\n\n", 
           numCompactedBins, gridDim.x, blockDim.x, gridOverGridDim.x);
*/
    assert(stride == blockSize);

    computeLowestBlockPrefixSum<<<gridDim, blockDim>>>(NULL, d_compactedBinNumBlocksPrefixSum, numCompactedBins, numRows, true, false);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    unsigned int strideLevels = 0;
    unsigned int currentStride = stride;
    unsigned int numBlocks = 0;

    // Step 2: Compute prefix sum (CDF) of the histogram
    while (numCompactedBins > currentStride) {
        gridOverGridDim.x = (numCompactedBins + currentStride * stride - 1) / (currentStride * stride);
        numBlocks = (numCompactedBins + currentStride - 1) / currentStride;
        computeStridedPrefixSum<<<gridOverGridDim, blockDim>>>(d_compactedBinNumBlocksPrefixSum, numBlocks, numRows, currentStride);
        strideLevels++;
        currentStride *= stride;
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }

    stridalAddress = currentStride/stride;

    unsigned int* h_compactedBinNumBlocksPrefixSum;
    checkCudaErrors(cudaMalloc(&h_compactedBinNumBlocksPrefixSum, sizeof(unsigned int) * numCompactedBins * nomRows));
    checkCudaErrors(cudaMemcpy(&h_compactedBinNumBlocksPrefixSum, d_compactedBinNumBlocksPrefixSum, 
                               sizeof(unsigned int) * numCompactedBins * nomRows, cudaMemcpyDeviceToHost));

    maxNumExpandedBins = 0;
    for (unsigned int i = 0; i < numRows; i++) {
        numExpandedBins[i] = h_compactedBinNumBlocksPrefixSum[numCompactedBins * i];
        if (numExpandedBins[i] > maxNumExpandedBins)
            maxNumExpandedBins = numExpandedBins[i];
    }

    checkCudaErrors(cudaFree(h_onesHistogram));

//    printf("maxNumExpandedBins = %u\n", maxNumExpandedBins);
}
