#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>

static const unsigned int numThreadsPerSM = 1024;

__global__
void computeLowestBlockReduceMinMax(      unsigned char* const d_inputVals,
                                          unsigned int* const  d_reduceMin,
                                          unsigned int* const  d_reduceMax,
                                    const size_t               numColumns,
                                    const size_t               numRows,
                                    const bool                 isInitialized)
{
    unsigned int localBlockSize = (((blockIdx.x + 1) * blockDim.x) <= numColumns)
                                  ? blockDim.x
                                  : (numColumns % blockDim.x);
    unsigned int rowId = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int columnId = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int pos_1d = rowId * numColumns + columnId; // position in input and output arrays

    if (!isInitialized) {
        if (columnId < numColumns and rowId < numRows) 
            {
                if (d_inputVals[pos_1d] == 0) {
                    d_reduceMin[pos_1d] = 1000000;  // largest unsigned int
                    d_reduceMax[pos_1d] = 0;
                } else {
                    d_reduceMin[pos_1d] = columnId; 
                    d_reduceMax[pos_1d] = columnId;
                }
            }
        __syncthreads();
    }

    unsigned int tid = threadIdx.x;
    unsigned int myVal;
    unsigned int otherVal;

    // The reduction phase of Blelloch algorithm
    for (unsigned int step = 1; step < localBlockSize; step *= 2) {
        if (rowId < numRows && tid < localBlockSize && 
            (tid % (step * 2) == 0) && (tid + step < localBlockSize)) {
            myVal = d_reduceMin[pos_1d];
            otherVal = d_reduceMin[pos_1d + step];
            d_reduceMin[pos_1d] = (myVal < otherVal) ? myVal : otherVal;

            myVal = d_reduceMax[pos_1d];
            otherVal = d_reduceMax[pos_1d + step];
            d_reduceMax[pos_1d] = (myVal > otherVal) ? myVal : otherVal;
        }
        __syncthreads();
    }

    // The min and max are now already stored at the first element of the block. 
    // These special elements will then be used in higher levels of PrefixMin/PrefixMax
}

__global__
void computeStridedReduceMinMax(unsigned int* const d_reduceMin, 
                                unsigned int* const d_reduceMax,
                                              const size_t numColumns,
                                              const size_t numRows,
                                unsigned int stride) 
{
    unsigned int localBlockSize = (((blockIdx.x + 1) * blockDim.x) <= numColumns)
                                   ? blockDim.x
                                   : (numColumns % blockDim.x);
    unsigned int rowId = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int columnId = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int pos_1d = rowId * numColumns + columnId * stride;

    unsigned int tid = threadIdx.x;
    unsigned int myVal;
    unsigned int otherVal;

    for (unsigned int step = 1; step < localBlockSize; step *= 2) {
        if (rowId < numRows && tid < localBlockSize && (
            tid % (step * 2) == 0) && (tid + step < localBlockSize)) {
            myVal = d_reduceMin[pos_1d];
            otherVal = d_reduceMin[pos_1d + step * stride];
            d_reduceMin[pos_1d] = (myVal < otherVal) ? myVal : otherVal;

            myVal = d_reduceMax[pos_1d];
            otherVal = d_reduceMax[pos_1d + step * stride];
            d_reduceMax[pos_1d] = (myVal > otherVal) ? myVal : otherVal;
            printf("h ");
        }
        __syncthreads();
    }

    // The min and max are now already stored at the first element of the strided block. 
    // These special elements will then be used in next level of PrefixMin/PrefixMax

    //    d_reduceMin[blockDim.x * blockIdx.x * stride] = totalZeros;
    //    d_reduceMax[blockDim.x * blockIdx.x * stride] = totalOnes;
}

__global__
void gatherMinAndMaxReduces(const unsigned int* const        d_reduceMin, 
                              const unsigned int* const        d_reduceMax,
                                    unsigned int* const        d_rowReduceMin,
                                    unsigned int* const        d_rowReduceMax,
                                    unsigned int* const        d_rowIds,
                                    unsigned int* const        d_rowPrefixSum, 
                                                  const size_t numColumns,
                                                  const size_t numRows)
{
    unsigned int columnId = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int rowId = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int pos_1d = rowId * numColumns + columnId;

    unsigned int value;

    if (columnId == 0 && rowId < numRows) {
        value = d_reduceMax[pos_1d];
        d_rowReduceMax[rowId] = value;
        d_rowReduceMin[rowId] = d_reduceMin[pos_1d];
        d_rowIds[rowId] = rowId;
        d_rowPrefixSum[rowId] = (value == 0) ? 0 : 1; 
        // There is a subtle bug here! If a row has a single masked pixel at x_max = 0, 
        // then the rowPrefixSum takes it to mean that that row does not have any masked pixels!
    }
}

__global__
void computeLowestBlockPrefixSum(const unsigned int* const d_inputVals,
                                       unsigned int* const d_prefixSum,
                                 const size_t              numColumns, // per row
                                 const size_t              numRows,
                                 const bool                isInitialized,
                                 const bool                enforceFirstElement) // if true, always set the value for position 0
{
    unsigned int localBlockSize = (((blockIdx.x + 1) * blockDim.x) <= numColumns)
                                 ? blockDim.x
                                 : (numColumns % blockDim.x);
    unsigned int rowId = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int columnId = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int pos_1d = rowId * numColumns + columnId; // position in input and output arrays

    // It is not a good practice to use data itself as the predicate, because 0 could be a valid
    // data, but when you compute a predicate on it, the result is false. That's why in general
    // we want ot have a data-array, a predicate-array, and a prefix-sum-array.

    if (not isInitialized) {
        if (columnId < numColumns and rowId < numRows) {
            d_prefixSum[pos_1d] = (d_inputVals[pos_1d] != 0) ? 1 : 0;
            if (columnId == 0 and enforceFirstElement)
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
        if (rowId < numRows && rid >= 0 && 
            (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize)) {
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
        if (rowId < numRows && rid >= 0 && 
            (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize)) {
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
    unsigned int localBlockSize = (((blockIdx.x + 1) * blockDim.x) <= numColumns)
                                  ? blockDim.x 
                                  : (numColumns % blockDim.x);
    unsigned int rowId = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int columnId = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int pos_1d = rowId * numColumns + columnId * stride;

    // reverse tid
    int rid = localBlockSize - 1 - threadIdx.x;
 
    unsigned int max_step = 0;
    // The reduction phase of Blelloch algorithm
    for (unsigned int step = 1; step < localBlockSize; step *= 2) {
        max_step = step;
        if (rowId < numRows && rid >= 0 && 
            (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize))
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
        if (rowId < numRows && rid >= 0 && 
            (rid % (step * 2) == 0) && (rid + (int)step < (int)localBlockSize)) {
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
void moveElements(const unsigned int* const d_predicate,
                  const unsigned int* const d_inputVals,
                        unsigned int* const d_outputVals, 
                  const unsigned int* const d_prefixSum, 
                  const size_t              numColumns,
                  const size_t              numRows,
                  const unsigned int        stride,
                  const unsigned int        strideLevels,
                  const unsigned int        startStridalAddress) // stride ^ strideLevels
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
    
    if (rowId < numRows && columnId > 0 && 
        columnId < numColumns && stridalAddress >= 1) {
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
    if (rowId < numRows && columnId < numColumns && d_predicate[pos_1d] != 0)
        d_outputVals[addressTobeWrittenTo] = d_inputVals[pos_1d];   
}

void findBoundaryBox(      unsigned char* const d_inputVals,
                     const size_t               numColumns,
                     const size_t               numRows,
                           unsigned int*        boundaryBox) // array of size 4: x_min, y_min, x_max, y_max
{ 
    dim3 blockDim(numThreadsPerSM);
    dim3 gridDim((numColumns + blockDim.x - 1) / blockDim.x, numRows);
    dim3 gridOverGridDim((gridDim.x + blockDim.x - 1) / blockDim.x, numRows);


    printf("findBoundaryBox: numColumns = %lu, numRows = %lu, gridDim.x = %u, blockDim.x = %u, gridOverGridDim = %u\n", 
           numColumns, numRows, gridDim.x, blockDim.x, gridOverGridDim.x);

    // Allocate memory for min and max reduces
    unsigned int *d_reduceMin, *d_reduceMax;

    checkCudaErrors(cudaMalloc(&d_reduceMin, 
                    sizeof(unsigned int) * numColumns * numRows));
    checkCudaErrors(cudaMalloc(&d_reduceMax, 
                    sizeof(unsigned int) * numColumns * numRows));

    unsigned int stride = numThreadsPerSM;
    bool alreadyInitialized = false;

    computeLowestBlockReduceMinMax<<<gridDim, blockDim>>>(d_inputVals, 
                                                          d_reduceMin, 
                                                          d_reduceMax, 
                                                          numColumns, 
                                                          numRows, 
                                                          alreadyInitialized);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    unsigned int strideLevels = 0;
    unsigned int currentStride = stride;
    unsigned int numBlocks = 0;

    while (numColumns > currentStride) {
        gridOverGridDim.x = (numColumns + currentStride * stride - 1) / (currentStride * stride);
        numBlocks = (numColumns + currentStride - 1) / currentStride;
        computeStridedReduceMinMax<<<gridOverGridDim, blockDim>>>(d_reduceMin, 
                                                                  d_reduceMax, 
                                                                  numBlocks,
                                                                  numRows, 
                                                                  currentStride);
        strideLevels++;
        currentStride *= stride;
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }

    unsigned int *d_rowReduceMin, *d_rowReduceMax, *d_rowPrefixSum, *d_rowIds, *d_compactedRowIds;

    checkCudaErrors(cudaMalloc(&d_rowReduceMin, sizeof(unsigned int) * numRows));
    checkCudaErrors(cudaMalloc(&d_rowReduceMax, sizeof(unsigned int) * numRows));
    checkCudaErrors(cudaMalloc(&d_rowPrefixSum, sizeof(unsigned int) * numRows));
    checkCudaErrors(cudaMalloc(&d_rowIds, sizeof(unsigned int) * numRows));
    checkCudaErrors(cudaMalloc(&d_compactedRowIds, sizeof(unsigned int) * numRows));

    gatherMinAndMaxReduces<<<gridDim, blockDim>>>(d_reduceMin, 
                                                  d_reduceMax,
                                                  d_rowReduceMin,
                                                  d_rowReduceMax,
                                                  d_rowIds,
                                                  d_rowPrefixSum,
                                                  numColumns,
                                                  numRows);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //============================================================================
    // To find y_min and y_max, we need to compute the prefixSum of d_rowReduceMax
    // with predicate isNonZero, and do a compaction of rowIds using the prefixSum.
    // Then the first element of the compaction will be y_min, and the last element
    // of the compaction will be y_max (indexed by first element of the prefixSum)
/*
    printf("============= h_rowPrefixSum raw =============\n\n");
    unsigned int h_rowPrefixSum[numRows];
    checkCudaErrors(cudaMemcpy(h_rowPrefixSum, d_rowPrefixSum, 
                                sizeof(unsigned int) * numRows, cudaMemcpyDeviceToHost));
    for (unsigned int i = 0; i < numRows; i++)
        printf("%u ", h_rowPrefixSum[i]);
    printf ("\n\n");
*/

    alreadyInitialized = true;
    bool enforceFirstElement = false;
    gridDim.y = 1;
    gridDim.x = (numRows + blockDim.x - 1) / blockDim.x;

    computeLowestBlockPrefixSum<<<gridDim, blockDim>>>(NULL, 
                                                       d_rowPrefixSum, 
                                                       numRows, 
                                                       1, 
                                                       alreadyInitialized, 
                                                       enforceFirstElement);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    strideLevels = 0;
    currentStride = stride;
    numBlocks = 0;

    // Step 2: Compute prefix sum (CDF) of the histogram
    while (numRows > currentStride) {
        gridOverGridDim.x = (numRows + currentStride * stride - 1) / (currentStride * stride);
        numBlocks = (numRows + currentStride - 1) / currentStride;
        computeStridedPrefixSum<<<gridOverGridDim, blockDim>>>(d_rowPrefixSum, 
                                                               numBlocks, 
                                                               1, 
                                                               currentStride);
        strideLevels++;
        currentStride *= stride;
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());        
    }

/*
    printf("============= h_rowPrefixSum final =============\n\n");
    checkCudaErrors(cudaMemcpy(h_rowPrefixSum, d_rowPrefixSum, 
                                sizeof(unsigned int) * numRows, cudaMemcpyDeviceToHost));
    for (unsigned int i = 0; i < numRows; i++)
        printf("%u ", h_rowPrefixSum[i]);
    printf ("\n\n");
*/

    moveElements<<<gridDim, blockDim>>>(d_rowReduceMax,
                                        d_rowIds,
                                        d_compactedRowIds,
                                        d_rowPrefixSum, 
                                        numRows,
                                        1, 
                                        stride, 
                                        strideLevels,
                                        currentStride / stride);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

/*
    printf("============= h_compactedRowIds=============\n\n");   
    unsigned int h_compactedRowIds[numRows];
    checkCudaErrors(cudaMemcpy(h_compactedRowIds, d_compactedRowIds, 
                                sizeof(unsigned int) * numRows, cudaMemcpyDeviceToHost));
    for (unsigned int i = 0; i < numRows; i++)
        printf("%u ", h_compactedRowIds[i]);
    printf ("\n\n");
*/

    // At this point, d_compactedRowIds[0] = y_min, d_compactedRowIds[d_rowPrefixSum[0] - 1] = y_max


    //============================================================================
    // Now compute x_min and x_max

    alreadyInitialized = true;
    computeLowestBlockReduceMinMax<<<gridDim, blockDim>>>(NULL, 
                                                          d_rowReduceMin, 
                                                          d_rowReduceMax, 
                                                          numRows, 
                                                          1, 
                                                          alreadyInitialized);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    strideLevels = 0;
    currentStride = stride;
    numBlocks = 0;

    gridOverGridDim.y = 1;

    while (numRows > currentStride) {
        gridOverGridDim.x = (numRows + currentStride * stride - 1) / (currentStride * stride);
        numBlocks = (numRows + currentStride - 1) / currentStride;
        computeStridedReduceMinMax<<<gridOverGridDim, blockDim>>>(d_rowReduceMin, 
                                                                  d_rowReduceMax,
                                                                  numBlocks,
                                                                  1, // numRows
                                                                  currentStride);
        strideLevels++;
        currentStride *= stride;
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }

    // At this point, d_rowReduceMin[0] = x_min, and d_rowReduceMax[0] = x_max
/*
    printf("============= h_rowReduceMin =============\n\n");
    unsigned int h_rowReduceMin[numRows];
    checkCudaErrors(cudaMemcpy(h_rowReduceMin, d_rowReduceMin, 
                                sizeof(unsigned int) * numRows, cudaMemcpyDeviceToHost));
    for (unsigned int i = 0; i < numRows; i++)
        printf("%u ", h_rowReduceMin[i]);
    printf ("\n\n");

    printf("============= h_rowReduceMax =============\n\n");   
    unsigned int h_rowReduceMax[numRows];
    checkCudaErrors(cudaMemcpy(h_rowReduceMax, d_rowReduceMax, 
                                sizeof(unsigned int) * numRows, cudaMemcpyDeviceToHost));
    for (unsigned int i = 0; i < numRows; i++)
        printf("%u ", h_rowReduceMax[i]);
    printf ("\n\n");
*/

    unsigned int maxRows;

    checkCudaErrors(cudaMemcpy(&boundaryBox[0], &d_rowReduceMin[0], 
                                sizeof(unsigned int), cudaMemcpyDeviceToHost)); // x_min
    checkCudaErrors(cudaMemcpy(&boundaryBox[1], &d_rowReduceMax[0], 
                                sizeof(unsigned int), cudaMemcpyDeviceToHost)); // x_max
    checkCudaErrors(cudaMemcpy(&boundaryBox[2], &d_compactedRowIds[0], 
                                sizeof(unsigned int), cudaMemcpyDeviceToHost)); // y_min
    checkCudaErrors(cudaMemcpy(&maxRows, &d_rowPrefixSum[0], 
                                sizeof(unsigned int), cudaMemcpyDeviceToHost)); // numNonZeroRows

    assert(maxRows > 0);
//    printf("maxRows = %u\n", maxRows);
    checkCudaErrors(cudaMemcpy(&boundaryBox[3], &d_compactedRowIds[maxRows - 1], 
                                sizeof(unsigned int), cudaMemcpyDeviceToHost)); // y_max
        
    printf("findBoundaryBox: x_min = %u, x_max = %u, y_min = %u, y_max = %u\n", 
           boundaryBox[0], boundaryBox[1], boundaryBox[2], boundaryBox[3]);

    checkCudaErrors(cudaFree(d_reduceMin));
    checkCudaErrors(cudaFree(d_reduceMax));
    checkCudaErrors(cudaFree(d_rowReduceMin));
    checkCudaErrors(cudaFree(d_rowReduceMax));
    checkCudaErrors(cudaFree(d_rowPrefixSum));
    checkCudaErrors(cudaFree(d_rowIds));
    checkCudaErrors(cudaFree(d_compactedRowIds));
}
