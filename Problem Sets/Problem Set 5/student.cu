/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


// @ 2025 Vida Vakilotojar

/* 
  The approach: 
  1- First, divide numBins into K coraser bins, and find all the numbers that
  belong to each of these coarse bins.
  2- To do the above, sort all the input numbers based on their coarse bin number.
  3- Then for each coarse bin (that corresponds to a section of size numBins/K of 
  the output d_histogram), create its histogram of the numbers in the coarse bin.

  To sort the numbers based on their coarse bin, use radix sort, but ignore the
  lsb bits of the numbers! If you ignore the n lowest bits of the numbers, 
  then each coarse bin will have a size of 2^n. 

  After the sort we can identify the boundaries of coarse bins in the sorted array 
  by having each thread compute the coarse bin number of the number it is responsible 
  for, and that of the number on its left, and if the two are not the same, then
  that number marks the start of the coarse bin for that coarse bin id, and the
  thread writes that address into a global array for the start of coarse bins, for
  the corresponding coarse bin.

  In a following step, one thread block can be assigned to each coarse bin, to compute
  its histogram and write it to the corresponding section of d_histogram. Note that 
  the number of fine bins in a coarse bin is 2^n. That is, you can find the fine bin
  of a number within its coarse bin by doing a mod 2^n operation.

  Now, to count the number of items in each fine bin of a corase bin, we can use 
  a shared memory and atomicAdds, and once all threads have performed their atomicAdds,
  2^n of those threads can go and do a non-atomic write to the global memory (because
  different threadblocks are writing to different bins of d_histogram).

  To avoid the atomicAdds in shared memory, we can sort the coarse bins, just like
  we had sorted the numbers based on their coarse bin numbers, and then find the
  boundaries of the fine bins in each sorted coarse bin, and by subtracting those
  boundaries find the number of elements in each fine bin. This will require
  implementing radix sort over the n lsb bits of the numbers in each coarse bin,
  where this time we have many thread blocks computing these smaller radix sort
  problems in parallel! But then, what it takes to do such radix sort is already
  enough to compute the fine-grained histogram we need! So, forget about Radix
  sort for each corase bin.

  Since the numbers are Normally distributed, the number of elements in each
  coarse bin would not be the same; some will have much more elements than the
  others. If we assign one thread block for each coarse bin, then the algorithm
  won't be efficient, because of load balance issues, and the slowest running 
  thread block will determine the speed of the whole algorithm. 

  If after sorting the numbers based on their coarse bins, we partition them
  into equaly sized thread blocks, then in each thread block we may have numbers that 
  belong to different coarse bins. However, if we use radix sort (on the lsb bits
  this time), that won't destroy the order of numbers with respect to their 
  msb bits (corase bin numbers). However, if a corase bin is split between two
  thread blocks, both of them will be contributing to the fine bins of the coarse
  bin, and thus cannot determine the final number for a particular fine bin, and 
  have to use atimicAdds in global memory! But this is precisely what we had tried
  to avoid!

  As for coarse bins that have
  too many elements in them, we can do a couple of radix sorts in them, on mid bits
  bellow bit n. This will break down the last coarse bin smaller segments that
  are sorted with respect to each other, but the elements in each segment are 
  not sorted. Then, we can assign these segments to different thread blocks.

  Perhaps we can do this:
  At the very beginning, perform a radix sort on the MSB bit, to partition the
  data into two segments. If any segment is larger than a given size (say maxSize),
  then sort that segment using radix sort and break it into two segments, and recurse.
  Perhaps for each element in the intermediate mixedly sorted array, we indicate
  what is the smallest bit it was sorted on! For all segments that are below
  maxSize, we can immediately call the last step of the algorithm on them,
  that uses atomicAdds in shared memory and then writes to global memory, or
  uses radix sort on the remaining bits to sort the elements and then count
  using pointers and write to global memory.

  The problem is that if a finest bin has more elements than maxSize, then 
  this algorithm won't work, because the elements that belong to that fine
  bint will be spread across multiple thread blocks of size maxSize, and they
  cannot count those without using global atomicAdds!

  Essentially, you can write to global memory without atomicAdds only for elements
  in a thread block that are larger than the first element in that thread block,
  and are smaller than the largest element in that thread block (it is like the
  special elements at the borders of an image we are performing convolution on).
  But hopefully we won't have many segments that cross boundaries. For any number
  that does not satisfy this requirement, after counts are computed in shared
  memory using atomicAdds (or if you used full radix sort inside the thread block
  then you can count them without using atomicAdds), then we would need to do
  gloabl atomicAdds for those fine bins in d_histogram.


  ///////////
  After doing a Radix sort to sort data based on their coarse bin IDs (K1), and identifying
  the boundaries of corase bins and writing them into an intermediate array (K2), the
  CPU can analyze that data and break down largers segments into subsegments, so 
  that all thread blocks (of K3) that compute the fine-grained histograms have the same
  amount of job to do. For that, the CPU has to create a structure of arrays as below:

  Unsinged int *startOfSement; // starting address of the coarsely sorted segment/sub-segment
  unsigned int *segmentSize;   // size of the segment for which to compute a fine-grained histogram
  unsigned int *coarseBinId;   // the coarse ID of this segment. Used by threads of the 
                               // thread block for determining the starting address of the 
                               // fine-grained histogram that they have to write back into
                               // global memory
  unsinged int *useAtomicAdd;  // if this is a sub-segment, and thus when writing its
                               // fine-grained histogram to global memory, the threads
                               // shold do a global atomicAdd() instead of a simple
                               // write, because other sub-segments of the same coarse
                               // segment are also writing to the same addresses.

  The size of the above arrays equals the number of thread blocks that are needed to
  compute the full histogram. The CPU scans through the results of K2, for each small
  size segment, it just copies its information to the next entry in the above SoA
  (structure of arrays), setting useAtomic = False. But for every large segment,
  it writes down multiple entries into the above SoA, until together they cover the 
  whole corase segment, and for all of them it sets useAtomic = True. The CPU uses two 
  separate pointers, one to the output of the K2 results, and one to the last entry 
  in the above SoA. Once the above SoA is prepared, the CPU calls K3, and they will 
  produce the final histogram.

  For K3, the threads will be given a share memory of size num_threads * numFineBins.
  Each threads writes a 1 to the corresponding entry of this shared memory, according 
  to which fine bin its data falls into. They then compute numBin Reduces, one per
  row/fineBin in the shared memory (they can be done in an inner loop of the Reduce).
  Thus, the number of elements in each fineBin will be computed as a result of the 
  Exclusive Reduce. Then, numBin of the threads would srite the results to the global
  histogram, using atomicAdds if they have to.

  If we wanted to avoid the atomicAdds required by the subsegments above, we have to 
  first Radix sort those subsegments based on their next lsb bit, and then break them
  into two sub-segments (of possibly different sizes). Since there could be multiple
  corase segments in the data that need such break down, and they may not be contiguous,
  and they each may need different number of levels of extra radix sort to break them
  all down to manageable sizes, implementing a CPU/GPU algorithm that does all of that
  can be complex, and the step complexity (the speed) would be determined by the 
  largest of those segments anyway (even if we could parallelize the sort of all 
  spread out segments that need a Radix sort by a given bit position). So, perhaps
  it's best to just accept some worst case serialized atomicAdds to global memory 
  (which is determined by the size of the largest coarse segment, divided by the number
  of sub-segments it it broken down to).

  If the normal distribution has a very low variance, then the atomicAdds can become
  very expensive, because most of the writes will be made to the same fine-grained
  bins. 

  To fix this issue, We can have the threadblocks that were told to use atomicAdds,
  to instead write out their fine-grained results into a dedicated output array each.
  Then a K4 kernel can go and do a Reduce over the fine-grained bins of the sub-segments
  of the large coarse segments. Although this would be worth sending to GPU only if
  the number of such subsegments is really huge! Otherwise, perhaps its better if the
  CPU itself performs a reduce over those results. But remember that for that,
  the CPU has to transfer all those results to the host memory, which could be more
  expensive than letting the K3 kernels just use atomicAdds!

*/

#include "utils.h"
#include "timer.h"
#if defined(_WIN16) || defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#else
#include <sys/time.h>
#endif


void radixSort(unsigned int* const d_inputVals,
               const size_t numElems,
               unsigned int radixLow,
               unsigned int radixHigh);

void compactArray(      unsigned int* const d_inputVals,  // some are zeros
                        unsigned int*      &d_vals,         // compacted array that has to be allocated
                        unsigned int       &compactedSize,         // size of the compacted array
                  const size_t              numElems,
                        bool                ignoreFirstElement);

void expandArray(      unsigned int* const d_compactedBinNumBlocksPrefixSum,
                       unsigned int       &numExpandedBins, // to be overwriten
                 const unsigned int        numCompactedBins,
                 const unsigned int        blockSize,
                       unsigned int        stride,
                       unsigned int       &stridalAddress);

static const unsigned int numThreadsPerSM = 1024;

__global__
void findCoarseBinBoundaries(const unsigned int* const d_vals,
                                   unsigned int* const d_coarseBinBoundaries,
                             const unsigned int        numElems,
                             const unsigned int        numFineBins,
                             const unsigned int        numCoarseBinsPlus1)
{
    // Entry n in d_coarseBinBoundaries will have the address of the first
    // element in the nth coarse bin. Any coarse bin that is empty, will
    // have the preset value of 0, which is ambiguous for bin 0, but for
    // all other bins it means the bin is empty.

    unsigned int bin;
    unsigned int pos_1d = blockDim.x * blockIdx.x + threadIdx.x;

    if (pos_1d < numElems) {
        bin = d_vals[pos_1d] / numFineBins;
        if (pos_1d == 0 or bin != d_vals[pos_1d - 1] / numFineBins) {
            assert(bin < numCoarseBinsPlus1 - 1);
            d_coarseBinBoundaries[bin] = pos_1d;
        }
    if (pos_1d == numElems - 1)
        d_coarseBinBoundaries[numCoarseBinsPlus1 - 1] = pos_1d + 1;
    }
}       

__global__
void breakDownCompactedCoarseBins(unsigned int* const d_compactedBoundaries,
                                  unsigned int* const d_coarseBinSizes,                 // to be filled
                                  unsigned int* const d_numBlocksPerCoarseBin,          // to be filled
                                  unsigned int* const d_numBlocksPerCoarseBinPrefixSum, // to be filled
                                  unsigned int blockSize,
                                  unsigned int compactedSize)
{
    unsigned int bin = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int binSize = 0;

    if (bin < compactedSize - 1)
        binSize = d_compactedBoundaries[bin + 1] - d_compactedBoundaries[bin];

    if (bin < compactedSize - 1) {
        d_coarseBinSizes[bin] = binSize;
        binSize = (binSize + blockSize - 1) / blockSize;
        d_numBlocksPerCoarseBin[bin] = binSize;
        d_numBlocksPerCoarseBinPrefixSum[bin] = binSize;
    }

    // The last bin is a dummy bin that we had created only so that the size of the bin before it could be calculated
    if (bin == compactedSize - 1) { 
        d_coarseBinSizes[bin] = 0;
        d_numBlocksPerCoarseBin[bin] = binSize;
        d_numBlocksPerCoarseBinPrefixSum[bin] = binSize;
    }
}
    
__global__
void computeSegmentInformation(const unsigned int* const d_coarseBinBoundaries,
                               const unsigned int* const d_coarseBinSizes,
                               const unsigned int* const d_numBlocksPerCoarseBin,
                               const unsigned int* const d_numCoarseBinsPrefixSum,
                                     unsigned int* const d_startOfSegment,
                                     unsigned int* const d_segmentSize,
                                     unsigned int* const d_useAtomicAdd,
                               const unsigned int        compactedSize,
                               const unsigned int        blockSize,
                               const unsigned int        stride,
                               const unsigned int        startStridalAddress)
{

    unsigned int pos_1d = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int offset = 0;
    unsigned int stridalAddress = startStridalAddress;
    const unsigned int* lookUpAddress = d_numCoarseBinsPrefixSum;

    unsigned int  roundedAddress = 0;    

    assert(blockDim.x == stride);
    assert(stridalAddress != 0);
    
    if (pos_1d > 0 and pos_1d < compactedSize and stridalAddress >= 1) {
        while (pos_1d % stridalAddress != 0) {
            roundedAddress =  stridalAddress * (pos_1d / stridalAddress);
            if (roundedAddress != 0 && roundedAddress % (stridalAddress * stride) != 0)
                offset += lookUpAddress[roundedAddress];
            stridalAddress /= stride;
            assert(offset < compactedSize);
        }
        assert(offset < compactedSize);
        offset += lookUpAddress[pos_1d];
    }

    if (pos_1d < compactedSize) {
        unsigned int boundaries = d_coarseBinBoundaries[pos_1d];
        unsigned int totalSize = d_coarseBinSizes[pos_1d];
        unsigned int numBLocks = d_numBlocksPerCoarseBin[pos_1d];
        unsigned int useAtomic = (numBLocks > 1) ? 1 : 0;
        for (int i = 0; i < numBLocks; i++) {
            d_startOfSegment[offset] = boundaries;
            d_segmentSize[offset] = (totalSize >= blockSize) ? blockSize : totalSize;
            d_useAtomicAdd[offset] = useAtomic;
            boundaries += blockSize;
            offset++;
            totalSize -= blockSize;
        }
    }
}

__global__
void computeFineGrainedHistograms (const unsigned int* const d_vals, 
                                   const unsigned int* const d_startOfSegment, 
                                   const unsigned int* const d_segmentSize, 
                                   const unsigned int* const d_useAtomicAdd,
                                   unsigned int* const d_histogram,                                    
                                   const unsigned int numFineBins,
                                   const unsigned int numBins,
                                   const unsigned int numElems,
                                   const unsigned int numCoarseBins,
                                   const unsigned int sharedMemoryStride)
{
    extern __shared__ unsigned int s_histogram[]; // [numFineBins][sharedMemoryStride] // +1 added to break bnank conflicts
    unsigned int tid = threadIdx.x;
    unsigned int segment = blockIdx.x;
    unsigned int localBlockSize = d_segmentSize[segment];
    unsigned int useAtomicAdd = d_useAtomicAdd[segment];

    // Reset the shared memory
    if (tid < localBlockSize)
        for (unsigned int i = 0; i < numFineBins; i++) {
            s_histogram[i * sharedMemoryStride + tid] = 0;
        }
    __syncthreads();

    unsigned int dataAddress = d_startOfSegment[segment] + tid;
    unsigned int data;
    unsigned int coarseBin = d_vals[d_startOfSegment[segment]]/numFineBins;
    unsigned int fineBin;

    if (tid < localBlockSize) {
        assert(dataAddress < numElems);
        assert(segment < numCoarseBins);
        data = d_vals[dataAddress];
        assert(coarseBin == data/numFineBins);
        assert(coarseBin < numCoarseBins);
        fineBin = data % numFineBins;
        s_histogram[fineBin * sharedMemoryStride + tid] = 1;
    }

    __syncthreads();

    // Reduce the rows of s_histogram!
    for (unsigned int step = 1; step < localBlockSize; step *= 2) {
        if ((tid < localBlockSize) && (tid % (step * 2) == 0) && (tid + step < localBlockSize)) {
            for (unsigned int i = 0; i < numFineBins; i++) {
                s_histogram[i * sharedMemoryStride + tid] += s_histogram[i * sharedMemoryStride + tid + step];
            }
        }
        __syncthreads();
    }

//    __syncthreads();

    // write s_histogram[i][0] to d_histogram.
    fineBin = coarseBin * numFineBins + tid;
    if (tid < numFineBins && fineBin < numBins) {
        data = s_histogram[tid * sharedMemoryStride];
        if (useAtomicAdd)
            atomicAdd(&d_histogram[fineBin], data); // data); // fineBin
        else
            d_histogram[fineBin] = data;
    }
}                                   

void computeHistogram(unsigned int* const d_vals, //INPUT  // removed const!
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{

    unsigned int radixLow = 0, radixHigh = 31;
    unsigned int numFineBins = 1 << radixLow;
    unsigned int blockSize = numThreadsPerSM;
    unsigned int gridSize = (numElems + blockSize - 1) / blockSize;
    unsigned int numCoarseBins = (numBins + numFineBins - 1) / numFineBins;
    unsigned int numCoarseBinsPlus1 = (numBins + 1 + numFineBins - 1) / numFineBins;

    unsigned int* d_coarseBinBoundaries;
    unsigned int* d_coarseBinSizes;
    unsigned int* d_numBlocksPerCoarseBin;
    unsigned int* d_numBlocksPerCoarseBinPrefixSum;

    // For simplicity assume below. Otherwise, we have to have more kernels to compute
    // a cdf that is distributed over multiple thread blocks.

    unsigned int *d_startOfSegment; 
    unsigned int *d_segmentSize;                
    unsigned int *d_useAtomicAdd;  

//    printf("numBins:%u, numElems:%u, numFineBins:%u, numCoarseBins:%u\n\n", numBins, numElems, numFineBins, numCoarseBins);
   
    radixSort(d_vals, numElems, radixLow, radixHigh);

    checkCudaErrors(cudaMalloc(&d_coarseBinBoundaries, sizeof(unsigned int) * numCoarseBinsPlus1)); // allocate one extra element
    checkCudaErrors(cudaMemset(d_coarseBinBoundaries, 0, sizeof(unsigned int) * numCoarseBinsPlus1));

    findCoarseBinBoundaries<<<gridSize, blockSize>>>(d_vals, d_coarseBinBoundaries, numElems, numFineBins, numCoarseBinsPlus1);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

/*
    unsigned int h_coarseBinBoundaries[numCoarseBinsPlus1];
    checkCudaErrors(cudaMemcpy(h_coarseBinBoundaries, d_coarseBinBoundaries, sizeof(unsigned int) * numCoarseBinsPlus1, cudaMemcpyDeviceToHost));
    printf("===d_coarseBinBoundaries===, numCoarseBinsPlus1: %u\n", numCoarseBinsPlus1);
    for (unsigned int i = 0; i < numCoarseBinsPlus1; i++)
        printf ("%u ", h_coarseBinBoundaries[i]);
    printf ("\n\n");
*/

    // Compact the array of boundaries, removing empty coarse bins.
    unsigned int* d_compactedBoundaries; // will be allocated by kernel
    unsigned int compactedSizePlus1;

    // findCoarseBinBoundaries returns an array of bin boundaries/addresses. An entry of 0
    // in this array means that the corresponding bin is empty. However, there is ambiguity
    // about bin 0, because if there is any data in that bin, then the boundary of bin 0
    // will indeed be 0. Whether or not that bin actually has data can be determined by
    // examining the first data element and seeing if it indeed belongs to bin 0.
    // However, for the task of compaction, we can assume that that bin is non-empty,
    // especially that we would have to use the results of the compacted list of addresses
    // to determine the size of each non-empty bin, and for that we indeed need a 0
    // in the first element of the compacted array, so that the size of the first bin
    // can be determined. That is why we have the following flag, ignoreFirstElement
    // and we set it to true for this special case.
    bool ignoreFirstElement = true;

    compactArray(d_coarseBinBoundaries,
                 d_compactedBoundaries, // will be allocated
                 compactedSizePlus1,         // to be returned
                 numCoarseBinsPlus1,
                 ignoreFirstElement);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMalloc(&d_coarseBinSizes, sizeof(unsigned int) * compactedSizePlus1));
    checkCudaErrors(cudaMalloc(&d_numBlocksPerCoarseBin, sizeof(unsigned int) * compactedSizePlus1));
    checkCudaErrors(cudaMalloc(&d_numBlocksPerCoarseBinPrefixSum, sizeof(unsigned int) * compactedSizePlus1));

/*
    unsigned int h_compactedBoundaries[compactedSizePlus1];
    checkCudaErrors(cudaMemcpy(h_compactedBoundaries, d_compactedBoundaries, sizeof(unsigned int) * compactedSizePlus1, cudaMemcpyDeviceToHost));
    printf("===d_compactedBoundaries===  compactedSizePlus1: %u\n", compactedSizePlus1);
    for (unsigned int i = 0; i < compactedSizePlus1; i++)
        printf ("%u ", h_compactedBoundaries[i]);
    printf ("\n\n");
*/

    unsigned int compactGridSize = (compactedSizePlus1 + blockSize - 1) / blockSize;
    breakDownCompactedCoarseBins<<<compactGridSize, blockSize>>>(d_compactedBoundaries,
                                 d_coarseBinSizes,
                                 d_numBlocksPerCoarseBin,
                                 d_numBlocksPerCoarseBinPrefixSum,
                                 blockSize,
                                 compactedSizePlus1);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

/*
    unsigned int h_numBlocksPerCoarseBinPrefixSum[compactedSizePlus1];
    checkCudaErrors(cudaMemcpy(h_numBlocksPerCoarseBinPrefixSum, d_numBlocksPerCoarseBinPrefixSum, sizeof(unsigned int) * compactedSizePlus1, cudaMemcpyDeviceToHost));
    printf("===d_numBlocksPerCoarseBinPrefixSum===\n");
    for (unsigned int i = 0; i < compactedSizePlus1; i++)
        printf ("%u ", h_numBlocksPerCoarseBinPrefixSum[i]);
    printf ("\n\n");
*/

    unsigned int compactedSize = compactedSizePlus1 - 1;

    unsigned int stridalAddress;
    expandArray(d_numBlocksPerCoarseBinPrefixSum,  // its prefixSum will be computed
                numCoarseBins,   // to be returned, the new number of coarsebins
                compactedSize, 
                blockSize,
                blockSize,       // stride, must be equal to blockSize
                stridalAddress); // to be returned, needed by computeSegmentInformation
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

/*
    checkCudaErrors(cudaMemcpy(h_numBlocksPerCoarseBinPrefixSum, d_numBlocksPerCoarseBinPrefixSum, sizeof(unsigned int) * compactedSize, cudaMemcpyDeviceToHost));
    printf("===d_numBlocksPerCoarseBinPrefixSum after expansion, compactedSize: %u ===\n", compactedSize);
    for (unsigned int i = 0; i < compactedSize; i++)
        printf ("%u ", h_numBlocksPerCoarseBinPrefixSum[i]);
    printf ("\n\n");
    checkCudaErrors(cudaFree(h_numBlocksPerCoarseBinPrefixSum));
*/

//    printf("After expansion: numCoarseBins: %u, stridalAddress: %u, compactedSize: %u\n\n", numCoarseBins, stridalAddress, compactedSize);

    // allocate memory for the SoAs:
    checkCudaErrors(cudaMalloc(&d_startOfSegment, sizeof(unsigned int) * numCoarseBins));
    checkCudaErrors(cudaMalloc(&d_segmentSize, sizeof(unsigned int) * numCoarseBins));
    checkCudaErrors(cudaMalloc(&d_useAtomicAdd, sizeof(unsigned int) * numCoarseBins));


    // Fill the SoA
    unsigned int  coarseBinGridSize = (numCoarseBins + blockSize - 1) / blockSize;
    computeSegmentInformation<<<coarseBinGridSize, blockSize>>>(d_compactedBoundaries,
                                                                d_coarseBinSizes,
                                                                d_numBlocksPerCoarseBin,
                                                                d_numBlocksPerCoarseBinPrefixSum,
                                                                d_startOfSegment,  // to be filled
                                                                d_segmentSize,     // to be filled
                                                                d_useAtomicAdd,    // to be filled
                                                                compactedSize,
                                                                blockSize,
                                                                blockSize, // stride
                                                                stridalAddress);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
/*
    unsigned int h_startOfSegment[numCoarseBins];
    unsigned int h_segmentSize[numCoarseBins];
    unsigned int h_useAtomicAdd[numCoarseBins];

    checkCudaErrors(cudaMemcpy(h_startOfSegment, d_startOfSegment, sizeof(unsigned int) * numCoarseBins, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_segmentSize, d_segmentSize, sizeof(unsigned int) * numCoarseBins, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_useAtomicAdd, d_useAtomicAdd, sizeof(unsigned int) * numCoarseBins, cudaMemcpyDeviceToHost));

    printf("=== d_startOfSegment ===\n");
    for (unsigned int i = 0; i < numCoarseBins; i++)
        printf ("%u ", h_startOfSegment[i]);
    printf ("\n\n");

    printf("=== d_segmentSize ===\n");
    for (unsigned int i = 0; i < numCoarseBins; i++)
        printf ("%u ", h_segmentSize[i]);
    printf ("\n\n");


    printf("=== d_useAtomicAdd ===\n");
    for (unsigned int i = 0; i < numCoarseBins; i++)
        printf ("%u ", h_useAtomicAdd[i]);
    printf ("\n\n");
*/

    checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));
    
    unsigned int sharedMemoryStride = blockSize + 1;
    unsigned int sharedMemorySize = sizeof(unsigned int) * numFineBins * sharedMemoryStride;

//    printf("numBins:%u, numFineBins: %u, numCoarseBins:%u, Shared Memory Requested:%zu \n", numBins, numFineBins, numCoarseBins, sharedMemorySize);

    computeFineGrainedHistograms<<<numCoarseBins, blockSize, sharedMemorySize>>>
            (d_vals, 
            d_startOfSegment, 
            d_segmentSize, 
            d_useAtomicAdd,
            d_histo, 
            numFineBins,
            numBins,
            numElems,
            numCoarseBins,
            sharedMemoryStride);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
/*
    unsigned int h_histo[numBins];
    checkCudaErrors(cudaMemcpy(h_histo, d_histo, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));
    printf("=== d_histo ===\n");
    for (unsigned int i = 0; i < numBins; i++)
        printf ("%u ", h_histo[i]);
    printf ("\n\n");
*/

    //if you want to use/launch more than one kernel,
    //feel free

    checkCudaErrors(cudaFree(d_compactedBoundaries));
    checkCudaErrors(cudaFree(d_coarseBinBoundaries));
    checkCudaErrors(cudaFree(d_coarseBinSizes));
    checkCudaErrors(cudaFree(d_numBlocksPerCoarseBin));
    checkCudaErrors(cudaFree(d_numBlocksPerCoarseBinPrefixSum));
    checkCudaErrors(cudaFree(d_startOfSegment));
    checkCudaErrors(cudaFree(d_segmentSize));
    checkCudaErrors(cudaFree(d_useAtomicAdd));
}
