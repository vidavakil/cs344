// @ 2025 Vida Vakilotojar

//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>

#include <thrust/sort.h>

#include "timer.h"
#if defined(_WIN16) || defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#else
#include <sys/time.h>
#endif


#define warpedSize 32
#define numThreadsPerSM 1024
#define numThreadsPerBlock 256  // This value seems to work best for blockDim.x, not 1024!

__global__
void computeLowestBlockPrefixSum(      unsigned int* const d_inputVals,
                                       unsigned int* const d_zerosPrefixSum,
                                       unsigned int* const d_onesPrefixSum,
                                 const size_t              numElems,
                                       unsigned int        bitPosition)
{
    extern __shared__ unsigned int s_prefixSum[];

    // each of the prefixSum buffers is warpedSize * (warpedSize + 1) * 2
    unsigned int* s_zerosPrefixSum = s_prefixSum;
    unsigned int* s_onesPrefixSum = &s_prefixSum[warpedSize * (warpedSize + 1) * 2];

    unsigned int blockSize = blockDim.x * 2; // each block covers twice as many elements
    unsigned int threadIndex = threadIdx.x * 2;

    unsigned int pos_1d = blockIdx.x * blockSize + threadIndex;
    unsigned int localBlockSize = (((blockIdx.x + 1) * blockSize) <= numElems)
                                 ? blockSize 
                                 : (numElems % (blockSize));         

    unsigned int bitPositionIsZero;
    unsigned int s_address;
    unsigned int s_otherAddress;

    s_address = (threadIndex / warpedSize) * (warpedSize + 1) + threadIndex % warpedSize;

    if (pos_1d < numElems) 
    {
        bitPositionIsZero = ((d_inputVals[pos_1d] & (1 << bitPosition)) == 0) ? 1 : 0;
        s_zerosPrefixSum[s_address] = bitPositionIsZero;
        s_onesPrefixSum[s_address] = 1 - bitPositionIsZero;
    }

    if (pos_1d + 1 < numElems) {
        s_otherAddress = ((threadIndex + 1) / warpedSize) * (warpedSize + 1) + (threadIndex + 1) % warpedSize;
        bitPositionIsZero = ((d_inputVals[pos_1d + 1] & (1 << bitPosition)) == 0) ? 1 : 0;
        s_zerosPrefixSum[s_otherAddress] = bitPositionIsZero;
        s_onesPrefixSum[s_otherAddress] = 1 - bitPositionIsZero;
    }
    __syncthreads();

    int rid = localBlockSize - 1 - threadIndex;
    s_address = (rid / warpedSize) * (warpedSize + 1) + rid % warpedSize;

    unsigned int max_step = 0;
    // The reduction phase of Blelloch algorithm
    for (unsigned int step = 1; step < localBlockSize; step *= 2) {
        max_step = step;
        if ((threadIndex % (step * 2) == 0) && (threadIndex + (int)step < (int)localBlockSize)) {
            s_otherAddress = ((rid - step)/ warpedSize) * (warpedSize + 1) + (rid - step) % warpedSize; 
            s_zerosPrefixSum[s_address] += s_zerosPrefixSum[s_otherAddress];
            s_onesPrefixSum[s_address] += s_onesPrefixSum[s_otherAddress];
        }
        __syncthreads();
    }

    unsigned int totalZeros = 0, totalOnes = 0;

    // The sweep phase of Blelloch algorithm

    // Before overriding the last element of the counts array,
    // store it away.
    if (threadIndex == 0) {
        totalZeros = s_zerosPrefixSum[s_address];
        s_zerosPrefixSum[s_address] = 0;
        totalOnes = s_onesPrefixSum[s_address];
        s_onesPrefixSum[s_address] = 0;
    }
    __syncthreads();
    
    unsigned int temp;
    for (unsigned int step = max_step; step >= 1; step /= 2) {
        if ((threadIndex % (step * 2) == 0) && (threadIndex + (int)step < (int)localBlockSize)) {
            s_otherAddress = ((rid - step) / warpedSize) * (warpedSize + 1) + (rid - step) % warpedSize;            
            temp = s_zerosPrefixSum[s_address];
            s_zerosPrefixSum[s_address] += s_zerosPrefixSum[s_otherAddress];
            s_zerosPrefixSum[s_otherAddress] = temp;

            temp = s_onesPrefixSum[s_address];
            s_onesPrefixSum[s_address] += s_onesPrefixSum[s_otherAddress];
            s_onesPrefixSum[s_otherAddress] = temp;
        }  
        __syncthreads();
    }

    // Store the totals in the first element of the array. These special
    // elements constitute the histogram of zeros and ones in each thread
    // block.
    if (threadIndex == 0) 
    {
        s_zerosPrefixSum[0] = totalZeros;
        s_onesPrefixSum[0] = totalOnes;
    }
    __syncthreads();

    s_address = (threadIndex / warpedSize) * (warpedSize + 1) + threadIndex % warpedSize;
    if (pos_1d < numElems) 
    {
        d_zerosPrefixSum[pos_1d] = s_zerosPrefixSum[s_address];
        d_onesPrefixSum[pos_1d] = s_onesPrefixSum[s_address];
    }
    if (pos_1d + 1 < numElems) {
        s_otherAddress = ((threadIndex + 1) / warpedSize) * (warpedSize + 1) + (threadIndex + 1) % warpedSize;
        d_zerosPrefixSum[pos_1d + 1] = s_zerosPrefixSum[s_otherAddress];
        d_onesPrefixSum[pos_1d + 1] = s_onesPrefixSum[s_otherAddress];
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
    unsigned int* s_onesPrefixSum = &s_prefixSum[warpedSize * (warpedSize + 1) * 2];

    unsigned int blockSize = blockDim.x * 2; // each block covers twice as many elements
    unsigned int threadIndex = threadIdx.x * 2;

    unsigned int localBlockSize = (((blockIdx.x + 1) * blockSize) <= numElems) 
                                  ? blockSize 
                                  : (numElems % blockSize);

    unsigned int pos_1d = (blockSize * blockIdx.x + threadIndex) * stride;

    unsigned int s_address;
    unsigned int s_otherAddress;

    s_address = (threadIndex / warpedSize) * (warpedSize + 1) + threadIndex % warpedSize;

    if (threadIndex < localBlockSize) {
        s_zerosPrefixSum[s_address] = d_zerosPrefixSum[pos_1d];
        s_onesPrefixSum[s_address] = d_onesPrefixSum[pos_1d];
    }

    if (threadIndex + 1 < localBlockSize) {
        s_otherAddress = ((threadIndex + 1) / warpedSize) * (warpedSize + 1) + (threadIndex + 1) % warpedSize;
        s_zerosPrefixSum[s_otherAddress] = d_zerosPrefixSum[pos_1d + stride];
        s_onesPrefixSum[s_otherAddress] = d_onesPrefixSum[pos_1d + stride];
    }

    __syncthreads();

    int rid = localBlockSize - 1 - threadIndex;  // could be negative
    s_address = (rid / warpedSize) * (warpedSize + 1) + rid % warpedSize;
 
    unsigned int max_step = 0;
    // The reduction phase of Blelloch algorithm
    for (unsigned int step = 1; step < localBlockSize; step *= 2) {
        max_step = step;
        if ((threadIndex % (step * 2) == 0) && (threadIndex + (int)step < (int)localBlockSize)) {
            s_otherAddress = ((rid - step) / warpedSize) * (warpedSize + 1) + (rid - step) % warpedSize;
            s_zerosPrefixSum[s_address] += s_zerosPrefixSum[s_otherAddress];
            s_onesPrefixSum[s_address] += s_onesPrefixSum[s_otherAddress];
        }
        __syncthreads();
    }

    // The sweep phase of Blelloch algorithm

    // Before overriding the last element of the counts array,
    // store it away.
    unsigned int totalZeros = 0;
    unsigned int totalOnes = 0;
    if (threadIndex == 0) {
        totalZeros = s_zerosPrefixSum[s_address];
        totalOnes = s_onesPrefixSum[s_address];
        s_zerosPrefixSum[s_address] = 0;
        s_onesPrefixSum[s_address] = 0;
    }

    __syncthreads();

    unsigned int temp;
    for (unsigned int step = max_step; step >= 1; step /= 2) {
        if ((threadIndex % (step * 2) == 0) && (threadIndex + (int)step < (int)localBlockSize)) {
            s_otherAddress = ((rid - step) / warpedSize) * (warpedSize + 1) + (rid - step) % warpedSize;
            temp = s_zerosPrefixSum[s_address];
            s_zerosPrefixSum[s_address] += s_zerosPrefixSum[s_otherAddress];
            s_zerosPrefixSum[s_otherAddress] = temp;

            temp = s_onesPrefixSum[s_address];
            s_onesPrefixSum[s_address] += s_onesPrefixSum[s_otherAddress];
            s_onesPrefixSum[s_otherAddress] = temp;
        }  
        __syncthreads();
    }

    // Store totalZeros in the first element of the arrays
    if (threadIndex == 0) {
       s_zerosPrefixSum[0] = totalZeros;
       s_onesPrefixSum[0] = totalOnes;
    }
    __syncthreads();

    s_address = (threadIndex / warpedSize) * (warpedSize + 1) + threadIndex % warpedSize;
    if (threadIndex < localBlockSize) {
        d_zerosPrefixSum[pos_1d] = s_zerosPrefixSum[s_address];
        d_onesPrefixSum[pos_1d] = s_onesPrefixSum[s_address];
    }

    if (threadIndex + 1 < localBlockSize) {
        s_otherAddress = ((threadIndex + 1) / warpedSize) * (warpedSize + 1) + (threadIndex + 1) % warpedSize;
        d_zerosPrefixSum[pos_1d + stride] = s_zerosPrefixSum[s_otherAddress];
        d_onesPrefixSum[pos_1d + stride] = s_onesPrefixSum[s_otherAddress];
    }    
}

__global__
void moveElements(const unsigned int* const d_inputVals,
                        unsigned int* const d_outputVals, 
                  const unsigned int* const d_zerosPrefixSum, 
                  const unsigned int* const d_onesPrefixSum, 
                  const size_t              numElems, 
                  const size_t              bitPosition,
                  const unsigned int        stride,
                  const unsigned int        strideLevels,
                  const unsigned int        startStridalAddress) // stride ^ strideLevels
{
    unsigned int blockSize = blockDim.x * 2; // each block covers twice as many elements
    unsigned int threadIndex = threadIdx.x * 2;

    unsigned int pos_1d;
    unsigned int bitPositionIsZero;
    unsigned int addressTobeWrittenTo;
    unsigned int stridalAddress;
    const unsigned int* lookUpAddress;
    unsigned int  roundedAddress;

    assert(blockDim.x == stride/2);
    assert(startStridalAddress != 0);

    for (unsigned int i = 0; i <= 1; i++) {
        pos_1d = blockSize * blockIdx.x + threadIndex + i;
        if (pos_1d < numElems) {
            bitPositionIsZero = ((d_inputVals[pos_1d] & (1 << bitPosition)) == 0);
            addressTobeWrittenTo = 0;
            stridalAddress = startStridalAddress;
            lookUpAddress = d_zerosPrefixSum;
            roundedAddress = 0;

            if (not bitPositionIsZero) {
                addressTobeWrittenTo = d_zerosPrefixSum[0];
                lookUpAddress = d_onesPrefixSum;
            }

            if (pos_1d > 0 and pos_1d < numElems and stridalAddress >= 1) {
                while (stridalAddress >= 1 && pos_1d % stridalAddress != 0) {
                    roundedAddress =  stridalAddress * (pos_1d / stridalAddress);
                    if (roundedAddress != 0 && roundedAddress % (stridalAddress * stride) != 0)
                        addressTobeWrittenTo += lookUpAddress[roundedAddress];
                    stridalAddress /= stride;
                    assert(addressTobeWrittenTo < numElems);
                }
                assert(addressTobeWrittenTo < numElems);
                addressTobeWrittenTo += lookUpAddress[pos_1d];
            }
            d_outputVals[addressTobeWrittenTo] = d_inputVals[pos_1d];
        }
    }
}

__global__
void compareArrays(const unsigned int* const d_outputVals, 
                   const unsigned int* const d_thrustVals, 
                   const unsigned int        numElems)
{
    unsigned int pos_1d = blockDim.x * blockIdx.x + threadIdx.x;

    if (pos_1d < numElems) 
        assert(d_outputVals[pos_1d] == d_thrustVals[pos_1d]);
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
               const size_t numElems,
               unsigned int radixLow,
               unsigned int radixHigh)
{ 

    GpuTimer timer;
    int err;

    timer.Start();       

    assert (radixHigh >= radixLow);
    unsigned int numIterations = radixHigh - radixLow + 1;

    dim3 blockDim(numThreadsPerBlock); // warpedSize * warpedSize
    dim3 gridDim((numElems + blockDim.x * 2 - 1) / (blockDim.x * 2));
    dim3 gridOverGridDim((gridDim.x + blockDim.x * 2 - 1) / (blockDim.x *2));

    unsigned int* d_outputVals;
    checkCudaErrors(cudaMalloc(&d_outputVals, sizeof(unsigned int) * numElems));

//    printf("radixSort: numElems = %lu, gridDim.x = %u, blockDim.x = %u, gridOverGridDim = %u\n", 
//           numElems, gridDim.x, blockDim.x, gridOverGridDim.x);


    // Allocate memory for the scatter addresses for zeros and ones
    unsigned int *d_zerosHistogram, *d_onesHistogram;

    checkCudaErrors(cudaMalloc(&d_zerosHistogram, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMalloc(&d_onesHistogram, sizeof(unsigned int) * numElems));

    unsigned int stride = numThreadsPerBlock * 2;

    for (unsigned int i = 0; i < numIterations; i++)
    {
        if (i % 2 == 0) 
            // shared memory size: each 32 items are offset by 1 to avoid bank conflicts
            // we need double warpedSize * (warpedSize + 1), for the zerosPrefix and onesPrefix,
            // and we need to double that to cover double memory size, as at the lowest
            // level of the scan, we want to keep all threads busy, instead of having only
            // half of them doing work.
            computeLowestBlockPrefixSum<<<gridDim, blockDim, sizeof(unsigned int) * 2 * warpedSize * (warpedSize + 1) * 2>>>( 
                                                    d_inputVals, 
                                                    d_zerosHistogram, 
                                                    d_onesHistogram, 
                                                    numElems, 
                                                    i + radixLow);
        else
            computeLowestBlockPrefixSum<<<gridDim, blockDim, sizeof(unsigned int) * 2 * warpedSize * (warpedSize + 1) * 2>>>(
                                                    d_outputVals, 
                                                    d_zerosHistogram, 
                                                    d_onesHistogram, 
                                                    numElems, 
                                                    i + radixLow);

        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        unsigned int strideLevels = 0;
        unsigned long currentStride = stride;
        unsigned int numBlocks = 0;

        // Step 2: Compute prefix sum (CDF) of the histogram
        while (numElems > currentStride) {
            gridOverGridDim.x = (numElems + currentStride * stride - 1) / (currentStride * stride);
            numBlocks = (numElems + currentStride - 1) / (currentStride);
            computeStridedPrefixSum<<<gridOverGridDim, blockDim, sizeof(unsigned int) * 2 * warpedSize * (warpedSize + 1) * 2>>>(
                                                            d_zerosHistogram, 
                                                            d_onesHistogram, 
                                                            numBlocks, 
                                                            currentStride);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
            strideLevels++;
            currentStride *= stride;
        }

        // Step 3: Combine the results of the first two kernels to compute the location each
        // element must be moved to and then move them.
        // Step 2: Compute prefix sum (CDF) of the histogram

        if (i % 2 == 0) 
            moveElements<<<gridDim, blockDim>>>(d_inputVals,
                                                d_outputVals,
                                                d_zerosHistogram, 
                                                d_onesHistogram, 
                                                numElems, 
                                                i + radixLow, 
                                                stride, 
                                                strideLevels, 
                                                currentStride/stride);
        else
            moveElements<<<gridDim, blockDim>>>(d_outputVals,
                                                d_inputVals,
                                                d_zerosHistogram, 
                                                d_onesHistogram, 
                                                numElems, 
                                                i + radixLow, 
                                                stride, 
                                                strideLevels, 
                                                currentStride/stride);

        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }

    if (numIterations % 2 == 1)
        checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals,
                               numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaFree(d_zerosHistogram));
    checkCudaErrors(cudaFree(d_onesHistogram));
    checkCudaErrors(cudaFree(d_outputVals));

    timer.Stop();
    err = printf("Radix sort ran in: %f msecs.\n", timer.Elapsed());


    // Now compare the results to what CUDA Thrust will do. Do this by checking whether
    // sorting the result by thrust will make it anymore sorted.

    if (radixLow == 0 && radixHigh == 31) {
        unsigned int* d_thrustVals;

        checkCudaErrors(cudaMalloc(&d_thrustVals, sizeof(unsigned int) * numElems));
        checkCudaErrors(cudaMemcpy(d_thrustVals, d_inputVals,
                               numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

        timer.Start();    
        thrust::device_ptr<unsigned int> d_a(d_thrustVals);
        thrust::sort(d_a, d_a + numElems);
        timer.Stop();
        err = printf("Thrust sort on device data ran in: %f msecs.\n", timer.Elapsed());

        gridDim.x = (numElems + blockDim.x - 1) / blockDim.x;

        compareArrays<<<gridDim, blockDim>>>(d_inputVals, d_thrustVals, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaFree(d_thrustVals));
    }

    // Notes: there are things that we can do to speed up the sort.
    // For example, we can find the largets element in the input array by
    // a Reduce Max operation, or by a simpler operation that just captures
    // the largest msb bit, and then use that to reduce the depth of 
    // radix sort.
    // Another observation is that whenever we compute a Scan, at each 
    // level of scan, half of the threads (with respect to the previous level) 
    // will become idle, and that is a waste. To fully utilize them, we can
    // have them start a separate Scan over data of half the size.
    // So, if we start with using 1024 threads in a block to do the first level
    // of scan over 2048 elements, in the next level, only 512 of the 1024 threads
    // would be needed to work on the 2048 elements, and the other 512 can start 
    // working on a Scan of a 1024 element array. Thus we will be utilizing 100% of the
    // threads in the threadblock. 
    // In the next level, only 256 of the original 1024 threads are needed to work on 
    // the 2048 element array, and out of 512 threads that were working on the 1024 Scan, 
    // only 256 are needed. We can then start a new Scan over a 512 array using 256 threads. 
    // Thus, we will be utilizing 3 * 256 = %75 of the 1024 threads in the threadblock 
    // and working on 3 scans of sizes 2048, 1024, 512 elements. We can even do better than
    // that, and start 2 scans over two arrays of size 512 each, and fully utilize all threads.
    // In that case, we will have separate scans of the form: 2048, 1024, 512, 512.
    // In the next level, we would only need 128 threads each for the 4 scans that
    // already exists, and we can use 4 * 128 threads of the idle ones to start the scan
    // of 4 chunks of 256 each.
    // Next level would be 64 threads each working on chunks of 2048, 1024, 2 * 512, and
    // 4 * 256. But we have 512 idle threads and we can have 8 * 64 of them start scanning
    // 8 chuncks of 128 each.
    // Finally, at next level we will have 32 threads each (a warp) working on 
    // chunks of 2048, 1024, 2 * 512, 4 * 256, 8 * 128, and we can utilize the remaining
    // 16 * 32 threads to start scanning 16 chunks of 64 each.
    // At this point we stop trying to utilize idle threads, because we simply won't
    // be able to collect idle threads into thread warps, and instead we have to continue 
    // scanning of the above chunks using the existing warps and at each level
    // of the remaining log 32 = 5 steps, half of the threads of each warp will
    // become idle anyway. 
    // After we create the above 1 + 1 + 2 + 4 + 8 + 16 = 32 blocks of sizes
    // 2048, 1024, 512, 256, 128, 64, we have to run a separate Scan that
    // computes a Scan over these 32 items for the whole 7 * 1024 chunk. This
    // other kernel can be just 32 threads (a single warp), and each block of it can
    // take care of a chunk of 7000 elements.
    // We then go to the next level of Scan hierarchy, when we compute a Scan
    // over the totals of each of the 7000 blocks. 
    // Once we have recursively computed all levels of scan, each with a stride
    // of 7 * 1024 = 7000, we have all the global and relative addresses 
    // computed, and all we need to do to compute the scatter address of each
    // element is to recursively zoom in on the hierarchical scan and find
    // which 7000 chunk we fall into (address / currentStride, where currentStride is a 
    // power of 7000), and compute which block within the 7000 chunk we fall into,
    // and for that we have to see which of the 1024 blocks within the 7000 chunk we fall into;
    // if it is the first or second 1024, we have to use the first total. For any other one,
    // we find the start of the 1024 boundary (of 1, 2, 4, 8, 16 chunks), and depending on
    // which one we fall into, we have to further compute the relative address within the
    // 1024, by another division, etc.
    // This last step is a bit complicated, but doable.
    // The 7000 element scan itself is also tricky, and best thing to do is probably to 
    // unroll its code into a huge mess, making sure that we never have code divergence
    // within the same thread block.

    // Another way to speed up the code is to better leverage the GPU memory. 
    // For example, instead of doing radix-sorts per bit, we can do per digit.
    // For n-bit digits when we do radix-sort, we need 2^n auxiliary PrefixSum
    // arrays instead of two (i.e., instead of the zeros/ones array in the above code).
    // So, we can increase n to the largest that the GPU memory can accomodate.
    // We may then face a challenge for shared memory which also needs to be 
    // increased per thread block, but for that we can reduce the number of threads
    // in a thread block, and thus the size of data that a threadblock performs a scan on.

    // Finally, another way to speed up the sort is to chose a different stride for 
    // each level of the prefixSum computation (Kernel Launches with currentStride).
    // We want the last level of the launch fully utilize all the threads in the GPU,
    // instead of requiring only a few threads and wasting all the other thread resources.
    // With that, we have two constraints, one is the original size of the data, and 
    // the other one is this last level prefix sum using all the threads (no more, no less).
    // If we can solve this problem, then for each hierarchical level of the scan we can
    // choose a different stride. When computing the scatter addresses for moveElements(),
    // we would have to take into account the different strides of each level, to carefully
    // compute the scatter address of each data item, as we zoom in to the levels of the scan.

    // With the above three techniques, we would be utilizing the resources available to us
    // in the GPU; i.e., the threads/SMs/Warps/Memory/Shared-Memory as best as possible,
    // at every stage.

    // It's hard to beleive that any super-fast radix-sort algorithm is doing something
    // outside the above techniques.
}
