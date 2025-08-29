//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>

void findBoundaryBox(      unsigned char* const d_inputVals,
                     const size_t              numColumns,
                     const size_t              numRows,
                           unsigned int*       boundaryBox); // array of size 4: x_min, y_min, x_max, y_max

__global__
void preprocessSourceImage(const uchar4*        const  d_sourceImg,
                                 uchar4*        const  d_destImg, 
                                 unsigned char* const  d_sourceMask, 
                                 float*         const  d_blendedRed, 
                                 float*         const  d_blendedGreen, 
                                 float*         const  d_blendedBlue,
                                 float*         const  d_diffRed,
                                 float*         const  d_diffGreen,
                                 float*         const  d_diffBlue,
                           const size_t                numRowsSource, 
                           const size_t                numColsSource)
{
   // coordinates in the grid
   int columnId = blockIdx.x * blockDim.x + threadIdx.x;
   int rowId = blockIdx.y * blockDim.y + threadIdx.y;

   // linear address in the image
   unsigned int pos_1d = rowId * numColsSource + columnId; // address in image

   uchar4 pixel, neighborPixel;
   const unsigned char white = 255;
   bool  copyFromDestination = false;
   float diffRed = 0.0, diffGreen = 0.0, diffBlue = 0.0;

   if (columnId < numColsSource and rowId < numRowsSource) {
      pixel = d_sourceImg[pos_1d];
      if (pixel.x == white && pixel.y == white && pixel.z == white) {
         d_sourceMask[pos_1d] = 0; // exterior, don't copy these pixels
         copyFromDestination = true;
      }
      else { // count and store the number of exterior neighbors
         unsigned int interiorNeigbors = 5;
         diffRed = 4.0f * pixel.x;
         diffGreen = 4.0f * pixel.y;
         diffBlue = 4.0f * pixel.z;
         if (columnId + 1 < numColsSource) { // pixel to the right
            neighborPixel = d_sourceImg[pos_1d + 1];
            if (neighborPixel.x == white && neighborPixel.y == white && neighborPixel.z == white)
               interiorNeigbors -= 1;
            diffRed -= neighborPixel.x;
            diffGreen -= neighborPixel.y;
            diffBlue -= neighborPixel.z;
         }
         if (columnId - 1 >= 0) { // pixel to the left
            neighborPixel = d_sourceImg[pos_1d - 1];
            if (neighborPixel.x == white && neighborPixel.y == white && neighborPixel.z == white)
               interiorNeigbors -= 1;
            diffRed -= neighborPixel.x;
            diffGreen -= neighborPixel.y;
            diffBlue -= neighborPixel.z;
         }
         if (rowId - 1 >= 0) { // pixel to the top
            neighborPixel = d_sourceImg[pos_1d - numColsSource];
            if (neighborPixel.x == white && neighborPixel.y == white && neighborPixel.z == white)
               interiorNeigbors -= 1;
            diffRed -= neighborPixel.x;
            diffGreen -= neighborPixel.y;
            diffBlue -= neighborPixel.z;
         }
         if (rowId + 1 < numRowsSource) { // pixel to the top
            neighborPixel = d_sourceImg[pos_1d + numColsSource];
            if (neighborPixel.x == white && neighborPixel.y == white && neighborPixel.z == white)
               interiorNeigbors -= 1;
            diffRed -= neighborPixel.x;
            diffGreen -= neighborPixel.y;
            diffBlue -= neighborPixel.z;
         }
         d_sourceMask[pos_1d] = interiorNeigbors; // number of exterior neighbors, plus 1.
         // d_sourceMask = 0 means that the pixel is exterior
         // d_sourceMask = 5 means that the pixel is interior and so are all its neighbors
         // d_sourceMask < 5 means that the pixel is at the boundary, with some exterior neighbors
         if (interiorNeigbors < 5)
            copyFromDestination = true;
      }
      if (copyFromDestination) {
         pixel = d_destImg[pos_1d];
         d_diffRed[pos_1d] = 0.0;
         d_diffGreen[pos_1d] = 0.0;
         d_diffBlue[pos_1d] = 0.0;
      }
      else {
         pixel = d_sourceImg[pos_1d];
         d_diffRed[pos_1d] = diffRed;
         d_diffGreen[pos_1d] = diffGreen;
         d_diffBlue[pos_1d] = diffBlue;
      }
      d_blendedRed[pos_1d] = pixel.x;
      d_blendedGreen[pos_1d] = pixel.y;
      d_blendedBlue[pos_1d] = pixel.z;
   }
}

__global__
void blendImage(const unsigned char* const  d_sourceMask, 
                      float*                d_blendedRed, 
                      float*                d_blendedGreen, 
                      float*                d_blendedBlue,
                      float*                d_diffRed, 
                      float*                d_diffGreen, 
                      float*                d_diffBlue,
                      unsigned int          numRowsSource, 
                      unsigned int          numColsSource, 
                      unsigned int*         boundaryBox,
                      unsigned int          sourceBufferOffset, 
                      unsigned int          destinationBufferOffset)
{
   // coordinates in the grid
   unsigned int columnId = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int rowId = blockDim.y * blockIdx.y + threadIdx.y;

   // coordinates in the image
   unsigned int pos_x = columnId + boundaryBox[0]; // x_min
   unsigned int pos_y = rowId + boundaryBox[2]; // y_min

   // position in the image data
   unsigned int pos_1d = pos_y * numColsSource + pos_x;
   float redSum, blueSum, greenSum;

   unsigned char maskValue = d_sourceMask[pos_1d];

   // for interior pixels
   if (pos_x <= boundaryBox[1] && pos_y <= boundaryBox[3]) {
      if (maskValue == 5) {
         redSum = d_diffRed[pos_1d];
         greenSum = d_diffGreen[pos_1d];
         blueSum = d_diffBlue[pos_1d];
         if (pos_x + 1 <= boundaryBox[1]) {
            redSum += d_blendedRed[sourceBufferOffset + pos_1d + 1];
            greenSum += d_blendedGreen[sourceBufferOffset + pos_1d + 1];
            blueSum += d_blendedBlue[sourceBufferOffset + pos_1d + 1];
         }
         if (pos_x - 1 >= boundaryBox[0]) {
            redSum += d_blendedRed[sourceBufferOffset + pos_1d - 1];
            greenSum += d_blendedGreen[sourceBufferOffset + pos_1d - 1];
            blueSum += d_blendedBlue[sourceBufferOffset + pos_1d - 1];
         }
         if (pos_y - 1 >= boundaryBox[2]) {
            redSum += d_blendedRed[sourceBufferOffset + pos_1d - numColsSource];
            greenSum += d_blendedGreen[sourceBufferOffset + pos_1d - numColsSource];
            blueSum += d_blendedBlue[sourceBufferOffset + pos_1d - numColsSource];
         }
         if (pos_y + 1 < boundaryBox[3]) {
            redSum += d_blendedRed[sourceBufferOffset + pos_1d + numColsSource];
            greenSum += d_blendedGreen[sourceBufferOffset + pos_1d + numColsSource];
            blueSum += d_blendedBlue[sourceBufferOffset + pos_1d + numColsSource];
         }
         redSum = min(255.0, max(0.0, redSum / 4.0f));
         greenSum = min(255.0, max(0.0, greenSum / 4.0f)); 
         blueSum = min(255.0, max(0.0, blueSum / 4.0f)); 

      } else if (maskValue >= 1) {
         redSum = d_blendedRed[sourceBufferOffset + pos_1d];
         greenSum = d_blendedGreen[sourceBufferOffset + pos_1d];
         blueSum = d_blendedBlue[sourceBufferOffset + pos_1d];
      }
      if (maskValue >= 1) {
         d_blendedRed[destinationBufferOffset + pos_1d] = redSum;
         d_blendedGreen[destinationBufferOffset + pos_1d] = greenSum;
         d_blendedBlue[destinationBufferOffset + pos_1d] = blueSum;    
      }
   }
}

__global__
void processDestinationImage(const float*  const d_blendedRed, 
                             const float*  const d_blendedGreen, 
                             const float*  const d_blendedBlue,
                                   uchar4* const d_destImg,
                             const unsigned int  sourceBufferOffset,
                             const unsigned int  numRowsSource, 
                             const unsigned int  numColsSource)
{
   // coordinates in the grid
   unsigned int columnId = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int rowId = blockDim.y * blockIdx.y + threadIdx.y;

   // position in the image data
   unsigned int pos_1d = rowId * numColsSource + columnId;

   uchar4 pixel;

   if (columnId < numColsSource && rowId < numRowsSource) {
      pixel.x = min(255, max(0, (int)d_blendedRed[sourceBufferOffset + pos_1d]));
      pixel.y = min(255, max(0, (int)d_blendedGreen[sourceBufferOffset + pos_1d]));
      pixel.z = min(255, max(0, (int)d_blendedBlue[sourceBufferOffset + pos_1d]));
      pixel.w = 255;
      d_destImg[pos_1d] = pixel;
   }
}


void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */
  
      uchar4 *d_sourceImg;
      uchar4 *d_destImg;
      uchar4 *d_blendedImg;
      unsigned char *d_sourceMask;
      float *d_blendedRed;
      float *d_blendedGreen;
      float *d_blendedBlue;
      float  *d_diffRed;
      float  *d_diffGreen;
      float  *d_diffBlue;

      unsigned int imageSize = numRowsSource * numColsSource;

      checkCudaErrors(cudaMalloc(&d_sourceImg, sizeof(uchar4) * imageSize));
      checkCudaErrors(cudaMalloc(&d_destImg, sizeof(uchar4) * imageSize));
      checkCudaErrors(cudaMalloc(&d_blendedImg, sizeof(uchar4) * imageSize));

      checkCudaErrors(cudaMalloc(&d_sourceMask, sizeof(unsigned char) * imageSize));

      checkCudaErrors(cudaMalloc(&d_blendedRed, sizeof(float) * imageSize * 2));
      checkCudaErrors(cudaMalloc(&d_blendedGreen, sizeof(float) * imageSize * 2));
      checkCudaErrors(cudaMalloc(&d_blendedBlue, sizeof(float) * imageSize * 2));
      checkCudaErrors(cudaMalloc(&d_diffRed, sizeof(float) * imageSize));
      checkCudaErrors(cudaMalloc(&d_diffGreen, sizeof(float) * imageSize));
      checkCudaErrors(cudaMalloc(&d_diffBlue, sizeof(float) * imageSize));
      
      checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, 
                                 sizeof(uchar4) * imageSize, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, 
                                 sizeof(uchar4) * imageSize, cudaMemcpyHostToDevice));

      dim3 blockDim(32, 32);
      dim3 gridDim((numColsSource + blockDim.x - 1) / blockDim.x, 
                   (numRowsSource + blockDim.y - 1) / blockDim.y);

      // Create the mask, and separate the image into 3 channels, and initialize the channels
      preprocessSourceImage<<<gridDim, blockDim>>>(d_sourceImg,
                                                   d_destImg, 
                                                   d_sourceMask, 
                                                   d_blendedRed, 
                                                   d_blendedGreen, 
                                                   d_blendedBlue,
                                                   d_diffRed, 
                                                   d_diffGreen, 
                                                   d_diffBlue,
                                                   numRowsSource, 
                                                   numColsSource);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      // find the rectangular boundary around the mask, by finding the mask bits with the
      // smallest and largest x and y coordinates. We use these results to limit the 
      // area that we would be processing in the next steps of the algorithm.
      // To do this, do the following:
      // 1- Per row of the image, find the two pixels in the row with the minimum and maximum
      //    x coordinates that are part of the mask. This requires identifying elements in the
      //    row that are part of the mask (using a predicate), and then computing Reduce-Min and 
      //    Reduce-Max. The first elements of the min/max reductions will
      //    be the x coordinate of the min/max masked pixel.
      // 2- Now that we know for each row, whether or not there is any masked bits, and if so
      //    what are the min and max x coordinates of such pixels, we need to find the 
      //    minimum of all the min x coordinates, which would be the x coordinate of the left edge 
      //    of the boundary box of the mask. We also need to find the maximum of all the max x
      //    coordinates, which would be the x coordinate of the right edge of the boundary
      //    box of the mask. For the min of mins, we do another Reduce-Min of the min results.
      //    For the max of max's, we do an Reduce-Max of the max results.
      // 3- Also, we need to find the smallest row number that had any masked
      //    pixels, which would be the top edge of the boundary box. And also the largest
      //    row number with any masked pixels, which would be the bpttom edge of the boundary
      //    box. To compute the top and bottom edges of the boundary box, we would perform 
      //    an exclusive-PrefixSum on the first element of the Reduce-Max results over rows.
      //    The predicate we use is whether or not that first element is non-zero (remember
      //    tha the first element of the per-row Reduce-Max contains x_max of the row, or 
      //    zero. We then do a compaction of rows using this vertical
      //    prefix sum, where each row with non-zero masked pixels writes its row number 
      //    into a compaction array. The first element of this compaction array would then
      //    be the row number of the top edge. The last element of this compaction array
      //    (whose index in the vertical compaction array is the total sum of the vertical
      //    Prefix Sum), will be the bottom edge of the boundary box. 

      unsigned int boundaryBox[4];
      unsigned int* d_boundaryBox;

      findBoundaryBox(d_sourceMask, numColsSource, numRowsSource, boundaryBox);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      checkCudaErrors(cudaMalloc(&d_boundaryBox, sizeof(unsigned int) * 4));   
      checkCudaErrors(cudaMemcpy(d_boundaryBox, boundaryBox, 
                                 sizeof(unsigned int) * 4, cudaMemcpyHostToDevice));

      // Now, we need to perform the main task 800 times.
      // We pass the boundary box to the kernels, so that threads focus on the 
      // bounding box. We also limit the grid to the bounding box.
/*
      1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]
*/

      // Sum2 is fixed, so we can compute it just once
      // Sum1 contributions by border neighbors are also fixed, and equal to DestinationImg[neighbor]
      // We only change the interior pixels (not the border pixels????). 
      // But their updates are influenced by both interior and border neighbors.

      unsigned int maskWidth = boundaryBox[1] - boundaryBox[0];
      unsigned int maskHeight = boundaryBox[3] - boundaryBox[2];

      blockDim.x = 32;
      blockDim.y = 32;

      gridDim.x = (maskWidth + blockDim.x - 1) / blockDim.x;
      gridDim.y = (maskHeight + blockDim.y - 1) / blockDim.y;
      unsigned int sourceBufferOffset, destinationBufferOffset;

      for (int i = 0; i < 800; i++) {
         if (i % 2 == 0) {
            sourceBufferOffset = 0;
            destinationBufferOffset = imageSize;
         } else {
            sourceBufferOffset = imageSize;
            destinationBufferOffset = 0;
         }
         blendImage<<<gridDim, blockDim>>>(d_sourceMask, 
                                           d_blendedRed, 
                                           d_blendedGreen, 
                                           d_blendedBlue,
                                           d_diffRed, 
                                           d_diffGreen, 
                                           d_diffBlue,
                                           numRowsSource, 
                                           numColsSource, 
                                           d_boundaryBox,
                                           sourceBufferOffset, 
                                           destinationBufferOffset);
         cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      }


      gridDim.x = (numColsSource + blockDim.x - 1) / blockDim.x;
      gridDim.y = (numRowsSource + blockDim.y - 1) / blockDim.y;
      processDestinationImage<<<gridDim, blockDim>>>(d_blendedRed, 
                                                     d_blendedGreen, 
                                                     d_blendedBlue,
                                                     d_blendedImg,
                                                     destinationBufferOffset,
                                                     numRowsSource, 
                                                     numColsSource);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, 
                                 sizeof(uchar4) * imageSize, cudaMemcpyDeviceToHost));

      checkCudaErrors(cudaFree(d_sourceImg));
      checkCudaErrors(cudaFree(d_destImg));
      checkCudaErrors(cudaFree(d_blendedImg));
      checkCudaErrors(cudaFree(d_sourceMask));
      checkCudaErrors(cudaFree(d_blendedRed));
      checkCudaErrors(cudaFree(d_blendedGreen));
      checkCudaErrors(cudaFree(d_blendedBlue));
      checkCudaErrors(cudaFree(d_diffRed));
      checkCudaErrors(cudaFree(d_diffGreen));
      checkCudaErrors(cudaFree(d_diffBlue));
      checkCudaErrors(cudaFree(d_boundaryBox));
}
