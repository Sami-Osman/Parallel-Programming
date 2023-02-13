
// Function decleration.
double sortArray(int *arr, int *aux, int count, int device, int cnk, int SHEMEM_flag);
__global__ void mergeSortKernel(int* arr, int* aux, unsigned int blockSize, const unsigned int last);
__global__ void mergeSortKernel_SHEMEM(int* arr1, int* aux, unsigned int blockSize, const unsigned int last);

__global__ void mergeSortKernel(int* arr, int* aux, unsigned int blockSize, const unsigned int last){
  // Calculate the index of the current thread
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  // Check if the current thread index is within the bounds of the array
  if(x <= last){
    // Calculate the start index for the current block
    int start = blockSize * x;
    // Check if the start index is still within the bounds of the array
     if(start<=last){
       // Calculate the end index for the current block
       int end = start + blockSize - 1;
       // If the end index exceeds the bounds of the array, adjust it to last index.
       if(end>last){end=last;}
       // Calculate the mid index for the current block
       int mid = start + (blockSize / 2) - 1;
       // Initialize the left and right pointers
       int l = start, r = mid + 1, i = start;
      
      // Compare and merge the two halves of the current block
       while (l <= mid && r <= end) {
         if (arr[l] <= arr[r]) {
           aux[i++] = arr[l++];
         }
         else {
           aux[i++] = arr[r++];
         }
       }
       // Copy any remaining elements from the left half
       while (l <= mid) { aux[i++] = arr[l++]; }
       // Copy any remaining elements from the right half
       while (r <= end) { aux[i++] = arr[r++]; }
      // Copy the sorted elements back to the original array
      for (i = start; i <= end; i++) {
        arr[i] = aux[i];
        }
     }
  }
}

__global__ void mergeSortKernel_SHEMEM(int* arr1, int* aux, unsigned int blockSize, const unsigned int last){
  extern __shared__ int arr[];
  // Calculate the index of the current thread
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Check if the current thread index is within the bounds of the array
  if(x <= last){
    // Calculate the start index for the current block
    int start = blockSize * x;
    // Check if the start index is still within the bounds of the array
     if(start<=last){
       // Calculate the end index for the current block
       int end = start + blockSize - 1;
         for(int i=start; i<=end; i++){
            arr[i] = arr1[i];
          }
          __syncthreads();
       // If the end index exceeds the bounds of the array, adjust it to last index.
       if(end>last){end=last;}
       // Calculate the mid index for the current block
       int mid = start + (blockSize / 2) - 1;
       // Initialize the left and right pointers
       int l = start, r = mid + 1, i = start;
      
      // Compare and merge the two halves of the current block
       while (l <= mid && r <= end) {
         if (arr[l] <= arr[r]) {
           aux[i++] = arr[l++];
         }
         else {
           aux[i++] = arr[r++];
         }
       }

       // Copy any remaining elements from the left half
       while (l <= mid) { aux[i++] = arr[l++]; }
       // Copy any remaining elements from the right half
       while (r <= end) { aux[i++] = arr[r++]; }
      __syncthreads();
      // Copy the sorted elements back to the original array
      for (i = start; i <= end; i++) {
        arr1[i] = aux[i];
        }
     }
  }
}


double sortArray(int *arr, int *aux, int count, int device, int cnk, int SHEMEM_flag) {
  double tstart, tstop, exec_time = 0.0;
  unsigned int kernelBlockSize = 10;
  int *arr1;
  int *aux1;
  const unsigned int size = count * sizeof(int);// size of the input array in bytes
  unsigned int threadCount;// number of threads to launch for each iteration
  cudaSetDevice(device);
  cudaMalloc((void **)&arr1, size);
  cudaMalloc((void **)&aux1, size);
  cudaMemcpy(arr1, arr, size, cudaMemcpyHostToDevice);
  cudaMemcpy(aux1, aux, size, cudaMemcpyHostToDevice);

  for (int chunk = cnk; chunk < 2 * count; chunk *= 2) {
      // calculate the number of threads to launch based on:
      // blockSize and kernelBlockSize
      threadCount = (count / (chunk) /kernelBlockSize)+1; //added one just to make sure we have enough threads to solve the problem.

      if(threadCount>1022){
        while(threadCount>1023){
          ++kernelBlockSize;
          threadCount = (count / (chunk) /kernelBlockSize)+1;
          }
      }
      // Define the number of blocks in the grid
      dim3 my_gridDim(kernelBlockSize);
      // Define the number of threads in each block
      dim3 my_BlockDim(threadCount);
      if(SHEMEM_flag){
          tstart = gettime();
          // Launch the kernel with the calculated block size and number of threads
          mergeSortKernel_SHEMEM<<<my_gridDim, my_BlockDim, size*2>>>(arr1, aux1, chunk, count-1);
        }
        else{
          tstart = gettime();
          // Launch the kernel with the calculated block size and number of threads
          mergeSortKernel<<<my_gridDim, my_BlockDim>>>(arr1, aux1, chunk, count-1);
        }
      cudaDeviceSynchronize();
      tstop = gettime();
      exec_time= exec_time + (tstop - tstart);
  }

  cudaMemcpy(arr, arr1, size, cudaMemcpyDeviceToHost);
  // Free GPU buffers
  cudaFree(arr1);
  cudaFree(aux1);
  return exec_time;
}
