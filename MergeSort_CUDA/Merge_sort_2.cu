//Function decleration
cudaError_t mergeSortWithCuda(int* arr, unsigned int count, unsigned int kernelBlockSize, bool analysis_flag, bool SHEMEM_flag);

cudaError_t mergeSortWithCuda(int* arr, unsigned int count, unsigned int kernelBlockSize, bool analysis_flag, bool SHEMEM_flag)
{
    const unsigned int size = count * sizeof(int);// size of the input array in bytes
    const unsigned int last = count - 1;// last index of the array
    unsigned int threadCount;// number of threads to launch for each iteration
    int* dev_arr = 0;//pointer to GPU buffer for the main array
    int* dev_aux = 0;// pointer to GPU buffer for the auxiliary array
    cudaError_t cudaStatus;
    double tstart, tstop, tnoshared; //tshared;
    tstart = tstop = tnoshared = 0;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Allocate GPU buffers for two vectors (main and aux array).
    cudaStatus = cudaMalloc((void**)&dev_arr, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void**)&dev_aux, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_arr);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_arr, arr, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    // chunk is Number of array elements to be sorted in single thread.
    // initially pair of elements from the array are mergeSorted in a single thread
    unsigned int chunk;

    for (chunk = 2; chunk < 2 * count; chunk *= 2) {
      // calculate the number of threads to launch based on:
      // blockSize and kernelBlockSize
      threadCount = (count / (chunk) /kernelBlockSize)+1; //added one just to make sure we have enough threads to solve the problem.

      if(threadCount>1023){
        if(analysis_flag==false){
          printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
          printf("!! Max Thread per Block is 1024        !!\n");
          printf("!! Can't fit Array size in a %d Blocks !!\n", kernelBlockSize);
          printf("!! Adjusting # Blocks per Grid ...... !!\n");
        }
        
  
        // if the number of threads is greater than the maximum limit (1024), adjust the block size
        while(threadCount>1023){
          ++kernelBlockSize;
          threadCount = (count / (chunk) /kernelBlockSize)+1;
          }
        if(analysis_flag==false){
          printf("!! Block size set to: %d              !!\n", kernelBlockSize);
          printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
          }
        }

        // Define the number of blocks in the grid
        dim3 my_gridDim(kernelBlockSize);
        // Define the number of threads in each block
        dim3 my_BlockDim(threadCount);

        //Displaying the first configuration of the Threads in the GPU
        if(chunk==2 && analysis_flag==false){
          printf("First loop Configuration\n");
          printf("Block/Grid: %d \n", kernelBlockSize);
          printf("Thread/Block: %d \n", threadCount);
          printf("Total threads: %d \n", threadCount * kernelBlockSize);
          printf("T_Time: ");
        }
        if(SHEMEM_flag){
          tstart = gettime();
          // Launch the kernel with the calculated block size and number of threads
          mergeSortKernel_SHEMEM<<<my_gridDim, my_BlockDim, size*2>>>(dev_arr, dev_aux, chunk, last);
        }
        else{
          tstart = gettime();
          // Launch the kernel with the calculated block size and number of threads
          mergeSortKernel<<<my_gridDim, my_BlockDim>>>(dev_arr, dev_aux, chunk, last);
        }
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetLastError failed!");
            return cudaStatus;
        }
        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        cudaStatus = cudaDeviceSynchronize();
        tstop = gettime();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize failed!");
            return cudaStatus;
        }
        tnoshared += (tstop - tstart);
    }
    printf("%f", tnoshared);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(arr, dev_arr, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    // Free GPU buffers
    cudaFree(dev_arr);
    cudaFree(dev_aux);
    return cudaSuccess;
}