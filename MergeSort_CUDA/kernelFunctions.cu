
// Function decleration.
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