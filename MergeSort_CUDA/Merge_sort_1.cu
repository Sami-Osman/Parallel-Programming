
#include "utilities.cu" //utilities.cu refers to different function I used, which are not directly related to the GPU task.
                        // It also includes all the .h files i used in my project.
#include "kernelFunctions.cu" //kernelFuntions.cu refers to all my __global__ kernel function definitions.
#include "Merge_sort_2.cu" //Merge_sort_2.cu reference to functions related to merge sort code.

#define OK 1
#define EXPECTATION_ERROR 1
#define MALLOC_ERROR 2
#define CUDA_ERROR 3
bool analysis_flag = false;
bool SHEMEM_flag = false;

// This function tests the `mergeSortWithCuda` function by comparing its output
// to the output of the `qsort` function applied to the expected array.
int testMergeSortWithCuda(int* actual, int* expected, const unsigned int count, unsigned int kernelBlockSize) {
    if(analysis_flag==false){
      printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");    
      printf("@@@@@ MergeSortWithCuda @@@@\n");
      printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
    }

  // Sort the `expected` array using the `qsort` function and the `cmpInt` comparison function.
    qsort(expected, count, sizeof(int), cmpInt);

  // Apply the `mergeSortWithCuda` function to the `actual` array.
    cudaError_t cudaStatus = mergeSortWithCuda(actual, count, kernelBlockSize, analysis_flag, SHEMEM_flag);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mergeSortWithCuda failed!");
        return CUDA_ERROR;
    }
  
  // Check if the `actual` array is equal to the `expected` array using the `assertArrEq` function.
    if (!assertArrEq(expected, actual, count * sizeof(int))) {
      printf("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      printf("!! CHECK MERGE SORT IS NOT CORRECT !!\n");
      printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      return EXPECTATION_ERROR;;
    }

    if(analysis_flag==false){
      printf("\n@@@@@@@@@@@@@@@@@@@@@@@@\n");
      printf("@@ SORTING IS CORRECT @@\n");
      printf("@@@@@@@@@@@@@@@@@@@@@@@@\n");
    // Save the sorted `actual` array to the "Mergesorted.txt" file.
      printf("Saving sorted array to sorted.txt file ...\n");
    }
    char fileName[] = "Sorted.txt";
    saveData(actual, fileName, count);
    return OK;
}

int main(int argc, char *argv[])
{

  // Get from the command line argument.
  // 1. the size of the arrays. to be randomly generated.
  // 2. # Block/Grid. set to preferable Block size (dim3 my_gridDim(B, 1, 1))
  // 3. annalysis Flag. set "true" to run Analysis, "false" to run sigle mergeSort output.
  // 4. Memory Sharing Flag. set "true" to use the gpu l1 shared meory to store the array during sorting.

    char *array_size = argv[1]; char *block_size = argv[2]; 
    char *A_flag = argv[3]; char *SHE_flag = argv[4]; 

    int status = MALLOC_ERROR;
    const unsigned int count = atoi(array_size);
    const unsigned int size = count * sizeof(int);
    unsigned int kernelBlockSize;
    //Check if a value for block_size is passed as an argument
    if(block_size) {kernelBlockSize = atoi(block_size);}
    else {kernelBlockSize = 1;}//If no value is passed, default to 1 block

    if(A_flag){analysis_flag = A_flag && strcasecmp(A_flag,"true")==0;}
    if(SHE_flag){SHEMEM_flag = SHE_flag && strcasecmp(SHE_flag,"true")==0;}
  
  // Allocate memory for the `actual` and `expected` arrays.
    int* actual = (int*) malloc(size);
    int* expected = (int*) malloc(size);

    if (actual != NULL && expected != NULL) {
        if(!analysis_flag) printf("Saving unsorted array to Unsorted.txt file ...\n");
        generateArray(actual, expected, count);
      // Test the `mergeSortWithCuda` function.
        status = testMergeSortWithCuda(actual, expected, count, kernelBlockSize);
      }
    else {fprintf(stderr, "malloc failed!");}

  // Deallocate memory for the `actual` and `expected` arrays.
    free(actual);
    free(expected);
  // Reset the CUDA device.
    int cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return CUDA_ERROR;
    }
  return status;
}
