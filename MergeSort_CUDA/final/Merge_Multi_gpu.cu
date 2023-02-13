// Referencing my code extension files. 
#include "utilities.cu" //utilities.cu refers to different function I used, which are not directly related to the GPU task.
                        // It also includes all the .h files i used in my project.
#include "kernelFunctions.cu" //kernelFuntions.cu refers to all my CUDA related code [__global__ kernel] function definitions.

// Function decleration
double startMergeSort(int* actual, int* expected, int* aux, int count, int gpu_count, int SHEMEM_flag);

#define OK 1
#define EXPECTATION_ERROR 1 // error code for expectation error
#define MALLOC_ERROR 2 // error code for memory allocation error
#define CUDA_ERROR 3 // error code for CUDA error

double startMergeSort(int* actual, int* expected, int* aux, int count, int gpu_count, int SHEMEM_flag) {
  double exec_time[9];
  for(int i=0; i<9; i++){exec_time[i] = 0.0;} // initialize all elements of exec_time to 0.0 otherwise junk values could add up later.
  double total_exec_time = 0.0; // total execution time for the entire merge sort
  int chunk_size = (count/gpu_count); // size of each chunk in the array
  int chunk_rmd = count%gpu_count; // size of remain chunk in the array

  // pointers to the chunks of the actual array
  int *chunk1 = actual;
  int *chunk2 = actual + chunk_size;
  int *chunk3 = actual + 2 * chunk_size;
  int *chunk4 = actual + 3 * chunk_size;
  int *chunk5 = actual + 4 * chunk_size;
  int *chunk6 = actual + 5 * chunk_size;
  int *chunk7 = actual + 6 * chunk_size;
  int *chunk8 = actual + 7 * chunk_size;

  // pointers to the chunks of the Aux array
  int *chunk1aux = aux;
  int *chunk2aux = aux + chunk_size;
  int *chunk3aux = aux + 2 * chunk_size;
  int *chunk4aux = aux + 3 * chunk_size;
  int *chunk5aux = aux + 4 * chunk_size;
  int *chunk6aux = aux + 5 * chunk_size;
  int *chunk7aux = aux + 6 * chunk_size;
  int *chunk8aux = aux + 7 * chunk_size;

  
  int id = 0; // later can be replaced by the gpu device index using openmp threading.
  if(chunk_size>6000) SHEMEM_flag = 0; //this line checks if the chunk size can fit in the L1 shared memory.
  if (gpu_count>=1) {
    // sort the chunk1 array and store the execution time in exec_time[0] ....do the same with the other chunk of the array sections.
    exec_time[0] = sortArray(chunk1, chunk1aux, chunk_size, id, 2, SHEMEM_flag); 
    } 
  if (gpu_count>=2) {
    exec_time[1]= sortArray(chunk2, chunk2aux, chunk_size+chunk_rmd, id, 2, SHEMEM_flag);
    } 
  if (gpu_count>=3) {
    exec_time[2] = sortArray(chunk3, chunk3aux, chunk_size+chunk_rmd, id, 2, SHEMEM_flag);
    } 
  if (gpu_count>=4) {
    exec_time[3] = sortArray(chunk4, chunk4aux, chunk_size+chunk_rmd, id, 2, SHEMEM_flag);
    } 
  if (gpu_count>=5) {
      exec_time[4] = sortArray(chunk5, chunk5aux, chunk_size+chunk_rmd, id, 2, SHEMEM_flag);
    } 
  if (gpu_count>=6) {
      exec_time[5] = sortArray(chunk6, chunk6aux, chunk_size+chunk_rmd, id, 2, SHEMEM_flag);
    } 
  if (gpu_count>=7) {
      exec_time[6] = sortArray(chunk7, chunk7aux, chunk_size+chunk_rmd, id, 2, SHEMEM_flag);
    } 
  if (gpu_count==8) {
      exec_time[7] = sortArray(chunk8, chunk8aux, chunk_size+chunk_rmd, id, 2, SHEMEM_flag);
    }

  //The following Commented code is my idea on how we can run all the 8 GPUs concurrently using OpenMp.
  // threads in OpenMp initiates all the GPUs with their respective section of the array to sort.
  // At the end one GPU sorts all the sub arrays which are partialy sorted. 
  //if(chunk_size>6000) SHEMEM_flag = 0; //this line checks if the chunk size can fit in the shared l1 memory.
  // In the following section i am using OpenMp to initialize the GPUs in parallel
  // int id;
  // #pragma omp parallel num_threads(gpu_count)
  // {
  //   id = omp_get_thread_num(); //0; // omp_get_thread_num(); // later can be replaced by the thread number that initiates gpu devices.
  //   if (gpu_count>=1 && id == 0) {
  //     // sort the chunk1 array and store the execution time in exec_time[0] ....do the same with the other chunk of the array sections.
  //     exec_time[0] = sortArray(chunk1, chunk1aux, chunk_size, id, 2, SHEMEM_flag); 
  //     } 
  //   if (gpu_count>=2 && id == 1) {
  //     exec_time[1]= sortArray(chunk2, chunk2aux, chunk_size+chunk_rmd, id, 2, SHEMEM_flag);
  //     } 
  //   if (gpu_count>=3 && id == 2) {
  //     exec_time[2] = sortArray(chunk3, chunk3aux, chunk_size+chunk_rmd, id, 2, SHEMEM_flag);
  //     } 
  //   if (gpu_count>=4 && id == 3) {
  //     exec_time[3] = sortArray(chunk4, chunk4aux, chunk_size+chunk_rmd, id, 2, SHEMEM_flag);
  //     } 
  //   if (gpu_count>=5 && id == 4) {
  //       exec_time[4] = sortArray(chunk5, chunk5aux, chunk_size+chunk_rmd, id, 2, SHEMEM_flag);
  //     } 
  //   if (gpu_count>=6 && id == 5) {
  //       exec_time[5] = sortArray(chunk6, chunk6aux, chunk_size+chunk_rmd, id, 2, SHEMEM_flag);
  //     } 
  //   if (gpu_count>=7 && id == 6) {
  //       exec_time[6] = sortArray(chunk7, chunk7aux, chunk_size+chunk_rmd, id, 2, SHEMEM_flag);
  //     } 
  //   if (gpu_count==8 && id == 7) {
  //       exec_time[7] = sortArray(chunk8, chunk8aux, chunk_size+chunk_rmd, id, 2, SHEMEM_flag);
  //     }
  // }


  // setting the flag to false because of the shared memory wall. The last merge sort takes all the sub-arrays sorted using multi-gpus and this time the shared memory might not fit.
  if(count>6000)SHEMEM_flag = false;
  exec_time[8] = sortArray(actual, aux, count, id, chunk_size, SHEMEM_flag);
  // Check if the actual array is sorted as expected and return EXPECTATION_ERROR if not
  if(!checkSort(actual, expected, count)){fprintf(stderr, "EXPECTATION_ERROR!"); return EXPECTATION_ERROR;}
  for(int i = 0; i < 9; i++){total_exec_time = total_exec_time + exec_time[i];}
  //printf("Saving sorted array to sorted.txt file ...\n");
  char fileName[] = "Sorted.txt";
  saveData(actual, fileName, count);
  return total_exec_time;
}



int main(int argc, char *argv[])
{
  // Get the size of the arrays, memory sharing flag, and number of GPUs to use from the command line arguments
  char *array_size = argv[1]; 
  char *SHE_flag = argv[2];
  char *gpu_cnt = argv[3];

  double exec_time = 0.0;
  // Convert array size from string to integer
  const unsigned int count = atoi(array_size);
  // Calculate the size in bytes of the arrays
  const unsigned int size = count * sizeof(int);
  // Convert the number of GPUs to use from string to integer
  const unsigned int gpu_count = atoi(gpu_cnt);
  int SHEMEM_flag;
  // Check the memory sharing flag, if set to "true", set SHEMEM_flag to true
  if(SHE_flag){SHEMEM_flag = SHE_flag && strcasecmp(SHE_flag,"true")==0;}

  // Allocate memory for the actual, expected, and aux arrays
  int* actual = (int*) malloc(size);
  int* expected = (int*) malloc(size);
  int* aux = (int*) malloc(size);
  

  // Check if memory allocation was successful
  if (actual != NULL && expected != NULL) {
      // Generate and sort the arrays
      //printf("Saving unsorted array to Unsorted.txt file ...\n");
      generateArray(actual, expected, count);
      exec_time = startMergeSort(actual, expected, aux, count, gpu_count, SHEMEM_flag);
    }
  else {fprintf(stderr, "malloc failed!"); return MALLOC_ERROR;}

  // // Deallocate memory for the actual and expected arrays
  free(actual);
  free(expected);
  free(aux);

  // Reset the CUDA device
  int cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return CUDA_ERROR;
  }
  // Print the execution time
  printf("%f", exec_time);
  return OK;
}
