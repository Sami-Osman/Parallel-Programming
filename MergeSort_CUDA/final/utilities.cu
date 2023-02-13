#include <stdbool.h> // stdbool.h: Boolean data type support
#include <stdio.h> // stdio.h: Input/Output operations
#include <stdlib.h> // stdlib.h: General utility functions
#include <string.h> // string.h: String handling functions
#include "cuda_runtime.h" //to provide the functions and data structures necessary for GPU programming using the CUDA platform.
#include <omp.h>
#include <time.h>


// Function definitions according their definition order:
bool checkSort(int *actual, int *expected, int count);
bool assertArrEq(int* expected, int* actual, size_t size);
int cmpInt(const void* a, const void* b);
int saveData(int data[], char *fileName, int size);
int testMergeSortWithCuda(int* actual, int* expected, const unsigned int count);
double gettime(void);
void generateArray(int *actual, int *expected, int count);

bool checkSort(int *actual, int *expected, int count)
{
    // Sort the `expected` array using `qsort` function and compare function `cmpInt`
    qsort(expected, count, sizeof(int), cmpInt);

    // Check if the `actual` array is equal to the `expected` array using the `assertArrEq` function.
    if (!assertArrEq(expected, actual, count * sizeof(int))) 
    {
        // If not equal, print the error message
        printf("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        printf("!! CHECK MERGE SORT IS NOT CORRECT !!\n");
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

        // Return `1` indicating an error
        return 0;
    }

    // Return `true` indicating success
    return true;
}

double gettime( void )
{
    // Declare a structure `ts` of type `timespec`
    struct timespec ts;

    // `clock_gettime` is a function that retrieves the current value of a clock identified by `CLOCK_MONOTONIC`
    // and stores it in the `ts` struct
    clock_gettime(CLOCK_MONOTONIC, &ts );

    // Return the sum of seconds (`ts.tv_sec`) and nanoseconds (`ts.tv_nsec`) in seconds
    // `1e9` represents one billion, so we are dividing the nanoseconds by one billion to get the value in seconds
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}

void generateArray(int *actual, int *expected, int count){
  // Fill the `actual` and `expected` arrays with random values.
  srand(1);
  for (unsigned int i = 0; i < count; i++) {
    expected[i] = actual[i] = rand();
    }
  // Save the unsorted `actual` array to the "Unsorted.txt" file.
  char fileName[] = "Unsorted.txt";
  saveData(actual, fileName, count);
}
// This function checks if two integer arrays are equal.
bool assertArrEq(int* expected, int* actual, size_t size) {
  // Compare the contents of the two arrays, `expected` and `actual`, with size `size`
  // using the `memcmp` function.
  // `memcmp` returns 0 if the two arrays have the same contents, otherwise, non-zero.
  // Return the result of `memcmp` == 0, indicating whether the two arrays are equal.
  return memcmp(expected, actual, size) == 0;
}
// This function compares two integers and returns their difference.
int cmpInt(const void* a, const void* b) {
  // Cast the void pointers `a` and `b` to `int` pointers.
  // De-reference the `int` pointers to access the actual integer values.
  // Return the difference of the two integers.
  return *(int*)a - *(int*)b;
}
// This function saves an integer array to a file.
int saveData(int data[], char *fileName, int size) {
  // Declare a file pointer `fptr`
  FILE *fptr;
  // Try to open the file with write mode, `"w"`.
  // The function returns the file pointer if the file is opened successfully,
  // otherwise, it returns `NULL`.
  if (fptr = fopen(fileName, "w")) {
    // Loop through the integer array, `data`, with `size` elements.
    for (int i = 0; i < size; i++) {
      // Write the integer value of `data[i]` to the file using `fprintf`.
      // The format specifier `"%i "` writes an integer followed by a space.
      fprintf(fptr, "%i ", data[i]);
      // If the current index is divisible by 5 (`(i + 1) % 5 == 0`),
      // add a newline character to the file using `fprintf`.
      if ((i + 1) % 5 == 0)
        fprintf(fptr, "\n");
    }
    // Close the file after writing is done.
    fclose(fptr);
  } else {
    // If the file could not be opened, print an error message with the file name.
    printf("Error in Saving %s File\n", fileName);
  }
  // Return 0 after saving is done.
  return 0;
}