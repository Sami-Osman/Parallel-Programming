#include <stdbool.h> // stdbool.h: Boolean data type support
#include <stdio.h> // stdio.h: Input/Output operations
#include <stdlib.h> // stdlib.h: General utility functions
#include <string.h> // string.h: String handling functions
#include "cuda_runtime.h" //to provide the functions and data structures necessary for GPU programming using the CUDA platform.


// Function definitions according their definition order:
bool assertArrEq(int* expected, int* actual, size_t size);
int cmpInt(const void* a, const void* b);
int saveData(int data[], char *fileName, int size);
int testMergeSortWithCuda(int* actual, int* expected, const unsigned int count);
double gettime(void);
void generateArray(int *actual, int *expected, int count);

double gettime( void )
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts );
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}
void generateArray(int *actual, int *expected, int count){
  // Fill the `actual` and `expected` arrays with random values.
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
