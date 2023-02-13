#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
// function to swap two elements in an array
void swap(int* a, int* b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

// function to partition the array and return the pivot index
int partition(int arr[], int low, int high)
{
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; j++)
    {
        if (arr[j] <= pivot)
        {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

// function to sort the array using sequencial quicksort
void quicksortV1(int arr[], int low, int high)
{
    if (low < high)
    {
        int pivot = partition(arr, low, high);
        quicksortV1(arr, low, pivot - 1);
        quicksortV1(arr, pivot + 1, high);
    }
}

// function to sort the array using quicksort using omp task directive
void quicksortV2(int arr[], int low, int high)
{
    if (low < high)
    {
        int pivot = partition(arr, low, high);
        #pragma omp task
        quicksortV2(arr, low, pivot - 1);
        #pragma omp task
        quicksortV2(arr, pivot + 1, high);
    }
}

// function to sort the array using quicksort
void quicksortV3(int arr[], int low, int high) {
    if (low < high) {
        int pivot = partition(arr, low, high);
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                #pragma omp parallel for
                for (int i = low; i < pivot; i++)
                    quicksortV3(arr, i, pivot - 1);
            }
            #pragma omp section
            {
                #pragma omp parallel for
                for (int i = pivot + 1; i <= high; i++)
                    quicksortV3(arr, pivot + 1, i);
            }
        }
    }
}


// function to sort the array using quicksort
void quicksortV4(int arr[], int low, int high) {
    if (low < high) {
        int pivot = partition(arr, low, high);
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                #pragma omp task
                quicksortV4(arr, low, pivot - 1);
                #pragma omp task
                quicksortV4(arr, pivot + 1, high);
            }
        }
    }
}

// main function
int main(int argc, char *argv[])
{
    char *a = argv[1];
    int size = atoi(a);
    int arrV1[size];
    int arrV2[size];
    int arrV3[size];
    int arrV4[size];
	srand(1);    
    for (int i=0;i<size;i++){
    	arrV1[i] = rand();
        arrV2[i]= arrV1[i];
        arrV3[i]= arrV1[i];
        arrV4[i]= arrV1[i];
    }
    printf("Number of avilable Threads = %d \n", omp_get_max_threads());
    int n = sizeof(arrV1) / sizeof(arrV1[0]);


    float tstart,elapsed;
    printf("######################################\n");
    printf("##### Sequencial Quick Sort ######\n");
    tstart = omp_get_wtime();
    // sort the array
    quicksortV1(arrV1, 0, n - 1);
    elapsed = omp_get_wtime() - tstart;
    printf("\t Elapsed time %f\n", elapsed);
    printf("## End of sequencial Quick Sort ##\n");
    printf("######################################\n");

    printf("######################################\n");
    printf("##### using omp single directive ######\n");
    tstart = omp_get_wtime();
    // enable parallel execution
    #pragma omp parallel
    {
        #pragma omp single
        {
            quicksortV2(arrV2, 0, n - 1);
        }
    }
    elapsed = omp_get_wtime() - tstart;
    printf("\t Elapsed time %f\n", elapsed);
    printf("## End of using single directive ##\n");
    printf("######################################\n");

    printf("######################################\n");
    printf("##### Using section+for directives ######\n");
    tstart = omp_get_wtime();
    // sort the array
    quicksortV3(arrV3, 0, n - 1);
    elapsed = omp_get_wtime() - tstart;
    printf("\t Elapsed time %f\n", elapsed);
    printf("## End of section+for directives  ##\n");
    printf("######################################\n");

    printf("######################################\n");
    printf("##### Using section+nowait directives ######\n");
    tstart = omp_get_wtime();
    // sort the array
    quicksortV4(arrV3, 0, n - 1);
    elapsed = omp_get_wtime() - tstart;
    printf("\t Elapsed time %f\n", elapsed);
    printf("## End of section+nowait directives  ##\n");
    printf("######################################\n");

    // print the sorted array
    printf("Sorted array: \n");
    for (int i = 0; i < n; i++)
    {
        printf("%d ", arrV4[i]);
    }
    printf("\n");
    return 0;
}