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

// function to sort the array using quicksort
void quicksort(int arr[], int low, int high)
{
    if (low < high)
    {
        int pivot = partition(arr, low, high);
        #pragma omp task
        quicksort(arr, low, pivot - 1);
        #pragma omp task
        quicksort(arr, pivot + 1, high);
    }
}

// main function
int main()
{
    // sample array to sort
    int arr[] = {9, 7, 5, 11, 12, 2, 14, 3, 10, 6};
    int n = sizeof(arr) / sizeof(arr[0]);

    // enable parallel execution
    #pragma omp parallel
    {
        #pragma omp single
        {
            quicksort(arr, 0, n - 1);
        }
    }

    // print the sorted array
    printf("Sorted array: \n");
    for (int i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
    return 0;
}
