#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

//function to merge two subarrays of arr[]
void merge(int arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1; //size of left array
    int n2 = r - m; //size of right array
    int L[n1], R[n2]; //temp arrays

    //copy data to temp arrays
    for (i = 0; i < n1; i++) {
        L[i] = arr[l + i];
    }
    for (j = 0; j < n2; j++) {
        R[j] = arr[m + 1 + j];
    }

    //merge temp arrays
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    //copy remaining elements of L[]
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    //copy remaining elements of R[]
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

//function to sort an array using merge sort Sequencial
void mergeSortV1(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSortV1(arr, l, m);
        mergeSortV1(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

//function to sort an array using merge sort using section directive
void mergeSortV2(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        //parallel execution of the two recursive calls
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSortV2(arr, l, m); //call mergeSort with left half of the array
            #pragma omp section
            mergeSortV2(arr, m + 1, r); //call mergeSort with right half of the array
        }
        //merge the two sorted subarrays
        merge(arr, l, m, r);
    }
    
}

//function to sort an array using merge sort using parallel for + critical directive
void mergeSortV3(int arr[], int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;

        #pragma omp parallel for
        for (int i = l; i <= r; i++) {
            mergeSortV3(arr, l, m);
            mergeSortV3(arr, m + 1, r);
        }
        #pragma omp critical
        {
            merge(arr, l, m, r);
        }
    }
}

//function to sort an array using merge sort using parallel task directive
void mergeSortV4(int arr[], int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;

        #pragma omp task
        mergeSortV4(arr, l, m);

        #pragma omp task
        mergeSortV4(arr, m + 1, r);

        #pragma omp taskwait
        merge(arr, l, m, r);
    }
}

int main(int argc, char *argv[]) {
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
    float tstart,elapsed;
    printf("######################################\n");
    printf("##### Sequencial implementation ######\n");
    tstart = omp_get_wtime();
    //sorting the array using merge sort
    mergeSortV1(arrV1, 0, size - 1);
    elapsed = omp_get_wtime() - tstart;
    printf("\t Elapsed time %f\n", elapsed);
    printf("## End of sequencial Implimentation ##\n");
    printf("######################################\n");

    printf("######################################\n");
    printf("##### Section implementation ######\n");
    tstart = omp_get_wtime();
    //sorting the array using merge sort
    mergeSortV2(arrV2, 0, size - 1);
    elapsed = omp_get_wtime() - tstart;
    printf("\t Elapsed time %f\n", elapsed);
    printf("## End of Section Implimentation ##\n");
    printf("######################################\n");

    printf("######################################\n");
    printf("##### task implementation ######\n");
    tstart = omp_get_wtime();
    //sorting the array using merge sort
    mergeSortV4(arrV4, 0, size - 1);
    elapsed = omp_get_wtime() - tstart;
    printf("\t Elapsed time %f\n", elapsed);
    printf("## End of task Implimentation ##\n");
    printf("######################################\n");

    printf("######################################\n");
    printf("##### For+Critical implementation ######\n");
    tstart = omp_get_wtime();
    //sorting the array using merge sort
    mergeSortV3(arrV3, 0, size - 1);
    elapsed = omp_get_wtime() - tstart;
    printf("\t Elapsed time %f\n", elapsed);
    printf("## End of For+Critical Implimentation ##\n");
    printf("######################################\n");


    return 0;
}
