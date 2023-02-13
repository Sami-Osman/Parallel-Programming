// Function to merge two sub-arrays in a sorted manner
void merge(int arr[], int l, int m, int r) {
    // i, j and k are pointers used in the merging process
    int i, j, k;
    
    // n1 is the size of the left sub-array
    int n1 = m - l + 1;
    // n2 is the size of the right sub-array
    int n2 = r - m;
    
    // Temporary arrays to store the left and right sub-arrays
    int L[n1], R[n2];
    
    // Copy data to temporary arrays L[] and R[]
    for (i = 0; i < n1; i++) {
        L[i] = arr[l + i];
    }
    for (j = 0; j < n2; j++) {
        R[j] = arr[m + 1 + j];
    }
    
    // Initialize pointers i, j and k
    i = 0;
    j = 0;
    k = l;
    
    // Merge the temporary arrays back into arr[l..r]
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
    
    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    
    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// Main function that sorts the given array using merge sort algorithm
void mergeSort(int arr[], int l, int r) {
    // Check if l is smaller than r
    if (l < r) {
        // Find the middle point
        int m = l + (r - l) / 2;
        
        // Sort the first and second halves of the array recursively
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        
        // Merge the sorted halves
        merge(arr, l, m, r);
    }
}