
The BASH file set to execute as follows:
The first 7 lines specify the job parameters: job name, email address for notifications, output file name, number of nodes, number of CPU's per task, GPU reservation, and time limit.

The next 5 lines set the initial values for array_size, block_size, shared memory flag and analysis flag.

The next line compiles the CUDA program "Merge_sort_1.cu" and generates an executable named "Merge_sort_1".

The script then enters an "if" block that checks the value of the "analysis_flag" variable. If it is set to "true", the script performs four sets of analyses with different parameters.

a. The first analysis tests the effect of increasing the block size on the execution time with no memory sharing.

b. The second analysis tests the effect of increasing the array size on the execution time with no memory sharing.

c. The third analysis tests the effect of increasing the block size on the execution time with memory sharing.

d. The fourth analysis tests the effect of increasing the array size on the execution time with memory sharing.

For each analysis, the script prints a header, sets the appropriate parameters, and executes the "Merge_sort_1" program 5 times. The execution time of each run is captured and displayed.

If the "analysis_flag" is set to "false", the script executes the "Merge_sort_1" program once with the specified parameters without any analysis.

The script ends after executing all the analyses or a single execution of the "Merge_sort_1" program.
