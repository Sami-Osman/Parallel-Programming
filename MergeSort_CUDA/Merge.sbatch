#!/bin/bash

#SBATCH --job-name=P_MergeSort
#SBATCH --mail-user=sami.osman@studio.unibo.it
#SBATCH --output=MergeSort_Out.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00

array_size=1000
block_size=10
SHEMEM_flag="true"
analysis_flag="true"

printf "Merge_Sort By Sami Osman\n"
printf "Array size:          $array_size\n"
printf "Block size:          $block_size\n"
printf "Analysis flag        $analysis_flag\n"
printf "Shared memory flag:  $SHEMEM_flag\n"


nvcc -o Merge_sort_1 Merge_sort_1.cu
if($analysis_flag)
  then
    printf "@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
    printf "@@  Increasing BLOCK Size @@\n"
    printf "@@    No MEMORY SHARING   @@\n"
    printf "@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
    printf "CUDA_BLOCKS\tT1\t\t T2\t\tT3\t\tT4\t\tT5"
    for block_sizes in {10..100..10}
      do
        printf "\n $block_sizes\t"
        for T in {1..5}
          do
	    #the last argument is "false" because i am doing this task without Memory sharing.
            cuda_EXECU_Time=$(./Merge_sort_1 $array_size $block_sizes $analysis_flag "false") 
            printf "\t$cuda_EXECU_Time"
        done
      done

    printf "\n@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
    printf "@@  Increasing DATA Size @@\n"
    printf "@@    No MEMORY SHARING  @@\n"
    printf "@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
      printf "A_Size\t\tT1\t\t T2\t\tT3\t\tT4\t\tT5"
      A_size=$array_size
      for array_sizes in {1..10}
        do
          printf "\n $A_size  "
          for T in {1..5}
            do
	      #the last argument is "false" because i am doing this task without Memory sharing.
              cuda_EXECU_Time=$(./Merge_sort_1 $A_size $block_size $analysis_flag "false")
              printf "\t$cuda_EXECU_Time"
          done
          A_size=$((A_size+1000))
      done
     
    printf "\n@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
    printf "@@  Increasing BLOCK Size @@\n"
    printf "@@   With MEMORY SHARING  @@\n"
    printf "@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
    printf "CUDA_BLOCKS\tT1\t\t T2\t\tT3\t\tT4\t\tT5"
    for block_sizes1 in {10..100..10}
      do
        printf "\n $block_sizes1\t"
        for T in {1..5}
          do
	    #the last argument is "true" because i am doing this task with Memory sharing.
            cuda_EXECU_Time=$(./Merge_sort_1 $array_size $block_sizes1 $analysis_flag "true")
            printf "\t$cuda_EXECU_Time"
        done
      done

    printf "\n@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
    printf "@@  Increasing DATA Size @@\n"
    printf "@@   With MEMORY SHARING @@\n"
    printf "@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
      printf "A_Size\t\tT1\t\t T2\t\tT3\t\tT4\t\tT5"
      A_size=$array_size
      for array_sizes in {1..10}
        do
          printf "\n $A_size  "
          for T in {1..5}
            do
	      #the last argument is "true" because i am doing this task with Memory sharing.
              cuda_EXECU_Time=$(./Merge_sort_1 $A_size $block_size $analysis_flag "true")
              printf "\t$cuda_EXECU_Time"
          done
          A_size=$((A_size+1000))
      done
else
  ./Merge_sort_1 $array_size $block_size $analysis_flag $SHEMEM_flag
fi
