#!/bin/bash
#SBATCH --job-name=MergeSort
#SBATCH --mail-user=sami.osman@studio.unibo.it
#SBATCH --output=Merge_Out.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00

array_size=6000
SHEMEM_flag="true" #The code will take "false" by default if the array size will not fit in case of using shared memory.
gpu_count=8

printf "Merge_Sort By Sami Osman\n"
nvcc -o Merge_Multi_gpu Merge_Multi_gpu.cu

printf "@@@@@@@@@@@@@@@@@@@@@@@\n"
printf "@@  Weak Scaling     @@\n"
printf "@@ No MEMORY SHARING @@\n"
printf "@@@@@@@@@@@@@@@@@@@@@@@\n"
printf "# GPU\tA_Size\tT1\t T2\tT3\tT4\tT5"
for gpu_cnt in {1..8}
  do
    printf "\n $gpu_cnt\t"
    printf "$array_size"
    for T in {1..5}
      do
        cuda_EXECU_Time=$(./Merge_Multi_gpu $array_size "false" $gpu_cnt)
        printf "\t$cuda_EXECU_Time"
    done
done
printf "\n@@@@@@@@@@@@@@@@@@@@@@@\n"
printf "@@  Strong Scaling   @@\n"
printf "@@ No MEMORY SHARING @@\n"
printf "@@@@@@@@@@@@@@@@@@@@@@@\n"
printf "# GPU\tA_Size\tT1\tT2\tT3\tT4\tT5"
A_size=$array_size
for gpu_cnt in {1..8}
  do
    A_size=$(($array_size*$gpu_cnt))
    printf "\n $gpu_cnt\t"
    printf "$A_size"
    for T in {1..5}
      do
        cuda_EXECU_Time=$(./Merge_Multi_gpu $A_size "false" $gpu_cnt)
        printf "\t$cuda_EXECU_Time"
    done
done
printf "\n@@@@@@@@@@@@@@@@@@@@@@@@@\n"
printf "@@  Weak Scaling       @@\n"
printf "@@ With MEMORY SHARING @@\n"
printf "@@@@@@@@@@@@@@@@@@@@@@@@@\n"
printf "# GPU\tA_Size\tT1\t T2\tT3\tT4\tT5"
for gpu_cnt in {1..8}
  do
    printf "\n$gpu_cnt\t"
    printf "$array_size"
    for T in {1..5}
      do
        cuda_EXECU_Time=$(./Merge_Multi_gpu $array_size "true" $gpu_cnt)
        printf "\t$cuda_EXECU_Time"
    done
done
printf "\n@@@@@@@@@@@@@@@@@@@@@@@@@\n"
printf "@@  Strong Scaling     @@\n"
printf "@@ With MEMORY SHARING @@\n"
printf "@@@@@@@@@@@@@@@@@@@@@@@@@\n"
printf "# GPU\tA_Size\tT1\t T2\tT3\tT4\tT5"
A_size=$array_size
for gpu_cnt in {1..8}
  do
    A_size=$(($array_size*$gpu_cnt))
    printf "\n $gpu_cnt\t"
    printf "$A_size"
    for T in {1..5}
      do
        cuda_EXECU_Time=$(./Merge_Multi_gpu $A_size "true" $gpu_cnt)
        printf "\t$cuda_EXECU_Time"
    done
done