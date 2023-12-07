#!/bin/bash
#SBATCH --job-name=example1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --account=eecs587f23_class
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=4G

# # Testing script
# ./CNN -test >> "output.txt"

#Training command
# ./CNN -train 5 "weights.txt" >> "output.txt"

# # Training and Testing on GPU
# ./CNN -both >> "gpu_train_test.txt"

# Traniing on GPU
# ./CNN -train 5 "weights_gpu.txt" >> "gpu_learn.txt"

# # Testing on GPU
# ./CNN -test "weights_gpu.txt" >> "gpu_test.txt"

# # Training on CPU
# ./CNN -cputrain 5 "weights_cpu.txt" >> "cpu_learn.txt"

# Testing on CPU
./CNN -cputest "weights_cpu.txt" >> "cpu_test.txt"