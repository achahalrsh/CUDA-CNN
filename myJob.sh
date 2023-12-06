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

# # Traniing and Testing on GPU
# ./CNN -both >> "output.txt"

# # Traniing and Testing on CPU
./CNN -cpu >> "output.txt"