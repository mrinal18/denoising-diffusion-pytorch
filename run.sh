#!/bin/bash
#SBATCH -J MNIST_diffusion
#SBATCH -o ./logs/%j.out
#SBATCH -e ./logs/%j.err
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:A40:2
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH --mem=70g
#SBATCH --oversubscribe
#SBATCH -A PSY53C17

#python run_cifar_100.py
accelerate launch run_cifar_100.py
