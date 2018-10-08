#!/bin/bash
#SBATCH --workdir /scratch/maulini
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 1G

echo STARTING AT `date`

./pi 64 1000000000

echo FINISHED at `date`

