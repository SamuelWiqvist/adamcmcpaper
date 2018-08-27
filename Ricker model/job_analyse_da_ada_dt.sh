#!/bin/sh

# Script to run alg. analyses for DA and ADA using dt selection method 

# Set up for run: 
 
# need this since I use a LU project 
#SBATCH -A lu2017-2-14

# need this since I use a LU project
#SBATCH -p lu

# time consumption HH:MM:SS
#SBATCH -t 40:00:00 

# name for script 
#SBATCH -J ricker_alg_prop_dt

# controll job outputs 
#SBATCH -o lunarc_output/outputs_ricker_alg_prop_dt_%j.out
#SBATCH -e lunarc_output/errors_ricker_alg_prop_dt_%j.err

# set number of nodes 
#SBATCH -N 1
#SBATCH -n 1


# notification 
#SBATCH --mail-user=samuel.wiqvist@matstat.lu.se 
#SBATCH --mail-type=ALL  


# load modules  
#module load GCC/4.9.3 
#module load impi/5.0.3.048
#module load julia/0.4 

ml load icc/2017.1.132-GCC-6.3.0-2.27
ml load impi/2017.1.132
ml load julia/0.5.2


# # set nbr of threads
# # export JULIA_NUM_THREADS=10
# export  MKL_NUM_THREADS=20 

# run program
julia analyse_alg_prop.jl dt 

