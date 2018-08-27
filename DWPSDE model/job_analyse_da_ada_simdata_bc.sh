#!/bin/sh

# Set up for run: 
 
# need this since I use a LU project 
#SBATCH -A lu2018-2-22

# need this since I use a LU project
#SBATCH -p lu

# time consumption HH:MM:SS
#SBATCH -t 100:00:00 

# name for script 
#SBATCH -J job_ana_sim_data_bc

# controll job outputs 
#SBATCH -o lunarc_output/outputs_job_ana_sim_data_bc_%j.out
#SBATCH -e lunarc_output/errors_job_ana_sim_data_bc_%j.err

# set number of nodes 
#SBATCH -N 1
#SBATCH --tasks-per-node=20
#SBATCH -n 5

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
julia analyse_alg_prop.jl simdata biasedcoin

