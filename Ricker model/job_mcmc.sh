#!/bin/sh

# Set up for run: 
 
# need this since I use a LU project 
#SBATCH -A lu2018-2-22

# need this since I use a LU project
#SBATCH -p lu

# time consumption HH:MM:SS
#SBATCH -t 02:00:00 

# name for script 
#SBATCH -J mcmc_ricker_model

# controll job outputs 
#SBATCH -o lunarc_output/outputs_mcmc_ricker_model_%j.out
#SBATCH -e lunarc_output/errors_mcmc_ricker_model_%j.err

# set number of nodes 
#SBATCH -N 1
#SBATCH -n 1


# notification 
#SBATCH --mail-user=samuel.wiqvist@matstat.lu.se 
#SBATCH --mail-type=ALL  

# load modules  

ml load icc/2017.1.132-GCC-6.3.0-2.27
ml load impi/2017.1.132
ml load julia/0.5.2

# run program
julia run_mcmc.jl 

