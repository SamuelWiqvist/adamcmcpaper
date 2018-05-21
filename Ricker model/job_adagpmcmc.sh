#!/bin/sh

# Description: This is a simple script to test Aurora

# Set up for run: 
 
# need this since I use a LU project 
#SBATCH -A lu2017-2-14

# need this since I use a LU project
#SBATCH -p lu

# time consumption HH:MM:SS
#SBATCH -t 24:00:00 

# name for script 
#SBATCH -J adagpmcmc_ricker_model

# controll job outputs 
#SBATCH -o outputs_%j.out
#SBATCH -e errors_%j.err

# set number of nodes 
#SBATCH -N 1
#SBATCH --tasks-per-node=20
#SBATCH -n 20


# notification 
#SBATCH --mail-user=samuel.wiqvist@matstat.lu.se 
#SBATCH --mail-type=ALL  

# load modules  
module load GCC/4.9.3 
module load impi/5.0.3.048
module load julia/0.4 

# run program
julia run_adagpmcmc.jl 
