# load packages for plotting

# load packages
using Plots
using PyPlot
using StatPlots
using DataFrames

include("plotting.jl")


jobname = "res_mcwm" # res_mcwm is the results for the mcwm algorithm
jobname = jobname_vec[4]
jobname = "_lunrac_ergp"
jobname = "accelerated"
jobname = ""

Theta = Array(readtable("./Results/Theta_ergp"*jobname*".csv"))
loklik_avec_priorv = Array(readtable("./Results/loglik_avec_priorvec_ergp"*jobname*".csv"))
algorithm_parameters = Array(readtable("./Results/algorithm_parameters_ergp"*jobname*".csv"))


# set values:
loglik = loklik_avec_priorv[1,:]
accept_vec = loklik_avec_priorv[2,:]
prior_vec = loklik_avec_priorv[3,:]
burn_in = Int64(algorithm_parameters[1,1])
theta_true = algorithm_parameters[2:4,1]
theta_0 = algorithm_parameters[5:7,1]
Theta_parameters = algorithm_parameters[8:end,:]

analyse_results(Theta, loglik, accept_vec, prior_vec,theta_true, burn_in, Theta_parameters)
