# plotting script for  PMCMC, MCWM, DA-GP-MCMC, and ADA-GP-MCMC

# load packages
using DataFrames

# load plotting function
include("plotting.jl")

# set job name
jobname = "test_new_code_structure"
jobname = "pmcmc"
jobname = "lunarc_pmcmc_test"
jobname = "pmcmc_lunarc"
jobname = "mcwm"
jobname = "scaling_mcwm_10000"
jobname = "test_new_code_structure"
jobname = "_training"
jobname = "_dagpmcmc"
jobname = "_adagpmcmc"

Theta = Array(readtable("./Results/Theta"*jobname*".csv"))
loklik_avec_priorv = Array(readtable("./Results/loglik_avec_priorvec"*jobname*".csv"))
algorithm_parameters = Array(readtable("./Results/algorithm_parameters"*jobname*".csv"))

# set values:
loglik = loklik_avec_priorv[1,:]
accept_vec = loklik_avec_priorv[2,:]
prior_vec = loklik_avec_priorv[3,:]
burn_in = Int64(algorithm_parameters[1,1])
theta_true = algorithm_parameters[2:4,1]
theta_0 = algorithm_parameters[5:7,1]
Theta_parameters = algorithm_parameters[8:end,:]

analyse_results(Theta, loglik, accept_vec, prior_vec,theta_true, burn_in, Theta_parameters)
