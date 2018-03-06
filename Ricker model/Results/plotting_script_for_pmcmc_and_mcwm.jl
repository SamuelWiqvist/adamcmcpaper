# plotting script for  PMCMC, MCWM, DA-GP-MCMC, and ADA-GP-MCMC

# load packages
using DataFrames

# load plotting function
include("plotting.jl")


################################################################################
###   Notes regarding the different job in Lunarc                            ###
################################################################################

# MCWM: Job_id=508748; Runtime: 00:26:08

# DA/ADA: Job_id=??; Runtime: ??

################################################################################
###   Results for MCWM, DA-GP-MCMC, and ADA-GP-MCMC                          ###
################################################################################

# set job name
jobname = "res_lunarc_mcwm"
jobname = "training"
jobname = "_dagp"
jobname = "_adagp"

# path to folder with data
savepath = "C:\\Users\\samuel\\Dropbox\\Phd Education\\Projects\\project 1 accelerated DA and DWP SDE\\results\\ricker model\\"

# load data
Theta = Array(readtable(savepath*"Theta"*jobname*".csv"))
loklik_avec_priorv = Array(readtable(savepath*"loglik_avec_priorvec"*jobname*".csv"))
algorithm_parameters = Array(readtable(savepath*"algorithm_parameters"*jobname*".csv"))

# set values
loglik = loklik_avec_priorv[1,:]
accept_vec = loklik_avec_priorv[2,:]
prior_vec = loklik_avec_priorv[3,:]
burn_in = Int64(algorithm_parameters[1,1])
theta_true = algorithm_parameters[2:4,1]
theta_0 = algorithm_parameters[5:7,1]
prior_parameters = algorithm_parameters[8:end,:]

# calc results and plot results 
analyse_results(Theta, loglik, accept_vec, prior_vec,theta_true, burn_in, prior_parameters)


################################################################################
###    Compare results for MCWM, DA-GP-MCMC, and ADA-GP-MCMC                                            ###
################################################################################


jobname_mcwm = "res_mcwm_lunarc"
jobname_da = "_dagp"
jobname_ada = "_adagp"

savepath = "C:\\Users\\samuel\\Dropbox\\Phd Education\\Projects\\project 1 accelerated DA and DWP SDE\\results\\ricker model\\"

Theta_mcwm = Array(readtable(savepath*"Theta"*jobname_mcwm*".csv"))
loklik_avec_priorv_mcwm = Array(readtable(savepath*"loglik_avec_priorvec"*jobname_mcwm*".csv"))
algorithm_parameters_mcwm = Array(readtable(savepath*"algorithm_parameters"*jobname_mcwm*".csv"))

# set values:
loglik_mcwm = loklik_avec_priorv[1,:]
accept_vec_mcwm = loklik_avec_priorv[2,:]
prior_vec_mcwm = loklik_avec_priorv[3,:]
burn_in_mcwm = Int64(algorithm_parameters[1,1])
theta_true_mcwm = algorithm_parameters[2:4,1]
theta_0_mcwm = algorithm_parameters[5:7,1]
prior_parameters = algorithm_parameters[8:end,:]


Theta_da = Array(readtable(savepath*"Theta"*jobname_da*".csv"))
loklik_avec_priorv_da = Array(readtable(savepath*"loglik_avec_priorvec"*jobname_da*".csv"))
algorithm_parameters_da = Array(readtable(savepath*"algorithm_parameters"*jobname_da*".csv"))

# set values:
loglik_da = loklik_avec_priorv[1,:]
accept_vec_da = loklik_avec_priorv[2,:]
prior_vec_da = loklik_avec_priorv[3,:]
burn_in_da = Int64(algorithm_parameters[1,1])
theta_true_da = algorithm_parameters[2:4,1]
theta_0_da = algorithm_parameters[5:7,1]


Theta_ada = Array(readtable(savepath*"Theta"*jobname_ada*".csv"))
loklik_avec_priorv_ada = Array(readtable(savepath*"loglik_avec_priorvec"*jobname_ada*".csv"))
algorithm_parameters_ada = Array(readtable(savepath*"algorithm_parameters"*jobname_ada*".csv"))

# set values:
loglik_ada = loklik_avec_priorv[1,:]
accept_vec_ada = loklik_avec_priorv[2,:]
prior_vec_ada = loklik_avec_priorv[3,:]
burn_in_ada = Int64(algorithm_parameters[1,1])
theta_true_ada = algorithm_parameters[2:4,1]
theta_0_ada = algorithm_parameters[5:7,1]

# Posterior
x_c1 = prior_parameters[1,1]-0.5:0.01:prior_parameters[1,2]+0.5
x_c2 = prior_parameters[2,1]-0.5:0.01:prior_parameters[2,2]+0.5
x_c3 = prior_parameters[3,1]-0.5:0.01:prior_parameters[3,2]+0.5


priordens_c1 = pdf(Uniform(prior_parameters[1,1], prior_parameters[1,2]), x_c1)
priordens_c2 = pdf(Uniform(prior_parameters[2,1], prior_parameters[2,2]), x_c2)
priordens_c3 = pdf(Uniform(prior_parameters[3,1], prior_parameters[3,2]), x_c3)

h1_mcwm = kde(Theta_mcwm[1,burn_in_mcwm:end])
h2_mcwm = kde(Theta_mcwm[2,burn_in_mcwm:end])
h3_mcwm = kde(Theta_mcwm[3,burn_in_mcwm:end])

h1_da = kde(Theta_da[1,burn_in_da:end])
h2_da = kde(Theta_da[2,burn_in_da:end])
h3_da = kde(Theta_da[3,burn_in_da:end])

h1_ada = kde(Theta_ada[1,burn_in_ada:end])
h2_ada = kde(Theta_ada[2,burn_in_ada:end])
h3_ada = kde(Theta_ada[3,burn_in_ada:end])


PyPlot.figure()
ax = axes()

subplot(311)
PyPlot.plot(h1.x,h1.density, "b")
PyPlot.hold(true)
PyPlot.plot(h4.x,h4.density, "r")
PyPlot.plot(x_c1,priordens_c1, "g")
PyPlot.plot((theta_true[1], theta_true[1]), (0, maximum(h1.density)), "k")
PyPlot.ylabel(L"log $r$",fontsize=text_size)

subplot(312)
PyPlot.plot(h2.x,h2.density, "b")
PyPlot.hold(true)
PyPlot.plot(h5.x,h5.density, "r")
PyPlot.plot(x_c2,priordens_c2, "g")
PyPlot.plot((theta_true[2], theta_true[2]), (0, maximum(h2.density)), "k")
PyPlot.ylabel(L"log $\phi$",fontsize=text_size)

subplot(313)
PyPlot.plot(h3.x,h3.density, "b")
PyPlot.hold(true)
PyPlot.plot(h6.x,h6.density, "r")
PyPlot.plot(x_c3,priordens_c3, "g")
PyPlot.plot((theta_true[3], theta_true[3]), (0, maximum(h3.density)), "k")
PyPlot.ylabel(L"log $\sigma$",fontsize=text_size)
ax[:tick_params]("both",labelsize=label_size)
