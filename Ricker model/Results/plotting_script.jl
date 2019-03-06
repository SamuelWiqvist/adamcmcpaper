# plotting script for  PMCMC, MCWM, DA-GP-MCMC, and ADA-GP-MCMC

# load packages
using DataFrames
using PyPlot

# fix for removing missing values
remove_missing_values(x) = reshape(collect(skipmissing(x)),3,:)

################################################################################
###   Plot data                                                              ###
################################################################################


# load data
y = Array(readtable("Ricker model/y_data_set_2.csv"))[:,1]

text_size = 25
label_size = 20

PyPlot.figure(figsize=(20,10))
ax = axes()
PyPlot.plot(y)
#PyPlot.xlabel("time", fontsize=text_size)
#PyPlot.ylabel("y", fontsize=text_size)
ax[:tick_params]("both",labelsize=label_size)


################################################################################
###   Plot resutls                                                              ###
################################################################################

# load plotting function
include(pwd()*"/Ricker model/Results/plotting.jl")

################################################################################
###   Results for PMCMC, MCWM, DA-GP-MCMC, and ADA-GP-MCMC                          ###
################################################################################

# set job name
jobname = "_dagpmcmc_lunarc"
jobname = "_adagpmcmc_lunarc"
jobname = "mcwm"
jobname = "pmcmc"


savepath = "Ricker model/Results/" # we are currently in the Results folder

# load data
Theta = remove_missing_values(Array(readtable(savepath*"Theta"*jobname*".csv")))
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

jobname_mcwm = "mcwm"
jobname_da = "_dagpmcmc_lunarc"
jobname_ada = "_adagpmcmc_lunarc"

savepath = "Ricker model/Results/"

Theta_mcwm = remove_missing_values(Array(readtable(savepath*"Theta"*jobname_mcwm*".csv")))
loklik_avec_priorv_mcwm = Array(readtable(savepath*"loglik_avec_priorvec"*jobname_mcwm*".csv"))
algorithm_parameters_mcwm = Array(readtable(savepath*"algorithm_parameters"*jobname_mcwm*".csv"))

# set values:
loglik_mcwm = loklik_avec_priorv_mcwm[1,:]
accept_vec_mcwm = loklik_avec_priorv_mcwm[2,:]
prior_vec_mcwm = loklik_avec_priorv_mcwm[3,:]
burn_in_mcwm = Int64(algorithm_parameters_mcwm[1,1])
theta_true = algorithm_parameters_mcwm[2:4,1]
theta_0_mcwm = algorithm_parameters_mcwm[5:7,1]
prior_parameters = algorithm_parameters_mcwm[8:end,:]


Theta_da = remove_missing_values(Array(readtable(savepath*"Theta"*jobname_da*".csv")))
loklik_avec_priorv_da = Array(readtable(savepath*"loglik_avec_priorvec"*jobname_da*".csv"))
algorithm_parameters_da = Array(readtable(savepath*"algorithm_parameters"*jobname_ada*".csv"))

# set values:
loglik_da = loklik_avec_priorv_da[1,:]
accept_vec_da = loklik_avec_priorv_da[2,:]
prior_vec_da = loklik_avec_priorv_da[3,:]
burn_in_da = Int64(algorithm_parameters_da[1,1])
theta_true_da = algorithm_parameters_da[2:4,1]
theta_0_da = algorithm_parameters_da[5:7,1]


Theta_ada = remove_missing_values(Array(readtable(savepath*"Theta"*jobname_ada*".csv")))
loklik_avec_priorv_ada = Array(readtable(savepath*"loglik_avec_priorvec"*jobname_ada*".csv"))
algorithm_parameters_ada = Array(readtable(savepath*"algorithm_parameters"*jobname_ada*".csv"))

# set values:
loglik_ada = loklik_avec_priorv_ada[1,:]
accept_vec_ada = loklik_avec_priorv_ada[2,:]
prior_vec_ada = loklik_avec_priorv_ada[3,:]
burn_in_ada = Int64(algorithm_parameters_ada[1,1])
theta_true_ada = algorithm_parameters_ada[2:4,1]
theta_0_ada = algorithm_parameters_ada[5:7,1]

# Posterior
x_c1 = prior_parameters[1,1]-0.5:0.01:prior_parameters[1,2]+0.5
x_c2 = prior_parameters[2,1]-0.5:0.01:prior_parameters[2,2]+0.5
x_c3 = prior_parameters[3,1]-0.5:0.01:prior_parameters[3,2]+0.5


priordens_c1 = pdf.(Uniform(prior_parameters[1,1], prior_parameters[1,2]), x_c1)
priordens_c2 = pdf.(Uniform(prior_parameters[2,1], prior_parameters[2,2]), x_c2)
priordens_c3 = pdf.(Uniform(prior_parameters[3,1], prior_parameters[3,2]), x_c3)

h1_mcwm = kde(Theta_mcwm[1,burn_in_mcwm:end])
h2_mcwm = kde(Theta_mcwm[2,burn_in_mcwm:end])
h3_mcwm = kde(Theta_mcwm[3,burn_in_mcwm:end])

h1_da = kde(Theta_da[1,burn_in_da:end])
h2_da = kde(Theta_da[2,burn_in_da:end])
h3_da = kde(Theta_da[3,burn_in_da:end])

h1_ada = kde(Theta_ada[1,burn_in_ada:end])
h2_ada = kde(Theta_ada[2,burn_in_ada:end])
h3_ada = kde(Theta_ada[3,burn_in_ada:end])

text_size = 15
label_size = 15

PyPlot.figure()
PyPlot.plot(h1_mcwm.x,h1_mcwm.density, "b")
PyPlot.hold(true)
PyPlot.plot(h1_da.x,h1_da.density, "r")
PyPlot.plot(h1_ada.x,h1_ada.density, "r--")
PyPlot.plot(x_c1,priordens_c1, "g")
PyPlot.plot((theta_true[1], theta_true[1]), (0, maximum([maximum(h1_mcwm.density);maximum(h1_da.density); maximum(h1_ada.density)])), "k")
PyPlot.xlabel(L"log $r$",fontsize=text_size)
PyPlot.ylabel(L"Density",fontsize=text_size)
PyPlot.figure()
PyPlot.plot(h2_mcwm.x,h2_mcwm.density, "b")
PyPlot.hold(true)
PyPlot.plot(h2_da.x,h2_da.density, "r")
PyPlot.plot(h2_ada.x,h2_ada.density, "r--")
PyPlot.plot(x_c2,priordens_c2, "g")
PyPlot.plot((theta_true[2], theta_true[2]), (0, maximum([maximum(h2_mcwm.density);maximum(h2_da.density); maximum(h2_ada.density)])), "k")
PyPlot.xlabel(L"log $\phi$",fontsize=text_size)
PyPlot.ylabel(L"Density",fontsize=text_size)
PyPlot.figure()
PyPlot.plot(h3_mcwm.x,h3_mcwm.density, "b")
PyPlot.hold(true)
PyPlot.plot(h3_da.x,h3_da.density, "r")
PyPlot.plot(h3_ada.x,h3_ada.density, "r--")
PyPlot.plot(x_c3,priordens_c3, "g")
PyPlot.plot((theta_true[3], theta_true[3]), (0, maximum([maximum(h3_mcwm.density);maximum(h3_da.density); maximum(h3_ada.density)])), "k")
PyPlot.xlabel(L"log $\sigma$",fontsize=text_size)
PyPlot.ylabel(L"Density",fontsize=text_size)



################################################################################
###    Compare results for PMCMC, MCWM, DA-GP-MCMC, and ADA-GP-MCMC                                            ###
################################################################################

jobname_pmcmc = "pmcmc"
jobname_mcwm = "mcwm"
jobname_da = "_dagpmcmc_lunarc"
jobname_ada = "_adagpmcmc_lunarc"

savepath = "Ricker model/Results/"


Theta_pmcmc = remove_missing_values(Array(readtable(savepath*"Theta"*jobname_pmcmc*".csv")))
loklik_avec_priorv_pmcmc = Array(readtable(savepath*"loglik_avec_priorvec"*jobname_pmcmc*".csv"))
algorithm_parameters_pmcmc = Array(readtable(savepath*"algorithm_parameters"*jobname_pmcmc*".csv"))

# set values:
loglik_pmcmc = loklik_avec_priorv_pmcmc[1,:]
accept_vec_pmcmc = loklik_avec_priorv_pmcmc[2,:]
prior_vec_pmcmc = loklik_avec_priorv_pmcmc[3,:]
burn_in_pmcmc = Int64(algorithm_parameters_pmcmc[1,1])
theta_true_pmcmc = algorithm_parameters_pmcmc[2:4,1]
theta_0_pmcmc = algorithm_parameters_pmcmc[5:7,1]
prior_parameters_pmcmc = algorithm_parameters_pmcmc[8:end,:]



Theta_mcwm = remove_missing_values(Array(readtable(savepath*"Theta"*jobname_mcwm*".csv")))
loklik_avec_priorv_mcwm = Array(readtable(savepath*"loglik_avec_priorvec"*jobname_mcwm*".csv"))
algorithm_parameters_mcwm = Array(readtable(savepath*"algorithm_parameters"*jobname_mcwm*".csv"))

# set values:
loglik_mcwm = loklik_avec_priorv_mcwm[1,:]
accept_vec_mcwm = loklik_avec_priorv_mcwm[2,:]
prior_vec_mcwm = loklik_avec_priorv_mcwm[3,:]
burn_in_mcwm = Int64(algorithm_parameters_mcwm[1,1])
theta_true = algorithm_parameters_mcwm[2:4,1]
theta_0_mcwm = algorithm_parameters_mcwm[5:7,1]
prior_parameters = algorithm_parameters_mcwm[8:end,:]



Theta_da = remove_missing_values(Array(readtable(savepath*"Theta"*jobname_da*".csv")))
loklik_avec_priorv_da = Array(readtable(savepath*"loglik_avec_priorvec"*jobname_da*".csv"))
algorithm_parameters_da = Array(readtable(savepath*"algorithm_parameters"*jobname_ada*".csv"))

# set values:
loglik_da = loklik_avec_priorv_da[1,:]
accept_vec_da = loklik_avec_priorv_da[2,:]
prior_vec_da = loklik_avec_priorv_da[3,:]
burn_in_da = Int64(algorithm_parameters_da[1,1])
theta_true_da = algorithm_parameters_da[2:4,1]
theta_0_da = algorithm_parameters_da[5:7,1]


Theta_ada = remove_missing_values(Array(readtable(savepath*"Theta"*jobname_ada*".csv")))
loklik_avec_priorv_ada = Array(readtable(savepath*"loglik_avec_priorvec"*jobname_ada*".csv"))
algorithm_parameters_ada = Array(readtable(savepath*"algorithm_parameters"*jobname_ada*".csv"))

# set values:
loglik_ada = loklik_avec_priorv_ada[1,:]
accept_vec_ada = loklik_avec_priorv_ada[2,:]
prior_vec_ada = loklik_avec_priorv_ada[3,:]
burn_in_ada = Int64(algorithm_parameters_ada[1,1])
theta_true_ada = algorithm_parameters_ada[2:4,1]
theta_0_ada = algorithm_parameters_ada[5:7,1]

# kernel density for posteriors
h1_pmcmc = kde(Theta_pmcmc[1,burn_in_pmcmc:end])
h2_pmcmc = kde(Theta_pmcmc[2,burn_in_pmcmc:end])
h3_pmcmc = kde(Theta_pmcmc[3,burn_in_pmcmc:end])


h1_mcwm = kde(Theta_mcwm[1,burn_in_mcwm:end])
h2_mcwm = kde(Theta_mcwm[2,burn_in_mcwm:end])
h3_mcwm = kde(Theta_mcwm[3,burn_in_mcwm:end])

h1_da = kde(Theta_da[1,burn_in_da:end])
h2_da = kde(Theta_da[2,burn_in_da:end])
h3_da = kde(Theta_da[3,burn_in_da:end])

h1_ada = kde(Theta_ada[1,burn_in_ada:end])
h2_ada = kde(Theta_ada[2,burn_in_ada:end])
h3_ada = kde(Theta_ada[3,burn_in_ada:end])


# prior
x_c1 = prior_parameters[1,1]-0.5:0.01:prior_parameters[1,2]+0.5
x_c2 = prior_parameters[2,1]-0.5:0.01:prior_parameters[2,2]+0.5
x_c3 = prior_parameters[3,1]-0.5:0.01:prior_parameters[3,2]+0.5


priordens_c1 = pdf.(Uniform(prior_parameters[1,1], prior_parameters[1,2]), x_c1)
priordens_c2 = pdf.(Uniform(prior_parameters[2,1], prior_parameters[2,2]), x_c2)
priordens_c3 = pdf.(Uniform(prior_parameters[3,1], prior_parameters[3,2]), x_c3)

text_size = 25
label_size = 20

PyPlot.figure(figsize=(10,10))
ax = axes()
PyPlot.plot(h1_pmcmc.x,h1_pmcmc.density, "b")
PyPlot.plot(h1_mcwm.x,h1_mcwm.density, "b--")
PyPlot.hold(true)
PyPlot.plot(h1_da.x,h1_da.density, "r")
PyPlot.plot(h1_ada.x,h1_ada.density, "r--")
PyPlot.plot(x_c1,priordens_c1, "g")
PyPlot.plot((theta_true[1], theta_true[1]), (0, maximum([maximum(h1_pmcmc.density); maximum(h1_mcwm.density);maximum(h1_da.density); maximum(h1_ada.density)]) ), "k")
#PyPlot.xlabel(L"log $r$",fontsize=text_size)
#PyPlot.ylabel(L"Density",fontsize=text_size)
PyPlot.xlim((minimum(h1_pmcmc.x)-0.2, maximum(h1_pmcmc.x)+0.2))
ax[:tick_params]("both",labelsize=label_size)

PyPlot.figure(figsize=(10,10))
ax = axes()
PyPlot.plot(h2_pmcmc.x,h2_pmcmc.density, "b")
PyPlot.plot(h2_mcwm.x,h2_mcwm.density, "b--")
PyPlot.hold(true)
PyPlot.plot(h2_da.x,h2_da.density, "r")
PyPlot.plot(h2_ada.x,h2_ada.density, "r--")
PyPlot.plot(x_c2,priordens_c2, "g")
PyPlot.plot((theta_true[2], theta_true[2]), (0, maximum([maximum(h2_pmcmc.density); maximum(h2_mcwm.density);maximum(h2_da.density); maximum(h2_ada.density)]) ), "k")
#PyPlot.xlabel(L"log $\phi$",fontsize=text_size)
#PyPlot.ylabel(L"Density",fontsize=text_size)
PyPlot.xlim((minimum(h2_pmcmc.x)-0.07, maximum(h2_pmcmc.x)+0.07))
ax[:tick_params]("both",labelsize=label_size)


PyPlot.figure(figsize=(10,10))
ax = axes()
PyPlot.plot(h3_pmcmc.x,h3_pmcmc.density, "b")
PyPlot.plot(h3_mcwm.x,h3_mcwm.density, "b--")
PyPlot.hold(true)
PyPlot.plot(h3_da.x,h3_da.density, "r")
PyPlot.plot(h3_ada.x,h3_ada.density, "r--")
PyPlot.plot(x_c3,priordens_c3, "g")
PyPlot.plot((theta_true[3], theta_true[3]), (0, maximum([maximum(h3_pmcmc.density); maximum(h3_mcwm.density);maximum(h3_da.density); maximum(h3_ada.density)]) ), "k")
#PyPlot.xlabel(L"log $\sigma$",fontsize=text_size)
#PyPlot.ylabel(L"Density",fontsize=text_size)
PyPlot.xlim((minimum(h3_pmcmc.x)-0.25, maximum(h3_pmcmc.x)+0.25))
ax[:tick_params]("both",labelsize=label_size)
