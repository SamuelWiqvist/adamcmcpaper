# Script for comparing the posterior distributions of the MCMC algorithm and
# the ER-GP-MCMC algorithm

# load packages
using Plots
using PyPlot
using StatPlots
using DataFrames
using Distributions
using KernelDensity

# results for PMCMC and ER-GP-MCMC

################################################################################
###             Compare one Run                                              ###
################################################################################

# load MCMC file
jobname_mcmc = "res_mcwm" # lunarc
jobname_ergpmcmc = "" # lunarc_beta_mh_100
jobname_ergpmcmc = "accelerated" # lunarc_beta_mh_100

Theta_PMCMC = Array(readtable("./Results/Theta"*jobname_mcmc*".csv"))
loklik_avec_priorv = Array(readtable("./Results/loglik_avec_priorvec"*jobname_mcmc*".csv"))
algorithm_parameters = Array(readtable("./Results/algorithm_parameters"*jobname_mcmc*".csv"))

# set values:
loglik = loklik_avec_priorv[1,:]
accept_vec = loklik_avec_priorv[2,:]
prior_vec = loklik_avec_priorv[3,:]
burn_in = Int64(algorithm_parameters[1,1])
theta_true = algorithm_parameters[2:4,1]
theta_0 = algorithm_parameters[5:7,1]
Theta_parameters = algorithm_parameters[8:end,:]

Theta_ERGPMCMC = Array(readtable("./Results/Theta_ergp"*jobname_ergpmcmc*".csv"))
loklik_avec_priorv = Array(readtable("./Results/loglik_avec_priorvec_ergp"*jobname_ergpmcmc*".csv"))
algorithm_parameters = Array(readtable("./Results/algorithm_parameters_ergp"*jobname_ergpmcmc*".csv"))

start_early_rejection = Int64(algorithm_parameters[1,1])

text_size = 20
label_size = 15

# plot chains

# PMCMC
PyPlot.figure()
ax = axes()
PyPlot.subplot(311)
PyPlot.plot(Theta_PMCMC[1,burn_in:end])
PyPlot.hold(true)
PyPlot.plot(ones(size(Theta_PMCMC[1,burn_in:end],1),1)*theta_true[1], "k")
PyPlot.ylabel(L"log $r$",fontsize=text_size)
PyPlot.subplot(312)
PyPlot.plot(Theta_PMCMC[2,burn_in:end])
PyPlot.hold(true)
PyPlot.plot(ones(size(Theta_PMCMC[1,burn_in:end],1),1)*theta_true[2], "k")
PyPlot.ylabel(L"log $\phi$",fontsize=text_size)
PyPlot.subplot(313)
PyPlot.plot(Theta_PMCMC[3,burn_in:end])
PyPlot.hold(true)
PyPlot.plot(ones(size(Theta_PMCMC[1,burn_in:end],1),1)*theta_true[3], "k")
PyPlot.ylabel(L"log $\sigma$",fontsize=text_size)
ax[:tick_params]("both",labelsize=label_size)

# ER-GP-MCMC
PyPlot.figure()
ax = axes()
PyPlot.subplot(311)
PyPlot.plot(Theta_ERGPMCMC[1,start_early_rejection:end])
PyPlot.hold(true)
PyPlot.plot(ones(size(Theta_ERGPMCMC[1,start_early_rejection:end],1),1)*theta_true[1], "k")
PyPlot.ylabel(L"log $r$",fontsize=text_size)
PyPlot.subplot(312)
PyPlot.plot(Theta_ERGPMCMC[2,start_early_rejection:end])
PyPlot.hold(true)
PyPlot.plot(ones(size(Theta_ERGPMCMC[1,start_early_rejection:end],1),1)*theta_true[2], "k")
PyPlot.ylabel(L"log $\phi$",fontsize=text_size)
PyPlot.subplot(313)
PyPlot.plot(Theta_ERGPMCMC[3,start_early_rejection:end])
PyPlot.hold(true)
PyPlot.plot(ones(size(Theta_ERGPMCMC[1,start_early_rejection:end],1),1)*theta_true[3], "k")
PyPlot.ylabel(L"log $\sigma$",fontsize=text_size)
ax[:tick_params]("both",labelsize=label_size)


std(Theta_PMCMC,2)
std(Theta_ERGPMCMC,2)

# ACF
acf_PMCMC_c1 = autocor(Theta_PMCMC[1,burn_in:end],1:50)
acf_PMCMC_c2 = autocor(Theta_PMCMC[2,burn_in:end],1:50)
acf_PMCMC_c3 = autocor(Theta_PMCMC[3,burn_in:end],1:50)

acf_ERGPMCMC_c1 = autocor(Theta_ERGPMCMC[1,start_early_rejection:end],1:50)
acf_ERGPMCMC_c2 = autocor(Theta_ERGPMCMC[2,start_early_rejection:end],1:50)
acf_ERGPMCMC_c3 = autocor(Theta_ERGPMCMC[3,start_early_rejection:end],1:50)


PyPlot.figure()
ax = axes()
PyPlot.subplot(311)
PyPlot.plot(acf_PMCMC_c1,"--b")
PyPlot.hold(true)
PyPlot.plot(acf_ERGPMCMC_c1,"--r")
PyPlot.ylabel(L"log $r$",fontsize=text_size)
PyPlot.subplot(312)
PyPlot.plot(acf_PMCMC_c2,"--b")
PyPlot.hold(true)
PyPlot.plot(acf_ERGPMCMC_c2,"--r")
PyPlot.ylabel(L"log $\phi$",fontsize=text_size)
PyPlot.subplot(313)
PyPlot.plot(acf_PMCMC_c3,"--b")
PyPlot.hold(true)
PyPlot.plot(acf_ERGPMCMC_c3,"--r")
PyPlot.ylabel(L"log $\sigma$",fontsize=text_size)
ax[:tick_params]("both",labelsize=label_size)


# Posterior
x_c1 = Theta_parameters[1,1]-0.5:0.01:Theta_parameters[1,2]+0.5
x_c2 = Theta_parameters[2,1]-0.5:0.01:Theta_parameters[2,2]+0.5
x_c3 = Theta_parameters[3,1]-0.5:0.01:Theta_parameters[3,2]+0.5


priordens_c1 = pdf(Uniform(Theta_parameters[1,1], Theta_parameters[1,2]), x_c1)
priordens_c2 = pdf(Uniform(Theta_parameters[2,1], Theta_parameters[2,2]), x_c2)
priordens_c3 = pdf(Uniform(Theta_parameters[3,1], Theta_parameters[3,2]), x_c3)

h1 = kde(Theta_PMCMC[1,burn_in:end])
h2 = kde(Theta_PMCMC[2,burn_in:end])
h3 = kde(Theta_PMCMC[3,burn_in:end])

h4 = kde(Theta_ERGPMCMC[1,start_early_rejection:end])
h5 = kde(Theta_ERGPMCMC[2,start_early_rejection:end])
h6 = kde(Theta_ERGPMCMC[3,start_early_rejection:end])


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



################################################################################
###             Compare multiple runs  different beta_MH                                      ###
################################################################################


jobname_mcmc = "pmcmc_lunarc"
jobname_ergpmcmc_001 = jobname_vec[1]
jobname_ergpmcmc_005 = jobname_vec[2]
jobname_ergpmcmc_015 = jobname_vec[3]
jobname_ergpmcmc_025 = jobname_vec[4]

Theta_PMCMC = Array(readtable("./Results/Theta"*jobname_mcmc*".csv"))
loklik_avec_priorv = Array(readtable("./Results/loglik_avec_priorvec"*jobname_mcmc*".csv"))
algorithm_parameters = Array(readtable("./Results/algorithm_parameters"*jobname_mcmc*".csv"))


# time_MCMC: 21.1
# time DA-GP-MCMC: 28.6
# early-rejections: 57502/(100000-10117)
# number direct MH: 10117
# set values:
loglik = loklik_avec_priorv[1,:]
accept_vec = loklik_avec_priorv[2,:]
prior_vec = loklik_avec_priorv[3,:]
burn_in = Int64(algorithm_parameters[1,1])
theta_true = algorithm_parameters[2:4,1]
theta_0 = algorithm_parameters[5:7,1]
Theta_parameters = algorithm_parameters[8:end,:]

Theta_ERGPMCMC_001 = Array(readtable("./Results/Theta_ergp"*jobname_ergpmcmc_001*".csv"))
loklik_avec_priorv_001 = Array(readtable("./Results/loglik_avec_priorvec_ergp"*jobname_ergpmcmc_001*".csv"))
algorithm_parameters_001 = Array(readtable("./Results/algorithm_parameters_ergp"*jobname_ergpmcmc_001*".csv"))
start_early_rejection_001 = Int64(algorithm_parameters_001[1,1])

Theta_ERGPMCMC_005 = Array(readtable("./Results/Theta_ergp"*jobname_ergpmcmc_005*".csv"))
loklik_avec_priorv_005 = Array(readtable("./Results/loglik_avec_priorvec_ergp"*jobname_ergpmcmc_005*".csv"))
algorithm_parameters_005 = Array(readtable("./Results/algorithm_parameters_ergp"*jobname_ergpmcmc_005*".csv"))
start_early_rejection_005 = Int64(algorithm_parameters_005[1,1])

Theta_ERGPMCMC_015 = Array(readtable("./Results/Theta_ergp"*jobname_ergpmcmc_015*".csv"))
loklik_avec_priorv_015 = Array(readtable("./Results/loglik_avec_priorvec_ergp"*jobname_ergpmcmc_015*".csv"))
algorithm_parameters_015 = Array(readtable("./Results/algorithm_parameters_ergp"*jobname_ergpmcmc_015*".csv"))
start_early_rejection_015 = Int64(algorithm_parameters_015[1,1])

Theta_ERGPMCMC_025 = Array(readtable("./Results/Theta_ergp"*jobname_ergpmcmc_025*".csv"))
loklik_avec_priorv_025 = Array(readtable("./Results/loglik_avec_priorvec_ergp"*jobname_ergpmcmc_025*".csv"))
algorithm_parameters_025 = Array(readtable("./Results/algorithm_parameters_ergp"*jobname_ergpmcmc_025*".csv"))
start_early_rejection_025 = Int64(algorithm_parameters_025[1,1])


# Posterior
x_c1 = Theta_parameters[1,1]-0.5:0.01:Theta_parameters[1,2]+0.5
x_c2 = Theta_parameters[2,1]-0.5:0.01:Theta_parameters[2,2]+0.5
x_c3 = Theta_parameters[3,1]-0.5:0.01:Theta_parameters[3,2]+0.5


priordens_c1 = pdf(Uniform(Theta_parameters[1,1], Theta_parameters[1,2]), x_c1)
priordens_c2 = pdf(Uniform(Theta_parameters[2,1], Theta_parameters[2,2]), x_c2)
priordens_c3 = pdf(Uniform(Theta_parameters[3,1], Theta_parameters[3,2]), x_c3)

h1 = kde(Theta_PMCMC[1,burn_in:end])
h2 = kde(Theta_PMCMC[2,burn_in:end])
h3 = kde(Theta_PMCMC[3,burn_in:end])

h1001 = kde(Theta_ERGPMCMC_001[1,start_early_rejection_001:end])
h2001 = kde(Theta_ERGPMCMC_001[2,start_early_rejection_001:end])
h3001 = kde(Theta_ERGPMCMC_001[3,start_early_rejection_001:end])

h1005 = kde(Theta_ERGPMCMC_005[1,start_early_rejection_005:end])
h2005 = kde(Theta_ERGPMCMC_005[2,start_early_rejection_005:end])
h3005 = kde(Theta_ERGPMCMC_005[3,start_early_rejection_005:end])


h1015 = kde(Theta_ERGPMCMC_015[1,start_early_rejection_015:end])
h2015 = kde(Theta_ERGPMCMC_015[2,start_early_rejection_015:end])
h3015 = kde(Theta_ERGPMCMC_015[3,start_early_rejection_015:end])


h1025 = kde(Theta_ERGPMCMC_025[1,start_early_rejection_025:end])
h2025 = kde(Theta_ERGPMCMC_025[2,start_early_rejection_025:end])
h3025 = kde(Theta_ERGPMCMC_025[3,start_early_rejection_025:end])



text_size = 20
label_size = 15


PyPlot.figure()
ax = axes()

subplot(311)
PyPlot.plot(h1.x,h1.density, "b")
PyPlot.hold(true)
PyPlot.plot(h1001.x,h1001.density, "r")
PyPlot.plot(h1005.x,h1005.density, "r", linestyle="--")
PyPlot.plot(h1015.x,h1015.density, "r", linestyle="-.")
PyPlot.plot(h1025.x,h1025.density, "r", linestyle=":")
PyPlot.plot(x_c1,priordens_c1, "g")
PyPlot.plot((theta_true[1], theta_true[1]), (0, maximum(h1.density)), "k")
PyPlot.ylabel(L"log $r$",fontsize=text_size)

subplot(312)
PyPlot.plot(h2.x,h2.density, "b")
PyPlot.hold(true)
PyPlot.plot(h2001.x,h2001.density, "r")
PyPlot.plot(h2005.x,h2005.density, "r", linestyle="--")
PyPlot.plot(h2015.x,h2015.density, "r", linestyle="-.")
PyPlot.plot(h2025.x,h2025.density, "r", linestyle=":")
PyPlot.plot(x_c2,priordens_c2, "g")
PyPlot.plot((theta_true[2], theta_true[2]), (0, maximum(h2.density)), "k")
PyPlot.ylabel(L"log $\phi$",fontsize=text_size)

subplot(313)
PyPlot.plot(h3.x,h3.density, "b")
PyPlot.hold(true)
PyPlot.plot(h3001.x,h3001.density, "r")
PyPlot.plot(h3005.x,h3005.density, "r", linestyle="--")
PyPlot.plot(h3015.x,h3015.density, "r", linestyle="-.")
PyPlot.plot(h3025.x,h3025.density, "r", linestyle=":")
PyPlot.plot(x_c3,priordens_c3, "g")
PyPlot.plot((theta_true[3], theta_true[3]), (0, maximum(h3.density)), "k")
PyPlot.ylabel(L"log $\sigma$",fontsize=text_size)
ax[:tick_params]("both",labelsize=label_size)


################################################################################
###             Compare multiple runs  different std_limit                                       ###
################################################################################

include("plotting.jl")

text_size = 20
label_size = 15


Theta_ergp_all = []
loklik_avec_all = []
algorithm_parameters_all = []
start_early_rejection_all = []

for jobname in jobname_vec
  push!(Theta_ergp_all, Array(readtable("./Results/Theta_ergp"*jobname*".csv")))
  push!(loklik_avec_all, Array(readtable("./Results/loglik_avec_priorvec_ergp"*jobname*".csv")))
  algorithm_parameters = Array(readtable("./Results/algorithm_parameters_ergp"*jobname*".csv"))
  push!(algorithm_parameters_all, algorithm_parameters)
  push!(start_early_rejection_all, Int64(algorithm_parameters[1,1]))
end


for i = 1:length(jobname_vec)

  Theta = Theta_ergp_all[i]
  loklik_avec_priorv = loklik_avec_all[i]
  algorithm_parameters = algorithm_parameters_all[i]

  # set values:
  loglik = loklik_avec_priorv[1,:]
  accept_vec = loklik_avec_priorv[2,:]
  prior_vec = loklik_avec_priorv[3,:]
  burn_in = Int64(algorithm_parameters[1,1])
  theta_true = algorithm_parameters[2:4,1]
  theta_0 = algorithm_parameters[5:7,1]
  Theta_parameters = algorithm_parameters[8:end,:]

  analyse_results(Theta, loglik, accept_vec, prior_vec,theta_true, burn_in, Theta_parameters)
end



jobname_mcmc = "test"

Theta_PMCMC = Array(readtable("./Results/Theta"*jobname_mcmc*".csv"))
loklik_avec_priorv = Array(readtable("./Results/loglik_avec_priorvec"*jobname_mcmc*".csv"))
algorithm_parameters = Array(readtable("./Results/algorithm_parameters"*jobname_mcmc*".csv"))

loglik = loklik_avec_priorv[1,:]
accept_vec = loklik_avec_priorv[2,:]
prior_vec = loklik_avec_priorv[3,:]
burn_in = Int64(algorithm_parameters[1,1])
theta_true = algorithm_parameters[2:4,1]
theta_0 = algorithm_parameters[5:7,1]
Theta_parameters = algorithm_parameters[8:end,:]


# Posterior
x_c1 = Theta_parameters[1,1]-0.5:0.01:Theta_parameters[1,2]+0.5
x_c2 = Theta_parameters[2,1]-0.5:0.01:Theta_parameters[2,2]+0.5
x_c3 = Theta_parameters[3,1]-0.5:0.01:Theta_parameters[3,2]+0.5


priordens_c1 = pdf(Uniform(Theta_parameters[1,1], Theta_parameters[1,2]), x_c1)
priordens_c2 = pdf(Uniform(Theta_parameters[2,1], Theta_parameters[2,2]), x_c2)
priordens_c3 = pdf(Uniform(Theta_parameters[3,1], Theta_parameters[3,2]), x_c3)

h1 = kde(Theta_PMCMC[1,burn_in:end])
h2 = kde(Theta_PMCMC[2,burn_in:end])
h3 = kde(Theta_PMCMC[3,burn_in:end])

linestyle_vec = ["-" "--" "-." ":"]


PyPlot.figure()
ax = axes()
for i = 1:length(jobname_vec)
  h1_dagp = kde(Theta_ergp_all[i][1,start_early_rejection_all[i]:end])
  h2_dagp = kde(Theta_ergp_all[i][2,start_early_rejection_all[i]:end])
  h3_dagp = kde(Theta_ergp_all[i][3,start_early_rejection_all[i]:end])


  subplot(311)
  PyPlot.plot(h1.x,h1.density, "b")
  PyPlot.hold(true)
  PyPlot.plot(h1_dagp.x,h1_dagp.density, "r", linestyle=linestyle_vec[i])
  PyPlot.plot(x_c1,priordens_c1, "g")
  PyPlot.plot((theta_true[1], theta_true[1]), (0, maximum(h1.density)), "k")
  PyPlot.ylabel(L"log $r$",fontsize=text_size)

  subplot(312)
  PyPlot.plot(h2.x,h2.density, "b")
  PyPlot.hold(true)
  PyPlot.plot(h2_dagp.x,h2_dagp.density, "r", linestyle=linestyle_vec[i])
  PyPlot.plot(x_c2,priordens_c2, "g")
  PyPlot.plot((theta_true[2], theta_true[2]), (0, maximum(h2.density)), "k")
  PyPlot.ylabel(L"log $\phi$",fontsize=text_size)

  subplot(313)
  PyPlot.plot(h3.x,h3.density, "b")
  PyPlot.hold(true)
  PyPlot.plot(h3_dagp.x,h3_dagp.density, "r", linestyle=linestyle_vec[i])
  PyPlot.plot(x_c3,priordens_c3, "g")
  PyPlot.plot((theta_true[3], theta_true[3]), (0, maximum(h3.density)), "k")
  PyPlot.ylabel(L"log $\sigma$",fontsize=text_size)
  ax[:tick_params]("both",labelsize=label_size)
end
