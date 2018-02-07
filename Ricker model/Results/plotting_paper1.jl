# todo

# plots in the paper:

# plot for MCWM results

# plots for DA

# Plots for A-DA

# Combined posterior plots for MCWM, DA, and A-DA

# plots in Appendix:

# Plots for tranining data: chains + histograms

# all quality eval plots (seperate file in quality eval.):

# 1) predictions

# 2) residuals vs covariates

# 3) residuals vs loglik

# 4) normplot of residuals

# Tabels:

# Show parameter results including standard deiviations for all algorithm

# Show run time for the different algorithms

# Showing how ofter the assumption holds

# load packages
using Plots
using PyPlot
using StatPlots
using StatsBase
using DataFrames

include("plotting.jl")

doc"""
    compute_results(Theta, burn_in)

Computes the results, i.e. computes posterior mean, posterior std,
and posterior quantiles.
"""
function compute_results(Theta, burn_in)

  m = mean(Theta[:,burn_in+1:end],2)
  s = std(Theta[:,burn_in+1:end], 2)


  I_cred = zeros(3,2)

  for i = 1:size(Theta,1)
    I_cred[i,:] = quantile(Theta[i,burn_in+1:end], [0.05 0.95])
  end

  println("mean:")
  println(m)

  println("std:")
  println(s)

  println("I_cred:")
  println(I_cred)

  return m,s,I_cred

end


# MCWM

jobname = "mcwm" # lunarc


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

compute_results(Theta,burn_in)

burn_in_mcwm = burn_in
Theta_mcwm  = Theta

# DA

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

compute_results(Theta,burn_in)

Theta_da  = Theta



# ADA

jobname = "accelerated"



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

compute_results(Theta,burn_in)

Theta_ada  = Theta

# plot marginal posterior

# Posterior
x_c1 = Theta_parameters[1,1]-0.5:0.01:Theta_parameters[1,2]+0.5
x_c2 = Theta_parameters[2,1]-0.5:0.01:Theta_parameters[2,2]+0.5
x_c3 = Theta_parameters[3,1]-0.5:0.01:Theta_parameters[3,2]+0.5


priordens_c1 = pdf(Uniform(Theta_parameters[1,1], Theta_parameters[1,2]), x_c1)
priordens_c2 = pdf(Uniform(Theta_parameters[2,1], Theta_parameters[2,2]), x_c2)
priordens_c3 = pdf(Uniform(Theta_parameters[3,1], Theta_parameters[3,2]), x_c3)

h1 = kde(Theta_mcwm[1,burn_in_mcwm:end])
h2 = kde(Theta_mcwm[2,burn_in_mcwm:end])
h3 = kde(Theta_mcwm[3,burn_in_mcwm:end])

h1da = kde(Theta_da[1,:])
h2da = kde(Theta_da[2,:])
h3da = kde(Theta_da[3,:])

h1ada = kde(Theta_ada[1,:])
h2ada = kde(Theta_ada[2,:])
h3ada = kde(Theta_ada[3,:])

text_size = 15
label_size = 15


PyPlot.figure()
ax = axes()

subplot(311)
PyPlot.plot(h1.x,h1.density, "b")
PyPlot.hold(true)
PyPlot.plot(h1da.x,h1da.density, "r")
PyPlot.plot(h1ada.x,h1da.density, "r", linestyle="--")
PyPlot.plot(x_c1,priordens_c1, "g")
PyPlot.plot((theta_true[1], theta_true[1]), (0, maximum(h1.density)), "k")
PyPlot.ylabel(L"log $r$",fontsize=text_size)

subplot(312)
PyPlot.plot(h2.x,h2.density, "b")
PyPlot.hold(true)
PyPlot.plot(h2da.x,h2da.density, "r")
PyPlot.plot(h2ada.x,h2ada.density, "r", linestyle="--")
PyPlot.plot(x_c2,priordens_c2, "g")
PyPlot.plot((theta_true[2], theta_true[2]), (0, maximum(h2.density)), "k")
PyPlot.ylabel(L"log $\phi$",fontsize=text_size)

subplot(313)
PyPlot.plot(h3.x,h3.density, "b")
PyPlot.hold(true)
PyPlot.plot(h3da.x,h3da.density, "r")
PyPlot.plot(h3ada.x,h3ada.density, "r", linestyle="--")
PyPlot.plot(x_c3,priordens_c3, "g")
PyPlot.plot((theta_true[3], theta_true[3]), (0, maximum(h3.density)), "k")
PyPlot.ylabel(L"log $\sigma$",fontsize=text_size)
ax[:tick_params]("both",labelsize=label_size)
