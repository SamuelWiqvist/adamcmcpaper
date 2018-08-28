# Script for running PMCMC/MCWM

using PyPlot
using ProfileView

include("rickermodel.jl")

# set up problem
problem = set_up_problem(ploton = false)

problem.alg_param.N = 1000
problem.alg_param.R = 50000
problem.alg_param.burn_in = 2000
problem.data.y = Array(readtable("Ricker model/y_data_set_2.csv"))[:,1]

# plot data
text_size = 15

PyPlot.figure(figsize=(20,15))
ax = axes()
PyPlot.plot(problem.data.y)
PyPlot.xlabel("time", fontsize=text_size)
PyPlot.ylabel("y", fontsize=text_size)
ax[:tick_params]("both",labelsize=text_size)

#problem.data.y = Array(readtable("y_data_200_obs_3.csv"))[:,1]

#Array(readtable("y.csv"))[:,1]
#problem.data.y = Array(readtable("y_data_set_abc.csv"))[:,1] #Array(readtable("y.csv"))[:,1]

problem.alg_param.print_interval = 1000 #problem.alg_param.R
# test starting at true parameters

#problem.model_param.theta_0 = problem.model_param.theta_true

# MCWM
jobname = "test_run"
problem.alg_param.alg = "MCWM"
problem.adaptive_update = AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.4, 1., 0.8, 25)

# PMCMC
jobname = "test_run"
problem.alg_param.alg = "PMCMC"
problem.adaptive_update = AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.2, 1., 0.8, 25)

# run adaptive PMCMC

tic()
res_MCMC = mcmc(problem)
time_MCMC = toc()
@printf "Run time (s): %.4f \n" time_MCMC

# write outputs
res = res_MCMC[1]

Theta = res.Theta_est
loglik = res.loglik_est
accept_vec = res.accept_vec
prior_vec = res.prior_vec

loglik_avec_priorvec = zeros(3, length(loglik))
loglik_avec_priorvec[1,:] = loglik
loglik_avec_priorvec[2,:] = accept_vec
loglik_avec_priorvec[3,:] = prior_vec

algorithm_parameters = zeros(10, 2)

algorithm_parameters[1,1] = problem.alg_param.burn_in
algorithm_parameters[2:4,1] = problem.model_param.theta_true
algorithm_parameters[5:7,1] = problem.model_param.theta_0
algorithm_parameters[8:end,:] = problem.prior_dist.prior_parameters

writetable("Ricker model/Results/Theta"*jobname*".csv", convert(DataFrame, Theta))
writetable("Ricker model/Results/loglik_avec_priorvec"*jobname*".csv", convert(DataFrame, loglik_avec_priorvec))
writetable("Ricker model/Results/algorithm_parameters"*jobname*".csv", convert(DataFrame, algorithm_parameters))
