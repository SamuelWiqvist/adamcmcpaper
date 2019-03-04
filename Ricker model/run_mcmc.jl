# Script for running the MCMC algorithm
# set number of threads for BLAS

# load case models
include("rickermodel.jl")


# set up problem
problem = set_up_problem(ploton = false)

problem.alg_param.N = 1000
problem.alg_param.R = 52000
problem.alg_param.burn_in = 2000
problem.data.y = Array(readtable("y_data_set_2.csv"))[:,1]
#problem.data.y = Array(readtable("y_data_200_obs_3.csv"))[:,1]
#problem.data.y = Array(readtable("y_10000_obs.csv"))[:,1]

#Array(readtable("y.csv"))[:,1]
#problem.data.y = Array(readtable("y_data_set_abc.csv"))[:,1] #Array(readtable("y.csv"))[:,1]

problem.alg_param.print_interval = 10000 #problem.alg_param.R
# test starting at true parameters
#problem.model_param.theta_0 = problem.model_param.theta_true

#MCWM
#jobname = "mcwm"
#problem.alg_param.alg = "MCWM"
#problem.adaptive_update = AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.4, 1., 0.8, 25)

# old setting of mcwm 
#problem.adaptive_update = AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.4, 1., 0.8, 25)

# PMCMC
jobname = "pmcmc"
problem.alg_param.alg = "PMCMC"
problem.adaptive_update = AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.4, 1., 0.8, 25)



# use AM alg for adaptive updating
#problem.adaptive_update = AMUpdate(eye(3), 2.4/sqrt(3), 1., 0.7, 25)

#problem.adaptive_update = noAdaptation(2.4/sqrt(3)*eye(3))

# or, use AM gen alg for adaptive updating
#problem.adaptive_update = AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.2, 1., 0.8, 25)

# run adaptive PMCMC
tic()
res_MCMC = mcmc(problem)
time_MCMC = toc()
@printf "Run time (s): %.4f \n" time_MCMC

# profiling
#using ProfileView
#Profile.clear()
#res_MCMC = @profile MCMC(problem)
#ProfileView.view()


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

writetable("Results/Theta"*jobname*".csv", convert(DataFrame, Theta))
writetable("Results/loglik_avec_priorvec"*jobname*".csv", convert(DataFrame, loglik_avec_priorvec))
writetable("Results/algorithm_parameters"*jobname*".csv", convert(DataFrame, algorithm_parameters))
