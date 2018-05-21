# set up 

include("set_up.jl")

using JLD
using HDF5

################################################################################
##      parameters for training data                                          ##
################################################################################

# set nbr parameters 
set_nbr_params = 2

# nbr particles 
nbr_particels = 25

# set burn-in 
burn_in = 2000

# set nbr of cores 
nbr_of_cores = 1

# set data to use 
sim_data = true
mcmc_alg = "MCWM"  # set MCWM or PMCMC
data_set = "old"

# scale for prior dist 
log_scale_prior = false

# set dt's 
dt = 0.035 # new = 0.5 old = 0.03
dt_U = 1. # new = 1 old = 1

# set length for training and test data 
length_training_data = 1000
length_test_data = 1000
length_training_data = length_training_data + length_test_data


# set nbr of iterations 
nbr_iterations = burn_in+length_training_data


################################################################################
##                         set model parameters  for training                 ##
################################################################################

problem_training = set_up_problem(nbr_of_unknown_parameters=set_nbr_params, use_sim_data = sim_data, data_set = data_set)
problem_training.alg_param.alg = mcmc_alg
problem_training.alg_param.R = nbr_iterations
problem_training.alg_param.N = nbr_particels
problem_training.alg_param.burn_in = burn_in
problem_training.alg_param.nbr_of_cores = nbr_of_cores
problem_training.alg_param.dt = dt
problem_training.alg_param.dt_U = dt_U
problem_training.alg_param.alg = mcmc_alg
problem_training.adaptive_update =  AMUpdate_gen(eye(set_nbr_params), 1/sqrt(set_nbr_params), 0.15, 1, 0.8, 25)



################################################################################
##                generate training data                                     ###
################################################################################



if !log_scale_prior
	tic()
	res_training, theta_training, loglik_training, cov_matrix = MCMC(problem_training, true, true)
	time_pre_er = toc()
	#export_parameters(res_problem_normal_prior_est_AM_gen[2],jobname)
else
	tic()
	res_training, theta_training, loglik_training, cov_matrix  = @time MCMC(problem_training_nonlog, true, true)
	time_pre_er = toc()
end

################################################################################
##                store generated data                                       ###
################################################################################

# export generated data 
export_data(problem_training, res_training[1],"gp_training_$(set_nbr_params)_par")

# split training and test data

theta_test = theta_training[1:length_training_data+1:end]
loglik_test = loglik_training[1:length_training_data+1:end]

theta_training = theta_training[1:length_training_data]
loglik_training = loglik_training[1:length_training_data]

# save training data, test data, and covaraince matrix to a Julia workspace file 
save("gp_training_$(set_nbr_params)_par_training_and_test_data.jld", "res_training", res_training, "theta_training", theta_training, "loglik_training", loglik_training, "theta_test", theta_test, "loglik_test", loglik_test,"cov_matrix",cov_matrix)




