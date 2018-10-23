# set up 

include("set_up.jl")

using JLD
using HDF5

################################################################################
##      parameters for training data                                          ##
################################################################################


# set parameters for all jobs

# burn-in
burn_in = 10000 # 10000

# length training data 
length_training_data = 5000 # 5000

# length test data 
length_test_data = 5000 # 5000

# nbr iterations 
nbr_iterations = burn_in+length_training_data + length_test_data

# set nbr parameters 
set_nbr_params = 7

# set nbr cores
nbr_of_cores = 4

# log-scale prior
log_scale_prior = false

# algorithm 
mcmc_alg = "MCWM"  # set MCWM or PMCMC

# type of job 
job = ARGS[1] # set work to simdata or new_data

# set jod dep. parameters 
if job == "simdata"

	# jobname 
	jobname = "da_ada_training_data"*job 
	
	# nbr particels 
	nbr_particels = 3*400

	# use simulated data 	
	sim_data = true # set to true to use sim data

	# data set 
	data_set = "old" # was "old"

	# dt 	
	dt = 0.035 # new = 0.35 old = 0.035

	# dt_U 	
	dt_U = 1. # new = 1 old = 1

elseif job == "new_data"

	# jobname 
	jobname = "da_ada_training_data"*job 
	
	# nbr particels 
	nbr_particels = 2000 #3*500

	# use simulated data 	
	sim_data = false # set to true to use sim data

	# data set 
	data_set = "new" # was "old"

	# dt 	
	dt = 0.35 # new = 0.35 old = 0.035

	# dt_U 	
	dt_U = 1. # new = 1 old = 1

end


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
    println("run Normal prior model")
	res_training, theta_training, loglik_training, cov_matrix = mcmc(problem_training, true, true)
	time_pre_er = toc()
	#export_parameters(res_problem_normal_prior_est_AM_gen[2],jobname)
else
	tic()
    println("run log-scale prior model")
	res_training, theta_training, loglik_training, cov_matrix  = @time mcmc(problem_training_nonlog, true, true)
	time_pre_er = toc()
end

################################################################################
##                store generated data                                       ###
################################################################################

# export generated data 
export_data(problem_training, res_training[1],"gp_training_$(set_nbr_params)_par_lunarc_new_data_4_cores"*job)

# split training and test data

theta_test = theta_training[:,(end-length_test_data+1):end]
loglik_test = loglik_training[(end-length_test_data+1):end]


theta_training = theta_training[:,1:length_training_data]
loglik_training = loglik_training[1:length_training_data]


# save training data, test data, and covaraince matrix to a Julia workspace file 
save("gp_training_$(set_nbr_params)_par_training_and_test"*job*"lunarc_new_new.jld", "res_training", res_training, "theta_training", theta_training, "loglik_training", loglik_training, "theta_test", theta_test, "loglik_test", loglik_test,"cov_matrix",cov_matrix)




