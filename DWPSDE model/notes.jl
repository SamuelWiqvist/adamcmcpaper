# setttings for generating the traninig data on my local computer


# set parameters for all jobs

# burn-in
burn_in = 10000

# length training data
length_training_data = 5000

# length test data
length_test_data = 5000

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
job = "new_data" # set work to simdata or new_data

# set jod dep. parameters
if job == "simdata"

	# jobname
	jobname = "da_ada_training_data"*job

	# nbr particels
	nbr_particels = 200

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
	jobname = "da_ada_training_data_local"*job

	# nbr particels
	nbr_particels = 250

	# use simulated data
	sim_data = false # set to true to use sim data

	# data set
	data_set = "new" # was "old"

	# dt
	dt = 0.35 # new = 0.35 old = 0.035

	# dt_U
	dt_U = 1. # new = 1 old = 1

end
