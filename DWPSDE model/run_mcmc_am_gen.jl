# running the MCMC algorithm using the adaptive AM algorithm for the adaptation

# load files and functions
include("set_up.jl")

# set parameters for all jobs 

# nbr iterations 
nbr_iterations = 30000 # was 30000

# burn-in
burn_in = 10000 # was 10000

# nbr cores 
nbr_of_cores = 4

# nbr parameters 
set_nbr_params = 7

# log-scale prior 
log_scale_prior = false

# algorithm
mcmc_alg = "PMCMC"  # set MCWM or PMCMC

# type of job 
job = "simdata" # set work to simdata or new_data

# set jod dep. parameters 
if job == "simdata"

	# jobname 
	jobname = "mcwm_7_par"*job 
	
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
	jobname = "mcwm_7_par"*job 
	
	# nbr particels 
	nbr_particels = 3*500

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
##                         set model parameters                               ##
################################################################################

problem_normal_prior_est_AM_gen = set_up_problem(nbr_of_unknown_parameters=set_nbr_params, use_sim_data = sim_data, data_set = data_set)
problem_normal_prior_est_AM_gen.alg_param.alg = mcmc_alg
problem_normal_prior_est_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_AM_gen.alg_param.dt = dt
problem_normal_prior_est_AM_gen.alg_param.dt_U = dt_U
problem_normal_prior_est_AM_gen.adaptive_update =  AMUpdate_gen(eye(set_nbr_params), 1/sqrt(set_nbr_params), 0.2, 1, 0.8, 25)


problem_nonlog_prior_est_AM_gen = set_up_problem(nbr_of_unknown_parameters=set_nbr_params, prior_dist="nonlog")
problem_nonlog_prior_est_AM_gen.alg_param.alg = mcmc_alg
problem_nonlog_prior_est_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_AM_gen.adaptive_update =  AMUpdate_gen(eye(set_nbr_params), 1/sqrt(set_nbr_params), 0.2, 1, 0.8, 25)

################################################################################
##                Run MCMC and export results                                 ##
################################################################################

if !log_scale_prior
  println("run Normal prior model")
  tic()
  res_problem_normal_prior_est_AM_gen = mcmc(problem_normal_prior_est_AM_gen)
  run_time = toc()
  export_data(problem_normal_prior_est_AM_gen, res_problem_normal_prior_est_AM_gen[1],jobname)
  #export_parameters(res_problem_normal_prior_est_AM_gen[2],jobname)
else
  println("run log-scale prior model")
  tic()
  res_problem_nonlog_prior_est_AM_gen  = @time mcmc(problem_nonlog_prior_est_AM_gen)
  run_time = toc()
  export_data(problem_nonlog_prior_est_AM_gen, res_problem_nonlog_prior_est_AM_gen[1],jobname)
  #export_parameters(res_problem_nonlog_prior_est_AM_gen[2],jobname)
end
