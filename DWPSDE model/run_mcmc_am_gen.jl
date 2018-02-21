# set correct folder

# running the MCMC algorithm using the adaptive AM algorithm for the adaptation

# set correct path
try
  cd("DWPSDE model")
catch
  warn("Already in the Ricker model folder")
end

# load files and functions
include("set_up.jl")

jobname = "test_new_calc_for_a"

# set parameters
nbr_iterations = 2000
nbr_particels = 25
burn_in = 1000
nbr_of_cores = 4
sim_data = true
set_nbr_params = 2
log_scale_prior = false
mcmc_alg = "MCWM"  # set MCWM or PMCMC
data_set = "old"
dt = 0.035 # new = 0.35 old = 0.035
dt_U = 1. # new = 1 old = 1

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
###               Plot data                                                  ###
################################################################################

#using PyPlot

#PyPlot.plot(problem_normal_prior_est_AM_gen.data.Z)

################################################################################
##                Run MCMC and export results                                 ##
################################################################################

if !log_scale_prior
  println("run Normal prior model")
  tic()
  res_problem_normal_prior_est_AM_gen = MCMC(problem_normal_prior_est_AM_gen)
  run_time = toc()
  export_data(problem_normal_prior_est_AM_gen, res_problem_normal_prior_est_AM_gen[1],jobname)
  #export_parameters(res_problem_normal_prior_est_AM_gen[2],jobname)
else
  println("run log-scale prior model")
  tic()
  res_problem_nonlog_prior_est_AM_gen  = @time MCMC(problem_nonlog_prior_est_AM_gen)
  run_time = toc()
  export_data(problem_nonlog_prior_est_AM_gen, res_problem_nonlog_prior_est_AM_gen[1],jobname)
  #export_parameters(res_problem_nonlog_prior_est_AM_gen[2],jobname)
end
