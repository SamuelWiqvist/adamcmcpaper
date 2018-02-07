# set correct folder

# running the MCMC algorithm using the adaptive AM algorithm for the adaptation

# load files and functions
include("set_up.jl")


jobname = "test_new_pf_2"

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

using PyPlot

PyPlot.plot(problem_normal_prior_est_AM_gen.data.Z)


################################################################################
##                Run MCMC and export results                                 ##
################################################################################

if !log_scale_prior
  tic()
  res_problem_normal_prior_est_AM_gen = MCMC(problem_normal_prior_est_AM_gen)
  run_time = toc()
  export_data(problem_normal_prior_est_AM_gen, res_problem_normal_prior_est_AM_gen[1],jobname)
  #export_parameters(res_problem_normal_prior_est_AM_gen[2],jobname)
else
  tic()
  res_problem_nonlog_prior_est_AM_gen  = @time MCMC(problem_nonlog_prior_est_AM_gen)
  run_time = toc()
  export_data(problem_nonlog_prior_est_AM_gen, res_problem_nonlog_prior_est_AM_gen[1],jobname)
  #export_parameters(res_problem_nonlog_prior_est_AM_gen[2],jobname)
end


#=
jobname = "test_new_data_est_2_param"

if !log_scale_prior
  # run adaptive PMCMC
  res = gpPMCMC(problem)

  mcmc_results = Result(res[1].Theta_est, res[1].loglik_est, res[1].accept_vec, res[1].prior_vec)

  # plot results
  export_data(problem, mcmc_results,jobname)
  #export_parameters(mcmc_results[2],jobname)
else
  # run adaptive PMCMC
  res = gpPMCMC(problem2)

  mcmc_results = Result(res[1].Theta_est, res[1].loglik_est, res[1].accept_vec, res[1].prior_vec)

  # plot results
  export_data(problem_nonlog, mcmc_results,jobname)
  #export_parameters(mcmc_results[2],jobname)
end

=#

# parameter settings
#=
# est 3 parameters

problem_normal_prior_est_3_AM_gen = set_up_problem(nbr_of_unknown_parameters=3,use_sim_data = sim_data)
problem_normal_prior_est_3_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_3_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_3_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_3_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_3_AM_gen.adaptive_update =  AMUpdate_gen(eye(3), 1/sqrt(3), 0.15, 1, 0.8, 25)

problem_nonlog_prior_est_3_AM_gen = set_up_problem(nbr_of_unknown_parameters=3, prior_dist="nonlog")
problem_nonlog_prior_est_3_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_3_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_3_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_3_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_3_AM_gen.adaptive_update =  AMUpdate_gen(eye(3), 1/sqrt(3), 0.24, 1, 0.8, 25)


# est 4 parameters

problem_normal_prior_est_4_AM_gen = set_up_problem(nbr_of_unknown_parameters=4)
problem_normal_prior_est_4_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_4_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_4_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_4_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_4_AM_gen.adaptive_update =  AMUpdate_gen(eye(4), 1/sqrt(4), 0.2, 1, 0.7, 25)


problem_nonlog_prior_est_4_AM_gen = set_up_problem(nbr_of_unknown_parameters=4, prior_dist="nonlog")
problem_nonlog_prior_est_4_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_4_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_4_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_4_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_4_AM_gen.adaptive_update =  AMUpdate_gen(eye(4), 1/sqrt(4), 0.24, 1, 0.8, 25)



# est 5 parameters

problem_normal_prior_est_5_AM_gen = set_up_problem(nbr_of_unknown_parameters=5,use_sim_data = sim_data)
problem_normal_prior_est_5_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_5_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_5_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_5_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_5_AM_gen.adaptive_update =  AMUpdate_gen(eye(5), 1/sqrt(5), 0.3, 1, 0.7, 25)


problem_nonlog_prior_est_5_AM_gen = set_up_problem(nbr_of_unknown_parameters=5, prior_dist="nonlog")
problem_nonlog_prior_est_5_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_5_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_5_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_5_AM_gen.adaptive_update =  AMUpdate_gen(eye(5), 1/sqrt(5), 0.2, 1, 0.8, 25)

# est 6 parameters


problem_normal_prior_est_6_AM_gen = set_up_problem(nbr_of_unknown_parameters=6)
problem_normal_prior_est_6_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_6_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_6_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_6_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_6_AM_gen.adaptive_update =  AMUpdate_gen(eye(6), 1/sqrt(6), 0.2, 1, 0.7, 25)

problem_nonlog_prior_est_6_AM_gen = set_up_problem(nbr_of_unknown_parameters=6, prior_dist="nonlog")
problem_nonlog_prior_est_6_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_6_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_6_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_6_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_6_AM_gen.adaptive_update =  AMUpdate_gen(eye(6), 1/sqrt(6), 0.2, 1, 0.7, 25)


# est 7 parameters


problem_normal_prior_est_7_AM_gen = set_up_problem(nbr_of_unknown_parameters=7,use_sim_data = sim_data)
problem_normal_prior_est_7_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_7_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_7_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_7_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_7_AM_gen.adaptive_update =  AMUpdate_gen(eye(7), 1/sqrt(7), 0.2, 1, 0.7, 25)


problem_nonlog_prior_est_7_AM_gen = set_up_problem(nbr_of_unknown_parameters=7, prior_dist="nonlog")
problem_nonlog_prior_est_7_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_7_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_7_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_7_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_7_AM_gen.adaptive_update =  AMUpdate_gen(eye(7), 1/sqrt(7), 0.2, 1, 0.7, 25)


# est 8 parameters


problem_normal_prior_est_8_AM_gen = set_up_problem(nbr_of_unknown_parameters=8)
problem_normal_prior_est_8_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_8_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_8_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_8_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_8_AM_gen.adaptive_update =  AMUpdate_gen(eye(8), 1/sqrt(8), 0.2, 1, 0.7, 25)


problem_nonlog_prior_est_8_AM_gen = set_up_problem(nbr_of_unknown_parameters=8, prior_dist="nonlog")
problem_nonlog_prior_est_8_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_8_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_8_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_8_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_8_AM_gen.adaptive_update =  AMUpdate_gen(eye(8), 1/sqrt(8), 0.2, 1, 0.7, 25)



# est 9 parameters

problem_normal_prior_est_9_AM_gen = set_up_problem(nbr_of_unknown_parameters=9,use_sim_data = sim_data)
problem_normal_prior_est_9_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_9_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_9_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_9_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_9_AM_gen.adaptive_update =  AMUpdate_gen(eye(9), 1/sqrt(9), 0.2, 1, 0.7, 25)

problem_nonlog_prior_est_9_AM_gen = set_up_problem(nbr_of_unknown_parameters=9,prior_dist="nonlog")
problem_nonlog_prior_est_9_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_9_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_9_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_9_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_9_AM_gen.adaptive_update =  AMUpdate_gen(eye(9), 1/sqrt(9), 0.2, 1, 0.7, 25)
=#
