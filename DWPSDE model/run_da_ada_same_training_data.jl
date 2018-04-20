# set up

# set correct path
try
  cd("DWPSDE model")
catch
  warn("Already in the DWPSDE model folder")
end

# load case models
cd("..")
include(pwd()*"\\select case\\selectcase.jl")
cd("DWPSDE model")

include("set_up.jl")

using JLD
using HDF5

################################################################################
##      parameters                                      					            ##
################################################################################

# set parameters for all jobs

# nbr parameters
set_nbr_params = 7

# nbr cores
nbr_of_cores = 4 # was 10

# length burn-in
burn_in = 1

# nbr iterations
nbr_iterations = 100 # should be 20000

# length training data
length_training_data = 5000 # thid should ne 5000

# log-scale priors
log_scale_prior = false

# algorithm
mcmc_alg = "MCWM"  # set MCWM or PMCMC

# prob run MH update
beta_MH = 0.15 # should be 0.1

# load training data
load_tranining_data = true

# type of job
job = "simdata" # set work to simdata or new_data

# set jod dep. parameters
if job == "simdata"

	# jobname
	global_jobname = "est7"*job

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
	global_jobname = "est7"*job

	# nbr particels
	nbr_particels = 500

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
##                         training data                                      ##
################################################################################


# set parameters
burn_in_tranining = 10000 # this should be 2000 when estimating 2 parameters

if set_nbr_params == 2
	burn_in_tranining = 1000 # this should be 2000
	length_training_data = 2000
end

nbr_iterations_tranining = burn_in_tranining +length_training_data
nbr_particels_tranining = 25 # should be 200
nbr_of_cores_tranining = 2 # should be 8


################################################################################
##      create (A)DA problem                                     					    ##
################################################################################

problem = set_up_gp_problem(nbr_of_unknown_parameters=set_nbr_params, use_sim_data = sim_data, data_set = data_set)
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = 1
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.alg_param.length_training_data = length_training_data


problem.adaptive_update =  AMUpdate_gen(eye(set_nbr_params), 1/sqrt(set_nbr_params), 0.2, 1, 0.8, 25) # was 0.3

problem.alg_param.dt = dt
problem.alg_param.dt_U = dt_U
problem.alg_param.alg = mcmc_alg
problem.alg_param.compare_GP_and_PF = false
problem.alg_param.noisy_est = false
problem.alg_param.pred_method = "sample"
problem.alg_param.length_training_data = length_training_data
problem.alg_param.beta_MH = beta_MH


################################################################################
##                         set model parameters  for training                               ##
################################################################################

problem_training = set_up_problem(nbr_of_unknown_parameters=set_nbr_params, use_sim_data = sim_data, data_set = data_set)
problem_training.alg_param.alg = mcmc_alg
problem_training.alg_param.R = nbr_iterations_tranining
problem_training.alg_param.N = nbr_particels_tranining
problem_training.alg_param.burn_in = burn_in_tranining
problem_training.alg_param.nbr_of_cores = nbr_of_cores_tranining
problem_training.alg_param.dt = dt
problem_training.alg_param.dt_U = dt_U
problem_training.alg_param.alg = mcmc_alg
problem_training.adaptive_update =  AMUpdate_gen(eye(set_nbr_params), 1/sqrt(set_nbr_params), 0.15, 1, 0.8, 25)



################################################################################
##                generate training data                                     ###
################################################################################


if !load_tranining_data

	if !log_scale_prior
	  res_training, theta_training, loglik_training, cov_matrix = @time MCMC(problem_training, true, true)
	  #export_parameters(res_problem_normal_prior_est_AM_gen[2],jobname)
	else
	  res_training, theta_training, loglik_training, cov_matrix  = @time MCMC(problem_training_nonlog, true, true)
	end

	export_data(problem_training, res_training[1],"da_ada_gpMCMC_training_data"*global_jobname)

	save("gp_training_$(set_nbr_params)_par_test_new_code.jld", "res_training", res_training, "theta_training", theta_training, "loglik_training", loglik_training,"cov_matrix",cov_matrix)

else

	if job == "simdata"
		@load "gp_training_7_par_training_and_test_lunarc.jld"
    @load "fited_gp_simdata.jld"
	elseif job == "new_data"
		@load "gp_training_7_par_training_and_test_new_data.jld"
	end


end




################################################################################
###            plot training data                                            ###
################################################################################

#export_data(problem_training, res_training[1],"da_ada_gpMCMC_training_data"*global_jobname)

#=
using PyPlot


for i = 1:set_nbr_params
  PyPlot.figure()
  PyPlot.plot(res_training[1].Theta_est[i,:])
end

for i = 1:set_nbr_params
  PyPlot.figure()
  PyPlot.plt[:hist](theta_training[i,:],100)
end

PyPlot.figure()
PyPlot.plt[:hist](loglik_training,100)
=#

println("Mean:")
println(mean(theta_training,2))
println("Std:")
println(std(theta_training,2))


################################################################################
###          Fit GP model                                                     ##
################################################################################

# create gp object
gp = GPModel("est_method",zeros(6), zeros(4),
eye(length_training_data-20), zeros(length_training_data-20),zeros(2,length_training_data-20),
collect(1:10))

# fit gp model

tic()

data_training = [theta_training; loglik_training']
data_test = [theta_test; loglik_test']

show(data_training[:,1:10])

# fit GP model
if true #problem.alg_param.est_method == "ml"
  # fit GP model using ml
  #perc_outlier = 0.1 # used when using PMCMC for trainig data 0.05
  #tail_rm = "left"

  perc_outlier = 0.02 # should be 0.05
  tail_rm = "left"
  lasso = false # should be true

  ml_est(gp, data_training,"SE", lasso,perc_outlier,tail_rm)
else
  error("The two stage estimation method is not in use")
  #two_stage_est(gp, data_training)
end

time_fit_gp = toc()


################################################################################
##                         set DA problem                               ##
################################################################################


accelerated_da = false

jobname = global_jobname*"da_gp_mcmc"

problem = set_up_gp_problem(nbr_of_unknown_parameters=set_nbr_params, use_sim_data = sim_data, data_set = data_set)
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = 1
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.alg_param.length_training_data = length_training_data

problem.adaptive_update =  AMUpdate_gen(eye(set_nbr_params), 1/sqrt(set_nbr_params), 0.2, 1, 0.8, 25) # was 0.3


problem.alg_param.alg = mcmc_alg
problem.alg_param.compare_GP_and_PF = false
problem.alg_param.noisy_est = false
problem.alg_param.pred_method = "sample"
problem.alg_param.length_training_data = length_training_data
problem.alg_param.nbr_predictions = 1
problem.alg_param.beta_MH = beta_MH

#problem.alg_param.print_interval = 500
problem.alg_param.selection_method = "max_loglik"  # "local_loglik_approx" # "max_loglik"




################################################################################
##               Run DA-GP-MCMC                                              ###
################################################################################

# w/o profiling

problem.model_param.theta_0 = mean(theta_training,2)


if !log_scale_prior
  # run adaptive PMCMC
  res = dagpmcmc(problem_training, problem, gp, cov_matrix)

  mcmc_results = Result(res[1][1].Theta_est, res[1][1].loglik_est, res[1][1].accept_vec, res[1][1].prior_vec)

  # plot results
  export_data(problem, mcmc_results,jobname)
  #export_parameters(mcmc_results[2],jobname)
else
  # run adaptive PMCMC
  res, res_traning, theta_training, loglik_training, assumption_list, loglik_list = dagpMCMC(problem_training, problem, gp, cov_matrix)

  mcmc_results = Result(res[1][1].Theta_est, res[1][1].loglik_est, res[1][1].accept_vec, res[1][1].prior_vec)

  # plot results
  export_data(problem_nonlog, mcmc_results,jobname)
  #export_parameters(mcmc_results[2],jobname)
end



################################################################################
##               Run DA-GP-MCMC                                              ###
################################################################################

Profile.clear()
Profile.clear_malloc_data()
Profile.init(n = 10^7, delay = 0.01)

problem.model_param.theta_0 = mean(theta_training,2)


if !log_scale_prior
  # run adaptive PMCMC
  res = @profile dagpmcmc(problem_training, problem, gp, cov_matrix)

  mcmc_results = Result(res[1][1].Theta_est, res[1][1].loglik_est, res[1][1].accept_vec, res[1][1].prior_vec)

  # plot results
  export_data(problem, mcmc_results,jobname)
  #export_parameters(mcmc_results[2],jobname)
else
  # run adaptive PMCMC
  res, res_traning, theta_training, loglik_training, assumption_list, loglik_list = dagpMCMC(problem_training, problem, gp, cov_matrix)

  mcmc_results = Result(res[1][1].Theta_est, res[1][1].loglik_est, res[1][1].accept_vec, res[1][1].prior_vec)

  # plot results
  export_data(problem_nonlog, mcmc_results,jobname)
  #export_parameters(mcmc_results[2],jobname)
end



# plot profiler results

using PyPlot
using ProfileView
ProfileView.view()
ProfileView.view(colorgc=false)

# save results

li, lidict = Profile.retrieve()
@save  "da_profiler_res.jlprof"  li lidict

# load results

try
  cd("DWPSDE model")
catch
  warn("Already in the DWPSDE model folder")
end

using JLD
using HDF5
using ProfileView

@load "da_profiler_res.jlprof"

ProfileView.view(li, lidict=lidict)
ProfileView.view(li, lidict=lidict, colorgc=false)

################################################################################
##                         set A-DA problem                               ##
################################################################################

accelerated_da = true

jobname = global_jobname*"ada_gp_mcmc"

problem = set_up_gp_problem(nbr_of_unknown_parameters=set_nbr_params, use_sim_data = sim_data, data_set = data_set)
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = 1
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.alg_param.length_training_data = length_training_data

problem.adaptive_update =  AMUpdate_gen(eye(set_nbr_params), 1/sqrt(set_nbr_params), 0.2, 1, 0.8, 25) # was 0.3


problem.alg_param.alg = mcmc_alg
problem.alg_param.compare_GP_and_PF = false
problem.alg_param.noisy_est = false
problem.alg_param.pred_method = "sample"
problem.alg_param.length_training_data = length_training_data
problem.alg_param.nbr_predictions = 1
problem.alg_param.beta_MH = beta_MH

#problem.alg_param.print_interval = 500
problem.alg_param.selection_method = "max_loglik"  # "local_loglik_approx" # "max_loglik"


################################################################################
##               Set std_limit                                               ###
################################################################################

#=
theta_gp_test_m = res_training[1].Theta_est[:,problem_training.alg_param.burn_in:end]
noisy_est = problem.alg_param.noisy_est

(mean_pred_ml_vec, std_pred_ml_vec, prediction_sample_ml)  = predict(theta_gp_test_m,gp,noisy_est)

problem.alg_param.std_limit = percentile(std_pred_ml_vec, 50)
=#

problem.alg_param.std_limit = 100

################################################################################
#                Estimate probabilities
################################################################################


println(problem.model_param.theta_true)

println(set_nbr_params)

println(problem.model_param)

n = size(data_training,2)
n_burn_in = problem_training.alg_param.burn_in
dim = set_nbr_params # was length(problem.model_param.theta_true)
data_signs = zeros(4,n)
data_signs[3,:] = data_training[dim+1,:]
data_signs[4,:] = res_training[1].loglik_est[end-n+1:end]

noisy_pred = problem.alg_param.noisy_est


for i = 1:n
  (loglik_est_star, var_pred_ml, prediction_sample_ml) = predict(data_training[1:dim,i],gp,noisy_pred)
  (loglik_est_old, var_pred_ml, prediction_sample_ml) = predict(res_training[1].Theta_est[:,i+n_burn_in],gp,noisy_pred)
  data_signs[1,i] = loglik_est_star[1]
  data_signs[2,i] = loglik_est_old[1]
end

nbr_GP_star_geq_GP_old = zero(Int64)
nbr_case_1 = zero(Int64)
nbr_case_4 = zero(Int64)

for i = 1:n
  if data_signs[1,i] > data_signs[2,i]
    nbr_GP_star_geq_GP_old += 1
    if data_signs[3,i] > data_signs[4,i]
      nbr_case_1 += 1
    end
  elseif data_signs[3,i] > data_signs[4,i]
    nbr_case_4 += 1
  end
end

nbr_GP_star_led_GP_old = n-nbr_GP_star_geq_GP_old

prob_case_1 = nbr_case_1/nbr_GP_star_geq_GP_old
prob_case_2 = (nbr_GP_star_led_GP_old-nbr_case_4)/nbr_GP_star_led_GP_old
prob_case_3 = 1-prob_case_1
prob_case_4 = nbr_case_4/nbr_GP_star_led_GP_old

prob_cases = [prob_case_1;prob_case_2;prob_case_3;prob_case_4]

println("Estimated probabilities:")
println(prob_cases)

#prob_cases = [0.377462; 0.940529; 0.622538; 0.0594714]


################################################################################
##               Run A-DA-GP-MCMC                                              ###
################################################################################

# w/o profiling

# start at mode (median)!
problem.model_param.theta_0 = mean(theta_training,2)


if !log_scale_prior
  # run adaptive PMCMC
  res = adagpmcmc(problem_training, problem, gp, cov_matrix,prob_cases)

  mcmc_results = Result(res[1][1].Theta_est, res[1][1].loglik_est, res[1][1].accept_vec, res[1][1].prior_vec)

  # plot results
  export_data(problem, mcmc_results,jobname)
  #export_parameters(mcmc_results[2],jobname)
else
  # run adaptive PMCMC
  res, res_traning, theta_training, loglik_training, assumption_list, loglik_list = adagpMCMC(problem_training, problem, gp, cov_matrix,prob_cases)

  mcmc_results = Result(res[1][1].Theta_est, res[1][1].loglik_est, res[1][1].accept_vec, res[1][1].prior_vec)

  # plot results
  export_data(problem_nonlog, mcmc_results,jobname)
  #export_parameters(mcmc_results[2],jobname)
end

################################################################################
##               Run A-DA-GP-MCMC                                              ###
################################################################################


Profile.clear()
Profile.clear_malloc_data()
Profile.init(n = 10^7, delay = 0.01)


# start at mode (median)!
problem.model_param.theta_0 = mean(theta_training,2)


if !log_scale_prior
  # run adaptive PMCMC
  res = @profile adagpmcmc(problem_training, problem, gp, cov_matrix,prob_cases)

  mcmc_results = Result(res[1][1].Theta_est, res[1][1].loglik_est, res[1][1].accept_vec, res[1][1].prior_vec)

  # plot results
  export_data(problem, mcmc_results,jobname)
  #export_parameters(mcmc_results[2],jobname)
else
  # run adaptive PMCMC
  res, res_traning, theta_training, loglik_training, assumption_list, loglik_list = adagpMCMC(problem_training, problem, gp, cov_matrix,prob_cases)

  mcmc_results = Result(res[1][1].Theta_est, res[1][1].loglik_est, res[1][1].accept_vec, res[1][1].prior_vec)

  # plot results
  export_data(problem_nonlog, mcmc_results,jobname)
  #export_parameters(mcmc_results[2],jobname)
end


using PyPlot
using ProfileView
ProfileView.view()
ProfileView.view(colorgc=false)

# save results

li, lidict = Profile.retrieve()
@save  "ada_slow_prob_profiler_res.jlprof"  li lidict
@save  "ada_fast_prob_profiler_res.jlprof"  li lidict

# load results

try
  cd("DWPSDE model")
catch
  warn("Already in the DWPSDE model folder")
end

using JLD
using HDF5
using ProfileView

@load "ada_slow_prob_profiler_res.jlprof"
@load "ada_fast_prob_profiler_res.jlprof"

ProfileView.view(li, lidict=lidict)
ProfileView.view(li, lidict=lidict, colorgc=false)
