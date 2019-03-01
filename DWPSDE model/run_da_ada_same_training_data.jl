# set up

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
nbr_iterations = 10000 #10000 # should be 20000

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
job = ARGS[1] # set work to simdata or new_data

# set jod dep. parameters
if job == "simdata"

	# jobname
	global_jobname = "est7"*job

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
	global_jobname = "est7"*job

	# nbr particels
	nbr_particels = 2000

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
		@load "gp_training_7_par_training_and_testsimdatalunarc_new.jld"
	elseif job == "new_data"
		@load "gp_training_7_par_training_and_testnew_datalunarc_new.jld"
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

data_training = [theta_training; loglik_training']
data_test = [theta_test; loglik_test']

show(data_training[:,1:10])

if true

	# create gp object
	gp = GPModel("est_method",zeros(6), zeros(4),
	eye(length_training_data-20), zeros(length_training_data-20),zeros(2,length_training_data-20),
	collect(1:10))

	# fit gp model

	tic()

	# fit GP model
	if true #problem.alg_param.est_method == "ml"
	  # fit GP model using ml
	  #perc_outlier = 0.0 # used when using PMCMC for trainig data 0.05
	  #tail_rm = "left"

	  perc_outlier = 0.01 # should be 0.05
	  tail_rm = "left"
	  lasso = false # should be true

	  ml_est(gp, data_training,"SE", lasso,perc_outlier,tail_rm)
	else
	  error("The two stage estimation method is not in use")
	  #two_stage_est(gp, data_training)
	end

	time_fit_gp = toc()

end

################################################################################
##               Run DA-GP-MCMC                                              ###
################################################################################


accelerated_da = false

jobname = global_jobname*"da_gp_mcmc"


problem.model_param.theta_0 = mean(theta_training,2)


if !log_scale_prior

  res = dagpmcmc(problem_training, problem, gp, cov_matrix)

  mcmc_results = Result(res[1].Theta_est, res[1].loglik_est, res[1].accept_vec, res[1].prior_vec)

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
##                         set A-DA problem                               ##
################################################################################

accelerated_da = true

jobname = global_jobname*"ada_gp_mcmc_dt"

problem.model_param.theta_0 = mean(res_training[1].Theta_est[:,end-size(data_training,2)-size(data_test,2):end],2)[:]

################################################################################
##  Create features for classification models                                                                            ##
################################################################################

n = size(data_training,2)
n_burn_in = length(res_training[2]) -  length(loglik_training) - length(loglik_test) #problem_traning.alg_param.burn_in

idx_training_start = n_burn_in+1
idx_training_end = idx_training_start+n-1

idx_test_start = idx_training_end+1
idx_test_end = idx_test_start+n-1

idx_training_old = idx_training_start-1:idx_training_end-1
idx_test_old = idx_test_start-1:idx_test_end-1

loglik_training_old = res_training[1].loglik_est[idx_training_old]
loglik_test_old = res_training[1].loglik_est[idx_test_old]


dim = length(problem.model_param.theta_true)
data_signs = zeros(4,n)
data_signs[3,:] = data_training[dim+1,:]
data_signs[4,:] = loglik_training_old

std_pred_gp_star = zeros(n)

noisy_pred = problem.alg_param.noisy_est

for i = 1:n
  (loglik_est_star, var_pred_ml_star, prediction_sample_ml_star) = predict(data_training[1:dim,i],gp,noisy_pred)
  (loglik_est_old, var_pred_ml, prediction_sample_ml_old) = predict(res_training[1].Theta_est[:,idx_training_old[i]],gp,noisy_pred)
  data_signs[1,i] = prediction_sample_ml_star[1]
  data_signs[2,i] = prediction_sample_ml_old[1]
  std_pred_gp_star[i] = sqrt(var_pred_ml_star[1])
end

nbr_GP_star_geq_GP_old = zero(Int64)
nbr_case_1 = zero(Int64)
nbr_case_4 = zero(Int64)

targets_case_1_and_3 = []
data_case_1_and_3 = []

targets_case_2_and_4 = []
data_case_2_and_4 = []

for i = 1:n
  if data_signs[1,i] > data_signs[2,i]
    nbr_GP_star_geq_GP_old += 1
    data_case_1_and_3 = vcat(data_case_1_and_3, [data_training[1:dim,i]; data_signs[1,i]/data_signs[2,i]; std_pred_gp_star[i]])
    if data_signs[3,i] > data_signs[4,i]
      append!(targets_case_1_and_3, 1)
      nbr_case_1 += 1
    else
      append!(targets_case_1_and_3, 0)
    end
  elseif data_signs[3,i] > data_signs[4,i]
    data_case_2_and_4 = vcat(data_case_2_and_4, [data_training[1:dim,i]; data_signs[1,i]/data_signs[2,i]; std_pred_gp_star[i]])
    append!(targets_case_2_and_4, 0)
    nbr_case_4 += 1
  else
    data_case_2_and_4 = vcat(data_case_2_and_4, [data_training[1:dim,i]; data_signs[1,i]/data_signs[2,i]; std_pred_gp_star[i]])
    append!(targets_case_2_and_4, 1)
  end
end


# tansform features and set input data

# convert matricies to floats
data_case_1_and_3 = convert(Array{Float64,2},reshape(data_case_1_and_3, (dim+2, length(targets_case_1_and_3))))
data_case_2_and_4 = convert(Array{Float64,2},reshape(data_case_2_and_4, (dim+2, length(targets_case_2_and_4))))

targets_case_1_and_3 = convert(Array{Float64,1}, targets_case_1_and_3)
targets_case_2_and_4 = convert(Array{Float64,1}, targets_case_2_and_4)



################################################################################
##   set case model                                                          ###
################################################################################

select_case_model = ARGS[2] # logisticregression or dt

# fit model, i.e. est probabilities
nbr_GP_star_led_GP_old = n-nbr_GP_star_geq_GP_old

prob_case_1 = nbr_case_1/nbr_GP_star_geq_GP_old
prob_case_2 = (nbr_GP_star_led_GP_old-nbr_case_4)/nbr_GP_star_led_GP_old
prob_case_3 = 1-prob_case_1
prob_case_4 = nbr_case_4/nbr_GP_star_led_GP_old
prob_cases = [prob_case_1;prob_case_2;prob_case_3;prob_case_4]

println("Est prob:")
println(prob_cases)

if select_case_model == "biasedcoin"


  casemodel = BiasedCoin(prob_cases)

elseif select_case_model == "logisticregression"


  # tansformed data for logistic model
  mean_posterior = mean(theta_training,2)[:]

  input_data_case_1_and_3 = zeros(length(targets_case_1_and_3), dim+3)
  input_data_case_1_and_3[:,1] = sqrt((mean_posterior[1] - data_case_1_and_3[1,:]).^2)
  input_data_case_1_and_3[:,2] = sqrt((mean_posterior[2] - data_case_1_and_3[2,:]).^2)
  input_data_case_1_and_3[:,3] = sqrt((mean_posterior[3] - data_case_1_and_3[3,:]).^2)
  input_data_case_1_and_3[:,4] = sqrt((mean_posterior[4] - data_case_1_and_3[4,:]).^2)
  input_data_case_1_and_3[:,5] = sqrt((mean_posterior[5] - data_case_1_and_3[5,:]).^2)
  input_data_case_1_and_3[:,6] = sqrt((mean_posterior[6] - data_case_1_and_3[6,:]).^2)
  input_data_case_1_and_3[:,7] = sqrt((mean_posterior[7] - data_case_1_and_3[7,:]).^2)
  input_data_case_1_and_3[:,8] = sqrt(sum((repmat(mean_posterior', size(data_case_1_and_3,2))'-data_case_1_and_3[1:7,:]).^2,1))
  input_data_case_1_and_3[:,8] = data_case_1_and_3[8,:]
  input_data_case_1_and_3[:,end] = targets_case_1_and_3

  input_data_case_2_and_4 = zeros(length(targets_case_2_and_4), dim+3)
  input_data_case_2_and_4[:,1] = sqrt((mean_posterior[1] - data_case_2_and_4[1,:]).^2)
  input_data_case_2_and_4[:,2] = sqrt((mean_posterior[2] - data_case_2_and_4[2,:]).^2)
  input_data_case_2_and_4[:,3] = sqrt((mean_posterior[3] - data_case_2_and_4[3,:]).^2)
  input_data_case_2_and_4[:,4] = sqrt((mean_posterior[4] - data_case_2_and_4[4,:]).^2)
  input_data_case_2_and_4[:,5] = sqrt((mean_posterior[5] - data_case_2_and_4[5,:]).^2)
  input_data_case_2_and_4[:,6] = sqrt((mean_posterior[6] - data_case_2_and_4[6,:]).^2)
  input_data_case_2_and_4[:,7] = sqrt((mean_posterior[7] - data_case_2_and_4[7,:]).^2)
  input_data_case_2_and_4[:,8] = sqrt(sum((repmat(mean_posterior', size(data_case_2_and_4,2))'-data_case_2_and_4[1:7,:]).^2,1))
  input_data_case_1_and_3[:,9] = data_case_1_and_3[8,:]
  input_data_case_2_and_4[:,end] = targets_case_2_and_4

  input_data_case_1_and_3 = DataFrame(input_data_case_1_and_3)

  log_reg_model_case_1_and_3 = glm(@formula(x10 ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8  + x9), input_data_case_1_and_3, Binomial(), LogitLink())

  # fit model for cases 2 and 4

  input_data_case_2_and_4 = DataFrame(input_data_case_2_and_4)

  log_reg_model_case_2_and_4 = glm(@formula(x10 ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8  + x9), input_data_case_2_and_4, Binomial(), LogitLink())

  β_for_model1or3 = coef(log_reg_model_case_1_and_3)
  β_for_model2or4 = coef(log_reg_model_case_2_and_4)

  casemodel = LogisticRegression(β_for_model1or3, β_for_model2or4, mean_posterior)

elseif select_case_model == "dt"


  #standardization!(data_case_1_and_3)

  input_data_case_1_and_3 = zeros(length(targets_case_1_and_3), dim+2)

  input_data_case_1_and_3[:,1] =  data_case_1_and_3[1,:]
  input_data_case_1_and_3[:,2] =  data_case_1_and_3[2,:]
  input_data_case_1_and_3[:,3] =  data_case_1_and_3[3,:]
  input_data_case_1_and_3[:,4] =  data_case_1_and_3[4,:]
  input_data_case_1_and_3[:,5] =  data_case_1_and_3[5,:]
  input_data_case_1_and_3[:,6] =  data_case_1_and_3[6,:]
  input_data_case_1_and_3[:,7] =  data_case_1_and_3[7,:]
  input_data_case_1_and_3[:,8] =  data_case_1_and_3[8,:]
  input_data_case_1_and_3[:,end] = targets_case_1_and_3

  #standardization!(data_case_2_and_4)


  input_data_case_2_and_4 = zeros(length(targets_case_2_and_4), dim+2)

  input_data_case_2_and_4[:,1] =  data_case_2_and_4[1,:]
  input_data_case_2_and_4[:,2] =  data_case_2_and_4[2,:]
  input_data_case_2_and_4[:,3] =  data_case_2_and_4[3,:]
  input_data_case_2_and_4[:,4] =  data_case_2_and_4[4,:]'
  input_data_case_2_and_4[:,5] =  data_case_2_and_4[5,:]
  input_data_case_2_and_4[:,6] =  data_case_2_and_4[6,:]
  input_data_case_2_and_4[:,7] =  data_case_2_and_4[7,:]
  input_data_case_2_and_4[:,8] =  data_case_2_and_4[8,:]
  input_data_case_2_and_4[:,end] = targets_case_2_and_4

  # 4 features model
  features_1_and_3 = convert(Array, input_data_case_1_and_3[:, 1:dim+1])


  labels_1_and_3 = convert(Array, input_data_case_1_and_3[:, end])

  labels_1_and_3 = Array{String}(size(features_1_and_3,1))

  for i = 1:length(labels_1_and_3)
    if input_data_case_1_and_3[i,end] == 0
      labels_1_and_3[i] = "case 3"
    else
      labels_1_and_3[i] = "case 1"
    end
  end

  decisiontree1or3 = build_tree(labels_1_and_3, features_1_and_3)

  decisiontree1or3 = prune_tree(decisiontree1or3, 0.9)

  # tree based model for case 2 and 4

  # 4 features model
  features_case_2_and_4 = convert(Array, input_data_case_2_and_4[:, 1:dim+1])

  labels_case_2_and_4 = Array{String}(size(features_case_2_and_4,1))

  for i = 1:length(labels_case_2_and_4)
    if input_data_case_2_and_4[i,end] == 0
      labels_case_2_and_4[i] = "case 4"
    else
      labels_case_2_and_4[i] = "case 2"
    end
  end

  # train full-tree classifier
  decisiontree2or4 = build_tree(labels_case_2_and_4, features_case_2_and_4)

  decisiontree2or4 = prune_tree(decisiontree2or4, 0.9)

  casemodel = DT(decisiontree1or3, decisiontree2or4)

end


if !log_scale_prior
  # run adaptive PMCMC
  res = adagpmcmc(problem_training, problem, gp, casemodel, cov_matrix)

  mcmc_results = Result(res[1].Theta_est, res[1].loglik_est, res[1].accept_vec, res[1].prior_vec)

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
