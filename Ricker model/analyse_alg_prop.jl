# Script for running DA and ADA using the same traninig data

# go to Ricker model folder
try
  cd("Ricker model")
catch
 warn("Already in the Ricker model folder")
end

# load algorithms
include("rickermodel.jl")

# packages for storing data
using JLD
using HDF5

################################################################################
###      set up problem                                                      ###
################################################################################

problem = set_up_gp_problem(ploton = false)

# set adaptive updating
#problem.adaptive_update = AMUpdate(eye(3), 2.4/sqrt(3), 1., 0.7, 50)

#problem.adaptive_update = noAdaptation(2.4/sqrt(3)*eye(3))

# or, use AM gen alg for adaptive updating
#problem.adaptive_update = AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.2, 1., 0.8, 25)
problem.adaptive_update = AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.3, 1., 0.8, 25)

# set algorithm parameters
problem.alg_param.N = 1000 # nbr particels
problem.alg_param.R = 50000 # nbr iterations
problem.alg_param.burn_in = 0 # burn in
problem.alg_param.length_training_data = 2000
problem.alg_param.alg = "MCWM" # we should only! use the MCWM algorithm
problem.alg_param.compare_GP_and_PF = false
problem.alg_param.noisy_est = false
problem.alg_param.pred_method = "sample"
problem.alg_param.print_interval = 10000 # problem.alg_param.R#
problem.alg_param.beta_MH = 0.15 # "local_loglik_approx" # "max_loglik"
problem.alg_param.lasso = false

#problem.data.y = Array(readtable("y.csv"))[:,1]
#problem.data.y = Array(readtable("y_data_set_1.csv"))[:,1]
problem.data.y = Array(readtable("y_data_set_2.csv"))[:,1]

################################################################################
###      generate traning data                                               ###
################################################################################

# set up training problem

#accelerated_da = true

problem_training = set_up_problem(ploton = false)

length_training_data = 2000
length_test_data = 2000
burn_in = 2000

problem_training.alg_param.N = 1000 # nbr particels
problem_training.alg_param.R = length_training_data + length_test_data + burn_in # nbr iterations
problem_training.alg_param.burn_in = burn_in # burn_in
problem_training.data.y = Array(readtable("y_data_set_2.csv"))[:,1] #Array(readtable("y.csv"))[:,1]
problem_training.alg_param.print_interval = 1000

# test starting at true parameters
#problem.model_param.theta_0 = problem.model_param.theta_true

# PMCMC
problem_training.alg_param.alg = "MCWM"

# use AM alg for adaptive updating
#problem.adaptive_update = AMUpdate(eye(3), 2.4/sqrt(3), 1., 0.7, 25)

#problem.adaptive_update = noAdaptation(2.4/sqrt(3)*eye(3))

# or, use AM gen alg for adaptive updating
#problem_training.adaptive_update = AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.2, 1., 0.8, 25)
problem_training.adaptive_update = AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.4, 1., 0.8, 25)

load_training_data = true

if !load_training_data

  # generate training data
  tic()
  # collect data
  res_training, Theta_star_training, loglik_star_training,Theta_old_training,loglik_old_training, cov_matrix = mcmc(problem_training, true, true)

  time_pre_er = toc()

  # write outputs
  res = res_training[1]

  Theta = res.Theta_est
  loglik = res.loglik_est
  accept_vec = res.accept_vec
  prior_vec = res.prior_vec

  loglik_avec_priorvec = zeros(3, length(loglik))
  loglik_avec_priorvec[1,:] = loglik
  loglik_avec_priorvec[2,:] = accept_vec
  loglik_avec_priorvec[3,:] = prior_vec

  algorithm_parameters = zeros(10, 2)

  algorithm_parameters[1,1] = problem_training.alg_param.burn_in
  algorithm_parameters[2:4,1] = problem_training.model_param.theta_true
  algorithm_parameters[5:7,1] = problem_training.model_param.theta_0
  algorithm_parameters[8:end,:] = problem_training.prior_dist.prior_parameters

  writetable("Results/Theta_training.csv", convert(DataFrame, Theta))
  writetable("Results/loglik_avec_priorvec_training.csv", convert(DataFrame, loglik_avec_priorvec))
  writetable("Results/algorithm_parameters_training.csv", convert(DataFrame, algorithm_parameters))

  # split tranining and test data

  Theta_test_star = Theta_star_training[:,(end-length_test_data+1):end]
  loglik_test_star = loglik_star_training[(end-length_test_data+1):end]

  Theta_test_old = Theta_old_training[:,(end-length_test_data+1):end]
  loglik_test_old = loglik_old_training[(end-length_test_data+1):end]

  data_test_star = [Theta_test_star; loglik_test_star']
  data_test_old = [Theta_test_old; loglik_test_old']

  Theta_training_star = Theta_star_training[:,1:length_training_data]
  loglik_training_star = loglik_star_training[1:length_training_data]

  Theta_training_old = Theta_old_training[:,1:length_training_data]
  loglik_training_old = loglik_old_training[1:length_training_data]

  data_training_star = [Theta_training_star; loglik_training_star']
  data_training_old = [Theta_training_old; loglik_training_old']

  save("gp_training_and_test_data.jld",
        "res_training", res_training,
        "data_training_star", data_training_star,
        "data_training_old", data_training_old,
        "data_test_star", data_test_star,
        "data_test_old", data_test_old,
        "cov_matrix",cov_matrix)

else

  @load "gp_training_and_test_data.jld"

end


################################################################################
###     fit gp model                                                         ###
################################################################################


# create gp object
gp = GPModel("est_method",zeros(6), zeros(4),
eye(problem.alg_param.length_training_data-20), zeros(problem.alg_param.length_training_data-20),zeros(2,problem.alg_param.length_training_data-20),
collect(1:10))

data_training = data_training_star
data_test = data_test_star

tic()


# fit GP model
if problem.alg_param.est_method == "ml"
  # fit GP model using ml

  perc_outlier = 0.1 # used when using PMCMC for trainig data 0.05
  tail_rm = "left"
  problem.alg_param.lasso = false

  ml_est(gp, data_training,"SE", problem.alg_param.lasso,perc_outlier,tail_rm)
else
  error("The two stage estimation method is not in use")
  #two_stage_est(gp, data_training)
end

time_fit_gp = toc()

# save fitted gp model
# @save "gp_fitted_model.jld" gp
nbr_alg_iter = 100

################################################################################
###     DA-GP-MCMC                                                           ###
################################################################################

accelerated_da = false

problem.model_param.theta_0 = mean(res_training[1].Theta_est[:,problem_training.alg_param.burn_in+1:problem_training.alg_param.burn_in+length_training_data],2)

# time multiple runs
alg_prop_da = zeros(nbr_alg_iter,5)

for i = 1:nbr_alg_iter
  println("Iterstion:")
  println(i)
  run_times_da = @elapsed res,run_info = dagpmcmc(problem_training, problem, gp, cov_matrix, true)
  alg_prop_da[i,1] = run_times_da
  alg_prop_da[i,2:end] = run_info
end


# calc res
mcmc_results = Result(res[1].Theta_est, res[1].loglik_est, res[1].accept_vec, res[1].prior_vec)

# write output
Theta = mcmc_results.Theta_est
loglik = mcmc_results.loglik_est
accept_vec = mcmc_results.accept_vec
prior_vec = mcmc_results.prior_vec

loglik_avec_priorvec = zeros(3, length(loglik))
loglik_avec_priorvec[1,:] = loglik
loglik_avec_priorvec[2,:] = accept_vec
loglik_avec_priorvec[3,:] = prior_vec

algorithm_parameters = zeros(10, 2)

algorithm_parameters[1,1] = problem.alg_param.burn_in + 1
algorithm_parameters[2:4,1] = problem.model_param.theta_true
algorithm_parameters[5:7,1] = problem.model_param.theta_0
algorithm_parameters[8:end,:] = problem.prior_dist.prior_parameters

if !accelerated_da
  writetable("Results/Theta_dagpmcmc_alg_prop.csv", convert(DataFrame, Theta))
  writetable("Results/loglik_avec_priorvec_dagpmcmc_alg_prop.csv", convert(DataFrame, loglik_avec_priorvec))
  writetable("Results/algorithm_parameters_dagpmcmc_alg_prop.csv", convert(DataFrame, algorithm_parameters))
else
  writetable("Results/Theta_adagpmcmc_alg_prop.csv", convert(DataFrame, Theta))
  writetable("Results/loglik_avec_priorvec_adagpmcmc_alg_prop.csv", convert(DataFrame, loglik_avec_priorvec))
  writetable("Results/algorithm_parameters_adagpmcmc_alg_prop.csv", convert(DataFrame, algorithm_parameters))
end


################################################################################
###     A-DA-GP-MCMC                                                         ###
################################################################################

accelerated_da = true

problem.model_param.theta_0 = mean(res_training[1].Theta_est[:,problem_training.alg_param.burn_in+1:problem_training.alg_param.burn_in+length_training_data],2)

################################################################################
##  Create features for classification models                                                                            ##
################################################################################

n = size(data_training,2)
n_burn_in = problem_training.alg_param.burn_in


dim = length(problem.model_param.theta_true)
data_gp_loglik_star_old = zeros(4,n)
data_gp_loglik_star_old[3,:] = data_training_star[dim+1,:]
data_gp_loglik_star_old[4,:] = data_training_old[dim+1,:]

# data_gp_loglik_star_old = [gp_star, gp_old, ll_star, ll_old]

std_pred_gp_star = zeros(n)

noisy_pred = problem.alg_param.noisy_est

for i = 1:n
  (loglik_est_star, var_pred_ml_star, prediction_sample_ml_star) = predict(data_training_star[1:dim,i],gp,noisy_pred)
  (loglik_est_old, var_pred_ml, prediction_sample_ml_old) = predict(data_training_old[1:dim,i],gp,noisy_pred)
  data_gp_loglik_star_old[1,i] = prediction_sample_ml_star[1]
  data_gp_loglik_star_old[2,i] = prediction_sample_ml_old[1]
  std_pred_gp_star[i] = sqrt(var_pred_ml_star[1])
end

nbr_GP_star_geq_GP_old = zero(Int64)
nbr_case_1 = zero(Int64)
nbr_case_2 = zero(Int64)

targets_case_1_and_3 = []
data_case_1_and_3 = []

targets_case_2_and_4 = []
data_case_2_and_4 = []

for i = 1:n
  if data_gp_loglik_star_old[1,i] > data_gp_loglik_star_old[2,i]
    nbr_GP_star_geq_GP_old += 1
    data_case_1_and_3 = vcat(data_case_1_and_3, [data_training[1:dim,i]; data_gp_loglik_star_old[1,i]/data_gp_loglik_star_old[2,i]; std_pred_gp_star[i]])
    if data_gp_loglik_star_old[3,i] > data_gp_loglik_star_old[4,i]
      append!(targets_case_1_and_3, 1)
      nbr_case_1 += 1
    else
      append!(targets_case_1_and_3, 0)
    end
  else
    data_case_2_and_4 = vcat(data_case_2_and_4, [data_training[1:dim,i]; data_gp_loglik_star_old[1,i]/data_gp_loglik_star_old[2,i]; std_pred_gp_star[i]])
    if data_gp_loglik_star_old[3,i] < data_gp_loglik_star_old[4,i]
      append!(targets_case_2_and_4, 1)
      nbr_case_2 += 1
    else
      append!(targets_case_2_and_4, 0)
    end
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

if ARGS[1] == "dt"
	select_case_model = "dt" # logisticregression or dt
elseif ARGS[1] == "biasedcoin"	
	select_case_model = "biasedcoin" # logisticregression or dt
end 


nbr_GP_star_led_GP_old = n-nbr_GP_star_geq_GP_old

prob_case_1 = nbr_case_1/nbr_GP_star_geq_GP_old
prob_case_2 = (nbr_case_2)/nbr_GP_star_led_GP_old
prob_case_3 = 1-prob_case_1
prob_case_4 = (nbr_GP_star_led_GP_old-nbr_case_2)/nbr_GP_star_led_GP_old
prob_cases = [prob_case_1;prob_case_2;prob_case_3;prob_case_4]

println("Est prob:")
println(prob_cases)
#prob_cases = [0.2;0.2;0.8;0.8]

if select_case_model == "biasedcoin"

  casemodel = BiasedCoin(prob_cases)

elseif select_case_model == "logisticregression"

  mean_posterior = mean(theta_training,2)[:]

  input_data_case_1_and_3 = zeros(length(targets_case_1_and_3), dim+3)
  input_data_case_1_and_3[:,1] = sqrt((mean_posterior[1] - data_case_1_and_3[1,:]).^2)
  input_data_case_1_and_3[:,2] = sqrt((mean_posterior[2] - data_case_1_and_3[2,:]).^2)
  input_data_case_1_and_3[:,3] = sqrt((mean_posterior[3] - data_case_1_and_3[3,:]).^2)
  input_data_case_1_and_3[:,4] = sqrt(sum((repmat(mean_posterior', size(data_case_1_and_3,2))'-data_case_1_and_3[1:3,:]).^2,1))
  input_data_case_1_and_3[:,5] = data_case_1_and_3[4,:]
  input_data_case_1_and_3[:,end] = targets_case_1_and_3

  input_data_case_1_and_3 = DataFrame(input_data_case_1_and_3)


  input_data_case_2_and_4 = zeros(length(targets_case_2_and_4), dim+3)
  input_data_case_2_and_4[:,1] = sqrt((mean_posterior[1] - data_case_2_and_4[1,:]).^2)
  input_data_case_2_and_4[:,2] = sqrt((mean_posterior[2] - data_case_2_and_4[2,:]).^2)
  input_data_case_2_and_4[:,3] = sqrt((mean_posterior[3] - data_case_2_and_4[3,:]).^2)
  input_data_case_2_and_4[:,4] = sqrt(sum((repmat(mean_posterior', size(data_case_2_and_4,2))'-data_case_2_and_4[1:3,:]).^2,1))
  input_data_case_2_and_4[:,5] = data_case_2_and_4[4,:]
  input_data_case_2_and_4[:,end] = targets_case_2_and_4

  input_data_case_2_and_4 = DataFrame(input_data_case_2_and_4)


  log_reg_model_case_1_and_3 = glm(@formula(x6 ~ x1 + x2 + x3 + x4 + x5), input_data_case_1_and_3, Binomial(), LogitLink())

  log_reg_model_case_2_and_4 = glm(@formula(x6 ~ x1 + x2 + x3 + x4 + x5), input_data_case_2_and_4, Binomial(), LogitLink())

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
  input_data_case_1_and_3[:,end] = targets_case_1_and_3

  #standardization!(data_case_2_and_4)


  input_data_case_2_and_4 = zeros(length(targets_case_2_and_4), dim+2)

  input_data_case_2_and_4[:,1] =  data_case_2_and_4[1,:]
  input_data_case_2_and_4[:,2] =  data_case_2_and_4[2,:]
  input_data_case_2_and_4[:,3] =  data_case_2_and_4[3,:]
  input_data_case_2_and_4[:,4] =  data_case_2_and_4[4,:]
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


alg_prop_ada = zeros(nbr_alg_iter,13)

for i = 1:nbr_alg_iter
  println("Iterstion:")
  println(i)
  run_times_ada = @elapsed res,run_info = adagpmcmc(problem_training, problem, gp, casemodel, cov_matrix, true)
  alg_prop_ada[i,1] = run_times_ada
  alg_prop_ada[i,2:end] = run_info
end



# write results
mcmc_results = Result(res[1].Theta_est, res[1].loglik_est, res[1].accept_vec, res[1].prior_vec)

# write output
Theta = mcmc_results.Theta_est
loglik = mcmc_results.loglik_est

accept_vec = mcmc_results.accept_vec
prior_vec = mcmc_results.prior_vec

loglik_avec_priorvec = zeros(3, length(loglik))
loglik_avec_priorvec[1,:] = loglik
loglik_avec_priorvec[2,:] = accept_vec
loglik_avec_priorvec[3,:] = prior_vec

algorithm_parameters = zeros(10, 2)

algorithm_parameters[1,1] = problem.alg_param.burn_in + 1
algorithm_parameters[2:4,1] = problem.model_param.theta_true
algorithm_parameters[5:7,1] = problem.model_param.theta_0
algorithm_parameters[8:end,:] = problem.prior_dist.prior_parameters

if !accelerated_da
  writetable("Results/Theta_dagpmcmc_alg_prop.csv", convert(DataFrame, Theta))
  writetable("Results/loglik_avec_priorvec_dagpmcmc_alg_prop.csv", convert(DataFrame, loglik_avec_priorvec))
  writetable("Results/algorithm_parameters_dagpmcmc_alg_prop.csv", convert(DataFrame, algorithm_parameters))
else
  writetable("Results/Theta_adagpmcmc_alg_prop.csv", convert(DataFrame, Theta))
  writetable("Results/loglik_avec_priorvec_adagpmcmc_alg_prop.csv", convert(DataFrame, loglik_avec_priorvec))
  writetable("Results/algorithm_parameters_adagpmcmc_alg_prop.csv", convert(DataFrame, algorithm_parameters))
end


if ARGS[1] == "dt"
	writetable("Results/alg_prop_da_dt.csv", convert(DataFrame, alg_prop_da))
	writetable("Results/alg_prop_ada_dt.csv", convert(DataFrame, alg_prop_ada))
elseif ARGS[1] == "biasedcoin"	
	writetable("Results/alg_prop_da_bc.csv", convert(DataFrame, alg_prop_da))
	writetable("Results/alg_prop_ada_bc.csv", convert(DataFrame, alg_prop_ada))
end 


