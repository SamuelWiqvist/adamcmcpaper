# Script for running DA and ADA using the same traninig data

# go to Ricker model folder
try
  cd("Ricker model")
catch
 warn("Already in the Ricker model folder")
end

# load case models
cd("..")
include(pwd()*"\\select case\\selectcase.jl")
cd("Ricker model")

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
problem.alg_param.R = 25000 # nbr iterations
problem.alg_param.burn_in = 0 # burn in
problem.alg_param.length_training_data = 2000
problem.alg_param.alg = "MCWM" # we should only! use the MCWM algorithm
problem.alg_param.compare_GP_and_PF = false
problem.alg_param.noisy_est = false
problem.alg_param.pred_method = "sample"
problem.alg_param.print_interval = 1000 # problem.alg_param.R#
problem.alg_param.beta_MH = 0.1 # "local_loglik_approx" # "max_loglik"
problem.alg_param.lasso = false

#problem.data.y = Array(readtable("y.csv"))[:,1]
#problem.data.y = Array(readtable("y_data_set_1.csv"))[:,1]
problem.data.y = Array(readtable("y_data_set_2.csv"))[:,1]

################################################################################
###      generate traning data                                               ###
################################################################################

# set up training problem

#accelerated_da = true

problem_traning = set_up_problem(ploton = false)

length_training_data = 2000
length_test_data = 2000
burn_in = 2000

problem_traning.alg_param.N = 1000 # nbr particels
problem_traning.alg_param.R = length_training_data + length_test_data + burn_in # nbr iterations
problem_traning.alg_param.burn_in = burn_in # burn_in
problem_traning.data.y = Array(readtable("y_data_set_2.csv"))[:,1] #Array(readtable("y.csv"))[:,1]
problem_traning.alg_param.print_interval = 1000

# test starting at true parameters
#problem.model_param.theta_0 = problem.model_param.theta_true

# PMCMC
problem_traning.alg_param.alg = "MCWM"

# use AM alg for adaptive updating
#problem.adaptive_update = AMUpdate(eye(3), 2.4/sqrt(3), 1., 0.7, 25)

#problem.adaptive_update = noAdaptation(2.4/sqrt(3)*eye(3))

# or, use AM gen alg for adaptive updating
#problem_traning.adaptive_update = AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.2, 1., 0.8, 25)
problem_traning.adaptive_update = AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.4, 1., 0.8, 25)

load_training_data = true

if !load_training_data

  # generate training data
  tic()
  # collect data
  res_training, theta_training, loglik_training, cov_matrix = mcmc(problem_traning, true, true)

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

  algorithm_parameters[1,1] = problem_traning.alg_param.burn_in
  algorithm_parameters[2:4,1] = problem_traning.model_param.theta_true
  algorithm_parameters[5:7,1] = problem_traning.model_param.theta_0
  algorithm_parameters[8:end,:] = problem_traning.prior_dist.Theta_parameters

  writetable("Results/Theta_training.csv", convert(DataFrame, Theta))
  writetable("Results/loglik_avec_priorvec_training.csv", convert(DataFrame, loglik_avec_priorvec))
  writetable("Results/algorithm_parameters_training.csv", convert(DataFrame, algorithm_parameters))

  # split tranining and test data

  theta_test = theta_training[:,(end-length_test_data+1):end]
  loglik_test = loglik_training[(end-length_test_data+1):end]

  data_test = [theta_test; loglik_test']

  theta_training = theta_training[:,1:length_training_data]
  loglik_training = loglik_training[1:length_training_data]

  data_training = [theta_training; loglik_training']

  save("gp_training_and_test_data_ricker_gen_local.jld", "res_training", res_training, "theta_training", theta_training, "loglik_training", loglik_training, "theta_test", theta_test, "loglik_test", loglik_test,"cov_matrix",cov_matrix)

else

  @load "gp_training_and_test_data_ricker_gen_local.jld"

end


################################################################################
###     fit gp model                                                         ###
################################################################################


# fit gp model
tic()

# create gp object
gp = GPModel("est_method",zeros(6), zeros(4),
eye(problem.alg_param.length_training_data-20), zeros(problem.alg_param.length_training_data-20),zeros(2,problem.alg_param.length_training_data-20),
collect(1:10))


data_training = [theta_training; loglik_training']
data_test = [theta_test; loglik_test']

#data_test = data_training[:, Int(size(data_training)[2]/2+1):end]
#data_training = data_training[:, 1:Int(size(data_training)[2]/2)]
data_test = data_training
data_training = data_training

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

################################################################################
###     DA-GP-MCMC                                                           ###
################################################################################

accelerated_da = false

problem.model_param.theta_0 = mean(res_training[1].Theta_est[:,end-size(data_training,2)-size(data_test,2):end],2)

res, res_traning, theta_training, loglik_training, assumption_list, loglik_list = dagpmcmc(problem_traning, problem, gp, cov_matrix)

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
  writetable("Results/Theta_dagpmcmc.csv", convert(DataFrame, Theta))
  writetable("Results/loglik_avec_priorvec_dagpmcmc.csv", convert(DataFrame, loglik_avec_priorvec))
  writetable("Results/algorithm_parameters_dagpmcmc.csv", convert(DataFrame, algorithm_parameters))
else
  writetable("Results/Theta_adagpmcmc.csv", convert(DataFrame, Theta))
  writetable("Results/loglik_avec_priorvec_adagpmcmc.csv", convert(DataFrame, loglik_avec_priorvec))
  writetable("Results/algorithm_parameters_adagpmcmc.csv", convert(DataFrame, algorithm_parameters))
end


################################################################################
###     A-DA-GP-MCMC                                                         ###
################################################################################

accelerated_da = true

problem.model_param.theta_0 = mean(res_training[1].Theta_est[:,end-size(data_training,2)-size(data_test,2):end],2)

################################################################################
##  Create features for classification models                                                                            ##
################################################################################

n = size(data_training,2)
n_burn_in = problem_traning.alg_param.burn_in

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

select_case_model = "biasedcoin" # logisticregression or dt

if select_case_model == "biasedcoin"

  # fit model, i.e. est probabilities
  nbr_GP_star_led_GP_old = n-nbr_GP_star_geq_GP_old

  prob_case_1 = nbr_case_1/nbr_GP_star_geq_GP_old
  prob_case_2 = (nbr_GP_star_led_GP_old-nbr_case_4)/nbr_GP_star_led_GP_old
  prob_case_3 = 1-prob_case_1
  prob_case_4 = nbr_case_4/nbr_GP_star_led_GP_old
  prob_cases = [prob_case_1;prob_case_2;prob_case_3;prob_case_4]

  casemodel = BiaseCoin(prob_cases)

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

# run ADA
res, res_traning, theta_training, loglik_training, assumption_list, loglik_list = adagpmcmc(problem_traning, problem, gp, casemodel, cov_matrix)

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
  writetable("Results/Theta_dagpmcmc.csv", convert(DataFrame, Theta))
  writetable("Results/loglik_avec_priorvec_dagpmcmc.csv", convert(DataFrame, loglik_avec_priorvec))
  writetable("Results/algorithm_parameters_dagpmcmc.csv", convert(DataFrame, algorithm_parameters))
else
  writetable("Results/Theta_adagpmcmc.csv", convert(DataFrame, Theta))
  writetable("Results/loglik_avec_priorvec_adagpmcmc.csv", convert(DataFrame, loglik_avec_priorvec))
  writetable("Results/algorithm_parameters_adagpmcmc.csv", convert(DataFrame, algorithm_parameters))
end


# run analysis
#=
# profiling
using ProfileView
Profile.clear()
res, res_traning, theta_training, loglik_training, assumption_list, loglik_list = @profile dagpMCMC(problem_traning, problem, gp, cov_matrix,accelerated_da)
ProfileView.view()



# analyse results
loglik_list_m = zeros(size(loglik_list,1),5)

for i = 1:size(loglik_list_m,1)
  loglik_list_m[i,:] = loglik_list[i][:]
end

idx_same = find(x->x==true, assumption_list)
idx_diff = find(x->x==false, assumption_list)

length(find(x->x==true, assumption_list))/length(assumption_list)

loglik_list_m_same = loglik_list_m[idx_same,:]
loglik_list_m_diff = loglik_list_m[idx_diff,:]

std(loglik_list_m_same,1)
std(loglik_list_m_diff,1)

colwise(summarystats, convert(DataFrame, loglik_list_m_same))
colwise(summarystats, convert(DataFrame, loglik_list_m_diff))

for i = 1:size(loglik_list_m_same,2)
  PyPlot.figure()
  h = PyPlot.plt[:hist](loglik_list_m_same[:,i],50)
  PyPlot.figure()
  h = PyPlot.plt[:hist](loglik_list_m_diff[:,i],50)
end


for i = 1:size(loglik_list_m_same,2)
  PyPlot.figure()
  PyPlot.plot(loglik_list_m_same[:,i], "*")
  PyPlot.figure()
  PyPlot.plot(loglik_list_m_diff[:,i],"*")
end


PyPlot.figure()
PyPlot.hold(true)
PyPlot.plot(loglik_list_m_same[:,1],loglik_list_m_same[:,end] , "g*")
PyPlot.plot(loglik_list_m_diff[:,1],loglik_list_m_diff[:,end],"r*")

PyPlot.figure()
PyPlot.hold(true)
PyPlot.plot(loglik_list_m_same[:,3], "g*")
PyPlot.plot(loglik_list_m_diff[:,3],"r*")




dist_ll_gp_new_ll_gp_old = loglik_list_m[:,3]-loglik_list_m[:,4]
dist_ll_pf_new_ll_pf_old = loglik_list_m[:,1]-loglik_list_m[:,2]

summarystats(dist_ll_gp_new_ll_gp_old[idx_same])
summarystats(dist_ll_gp_new_ll_gp_old[idx_diff])

PyPlot.figure()
PyPlot.plot(dist_ll_gp_new_ll_gp_old[idx_same], "*")
PyPlot.figure()
PyPlot.plot(dist_ll_gp_new_ll_gp_old[idx_diff],"*")

PyPlot.figure()
h = PyPlot.plt[:hist](dist_ll_gp_new_ll_gp_old[idx_same],50)
PyPlot.figure()
h = PyPlot.plt[:hist](dist_ll_gp_new_ll_gp_old[idx_diff],50)

dist_ll_pf_new_ll_pf_old = loglik_list_m[:,3]-loglik_list_m[:,4]

summarystats(dist_ll_pf_new_ll_pf_old[idx_same])
summarystats(dist_ll_pf_new_ll_pf_old[idx_diff])

PyPlot.figure()
PyPlot.plot(dist_ll_pf_new_ll_pf_old[idx_same], "*")
PyPlot.figure()
PyPlot.plot(dist_ll_pf_new_ll_pf_old[idx_diff],"*")

PyPlot.figure()
h = PyPlot.plt[:hist](dist_ll_pf_new_ll_pf_old[idx_same],50)
PyPlot.figure()
h = PyPlot.plt[:hist](dist_ll_pf_new_ll_pf_old[idx_diff],50)

=#
