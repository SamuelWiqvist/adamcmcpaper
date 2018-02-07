# Script for testing the fit of the GP model

include("rickermodel.jl")

using JLD
using HDF5

# load functions
if Sys.CPU_CORES == 8
  include("C:\\Users\\samuel\\Dropbox\\Phd Education\\Projects\\project 1 accelerated DA and DWP SDE\\code\\utilities\\normplot.jl")
else
  include("C:\\Users\\samuel\\Dropbox\\Phd Education\\Projects\\project 1 accelerated DA and DWP SDE\\code\\utilities\\normplot.jl")
end

################################################################################
##     set up  problem                                                        ##
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
problem.alg_param.length_training_data = 5000
problem.alg_param.alg = "MCWM" # we should only! use the MCWM algorithm
problem.alg_param.compare_GP_and_PF = false
problem.alg_param.noisy_est = false
problem.alg_param.pred_method = "sample"
problem.alg_param.nbr_predictions = 1
problem.alg_param.print_interval = 10000 # problem.alg_param.R#
problem.alg_param.selection_method = "max_loglik"  # "local_loglik_approx" # "max_loglik"
problem.alg_param.beta_MH = 0.1 # "local_loglik_approx" # "max_loglik"
problem.alg_param.std_limit = 1

#problem.data.y = Array(readtable("y.csv"))[:,1]
#problem.data.y = Array(readtable("y_data_set_1.csv"))[:,1]
problem.data.y = Array(readtable("y_data_set_2.csv"))[:,1]

################################################################################
##                         training data                                      ##
################################################################################

load_tranining_data = false

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


################################################################################
##                generate training data                                     ###
################################################################################

if !load_tranining_data

    tic()
    res_training, theta_training, loglik_training, cov_matrix = MCMC(problem_traning, true, true)
    time_pre_er = toc()
    #export_parameters(res_problem_normal_prior_est_AM_gen[2],jobname)


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

    save("gp_training_and_test_data_ricker_test.jld", "res_training", res_training, "theta_training", theta_training, "loglik_training", loglik_training, "theta_test", theta_test, "loglik_test", loglik_test,"cov_matrix",cov_matrix)


else

  #@load "gp_training_$(set_nbr_params)_par.jld"
  #@load "gp_training_$(set_nbr_params)_par.jld"

  @load "gp_training_and_test_data_ricker.jld"

end


################################################################################
###            plot training data                                            ###
################################################################################

#export_data(problem_training, res_training[1],"dagpMCMC_training_data"*jobname)


using PyPlot



text_size = 15

PyPlot.figure(figsize=(10,20))

ax1 = PyPlot.subplot(311)
PyPlot.plot(res_training[1].Theta_est[1,:])
PyPlot.plot(problem.model_param.theta_true[1]*ones(size(res_training[1].Theta_est)[2]), "k")
PyPlot.ylabel(L"$\log r$",fontsize=text_size)
ax1[:axes][:get_xaxis]()[:set_ticks]([])

ax2 = PyPlot.subplot(312)
PyPlot.plot(res_training[1].Theta_est[2,:])
PyPlot.plot(problem.model_param.theta_true[2]*ones(size(res_training[1].Theta_est)[2]), "k")
PyPlot.ylabel(L"$\log \phi$",fontsize=text_size)
ax2[:axes][:get_xaxis]()[:set_ticks]([])

PyPlot.subplot(313)
PyPlot.plot(res_training[1].Theta_est[3,:])
PyPlot.plot(problem.model_param.theta_true[3]*ones(size(res_training[1].Theta_est)[2]), "k")
PyPlot.ylabel(L"$\log \sigma$",fontsize=text_size)
PyPlot.xlabel("Iteration",fontsize=text_size)



for i = 1:3
  PyPlot.figure()
  PyPlot.plot(res_training[1].Theta_est[i,:])
  PyPlot.plot(problem.model_param.theta_true[i]*ones(size(res_training[1].Theta_est)[2]), "k")
end

for i = 1:3
  PyPlot.figure()
  PyPlot.plot(theta_training[i,:])
  PyPlot.plot(problem.model_param.theta_true[i]*ones(size(theta_training,2)), "k")
end

for i = 1:3
  #PyPlot.figure()
  #PyPlot.plt[:hist](theta_training[i,:],100)
  PyPlot.figure()
  h = PyPlot.plt[:hist](theta_training[i,:],100)
  PyPlot.plot((problem.model_param.theta_true[i], problem.model_param.theta_true[i]), (0, maximum(h[1])+5), "k");

end

PyPlot.figure()
PyPlot.plt[:hist](loglik_training,100)

PyPlot.figure()
PyPlot.plot(res_training[1].loglik_est)


show(mean(theta_training,2))
show(std(theta_training,2))


################################################################################
###          Fit GP model                                                     ##
################################################################################

# create gp object
gp = GPModel("est_method",zeros(6), zeros(4),
eye(problem.alg_param.length_training_data-20), zeros(problem.alg_param.length_training_data-20),zeros(2,problem.alg_param.length_training_data-20),
collect(1:10))


# fit gp model

tic()

data_training = [theta_training; loglik_training']
data_test = [theta_test; loglik_test']

#data_test = data_training[:, Int(size(data_training)[2]/2+1):end]
#data_training = data_training[:, 1:Int(size(data_training)[2]/2)]
data_test = data_training
data_training = data_training

# fit GP model
if true #problem.alg_param.est_method == "ml"
  # fit GP model using ml
  #perc_outlier = 0.1 # used when using PMCMC for trainig data 0.05
  #tail_rm = "left"

  perc_outlier = 0.05
  tail_rm = "left"
  lasso = true

  ml_est(gp, data_training,"SE", lasso,perc_outlier,tail_rm)
else
  error("The two stage estimation method is not in use")
  #two_stage_est(gp, data_training)
end

time_fit_gp = toc()


################################################################################
##  Compare loglik predictions                                                                           ##
################################################################################

loglik_pf = data_test[end,:]

(loglik_mean,loglik_std,loglik_sample) = predict(data_test[1:end-1,:], gp, problem.alg_param.noisy_est)


# fit plot


# plot param vs loglik values
PyPlot.figure(figsize=(10,5))
PyPlot.plot(loglik_pf, "*b",alpha=0.3)
PyPlot.plot(loglik_sample, "*r",alpha=0.3)
PyPlot.ylabel(L"$\ell$",fontsize=text_size)
PyPlot.xlabel(L"Index",fontsize=text_size)


# plot param vs loglik values

for i = 1:3
  PyPlot.figure()
  PyPlot.plot(data_test[i,:], loglik_pf, "*b",alpha=0.3)
  PyPlot.plot(data_test[i,:], loglik_sample, "*r",alpha=0.3)
end

for i = 1:3
  PyPlot.figure()
  PyPlot.plot(data_test[i,:], loglik_pf, "*b",alpha=0.3)
  PyPlot.plot(data_test[i,:], loglik_mean, "*r",alpha=0.3)
end


# compute RMSE

RMSE_ml_mean = RMSE(loglik_pf, loglik_mean)
RMSE_ml_sample = RMSE(loglik_pf, loglik_sample)



################################################################################
##  Compute residuals                                                                          ##
################################################################################

residuals = loglik_pf - loglik_sample

# plot residuals
PyPlot.figure(figsize=(10,5))
PyPlot.plot(residuals)
PyPlot.ylabel("Residual",fontsize=text_size)
PyPlot.xlabel("Index",fontsize=text_size)

PyPlot.figure(figsize=(7,5))
PyPlot.plot(data_test[1,:], residuals, "*b")
PyPlot.ylabel("Residual",fontsize=text_size)
PyPlot.xlabel(L"$\log r$",fontsize=text_size)

PyPlot.figure(figsize=(7,5))
PyPlot.plot(data_test[2,:], residuals, "*b")
PyPlot.ylabel("Residual",fontsize=text_size)
PyPlot.xlabel(L"$\log \phi$",fontsize=text_size)

PyPlot.figure(figsize=(7,5))
PyPlot.plot(data_test[3,:], residuals, "*b")
PyPlot.ylabel("Residual",fontsize=text_size)
PyPlot.xlabel(L"$\log \sigma$",fontsize=text_size)

PyPlot.figure(figsize=(7,5))
PyPlot.plot(data_test[4,:], residuals, "*b")
PyPlot.ylabel("Residual",fontsize=text_size)
PyPlot.xlabel(L"$\ell$",fontsize=text_size)

PyPlot.figure(figsize=(7,5))
h1 = PyPlot.plt[:hist](residuals,100, normed=true)
PyPlot.xlabel("Residual",fontsize=text_size)
PyPlot.ylabel("Freq.",fontsize=text_size)

normplot(residuals)

################################################################################
##  Compare assumption using biased coin model                                                                             ##
################################################################################


# estimate probabilities for assumptions


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

noisy_pred = problem.alg_param.noisy_est

for i = 1:n
  (loglik_est_star, var_pred_ml, prediction_sample_ml_star) = predict(data_training[1:dim,i],gp,noisy_pred)
  (loglik_est_old, var_pred_ml, prediction_sample_ml_old) = predict(res_training[1].Theta_est[:,idx_training_old[i]],gp,noisy_pred)
  data_signs[1,i] = prediction_sample_ml_star[1]
  data_signs[2,i] = prediction_sample_ml_old[1]
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

n = size(data_test,2)

nbr_case_1_selceted = 0
nbr_case_2_selceted = 0
nbr_case_3_selceted = 0
nbr_case_4_selceted = 0

nbr_case_1_correct = 0
nbr_case_2_correct = 0
nbr_case_3_correct = 0
nbr_case_4_correct = 0

include("pf.jl")
N = 1000
y = problem_traning.data.y


case_1_theta_val = []
case_1_assumption_holds = []

case_2_theta_val = []
case_2_assumption_holds = []

case_3_theta_val = []
case_3_assumption_holds = []

case_4_theta_val = []
case_4_assumption_holds = []

for i = 1:n

  (loglik_est_star, var_pred_ml, prediction_sample_ml_star) = predict(data_test[1:dim,i],gp,noisy_pred)
  (loglik_est_old, var_pred_ml, prediction_sample_ml_old) = predict(res_training[1].Theta_est[:,idx_test_old[i]],gp,noisy_pred)

  theta_old = res_training[1].Theta_est[:,idx_test_old[i]]
  loglik_pf_old = pf(y, theta_old,problem_traning.model_param.theta_known,N,false)

  theta_new = data_test[1:dim,i] #res_training[1].Theta_est[:,i+n_burn_in+size(data_training,2)]
  loglik_pf_new = pf(y, theta_new,problem_traning.model_param.theta_known,N,false)

  if prediction_sample_ml_star[1] > prediction_sample_ml_old[1]
    go_to_case_1 = rand() < prob_case_1
    if go_to_case_1

      case_1_theta_val = vcat(case_1_theta_val, theta_new)

      nbr_case_1_selceted = nbr_case_1_selceted+1

      if loglik_pf_new > loglik_pf_old
        nbr_case_1_correct = nbr_case_1_correct+1
        append!(case_1_assumption_holds, 1)

      else
        append!(case_1_assumption_holds, 0)
      end

    else
      nbr_case_3_selceted = nbr_case_3_selceted+1
      case_3_theta_val = vcat(case_3_theta_val, theta_new)


      if loglik_pf_new < loglik_pf_old
        nbr_case_3_correct = nbr_case_3_correct+1
        append!(case_3_assumption_holds, 1)
      else
        append!(case_3_assumption_holds, 0)
      end

    end

  else

    go_to_case_2 = rand() < prob_case_2

    if go_to_case_2

      nbr_case_2_selceted = nbr_case_2_selceted+1
      case_2_theta_val = vcat(case_2_theta_val, theta_new)

      if loglik_pf_new < loglik_pf_old
        nbr_case_2_correct = nbr_case_2_correct+1
        append!(case_2_assumption_holds, 1)
      else
        append!(case_2_assumption_holds, 0)
      end

    else
      nbr_case_4_selceted = nbr_case_4_selceted+1
      case_4_theta_val = vcat(case_4_theta_val, theta_new)

      if loglik_pf_new > loglik_pf_old
        nbr_case_4_correct = nbr_case_4_correct+1
        append!(case_4_assumption_holds, 1)
      else
        append!(case_4_assumption_holds, 0)
      end

    end
  end
end

case_1_theta_val = reshape(case_1_theta_val, (dim, length(case_1_assumption_holds)))
case_2_theta_val = reshape(case_2_theta_val, (dim, length(case_2_assumption_holds)))
case_3_theta_val = reshape(case_3_theta_val, (dim, length(case_3_assumption_holds)))
case_4_theta_val = reshape(case_4_theta_val, (dim, length(case_4_assumption_holds)))

nbr_case_selceted = [nbr_case_1_selceted nbr_case_2_selceted nbr_case_3_selceted nbr_case_4_selceted]

nbr_case_correct = [nbr_case_1_correct nbr_case_2_correct nbr_case_3_correct nbr_case_4_correct]

prob_correct_given_selected = nbr_case_correct./nbr_case_selceted



PyPlot.figure()
PyPlot.plot3D(case_1_theta_val[1,find(x -> x==0, case_1_assumption_holds)],
              case_1_theta_val[2,find(x -> x==0, case_1_assumption_holds)],
              case_1_theta_val[3,find(x -> x==0, case_1_assumption_holds)],
              "*r")
PyPlot.plot3D(case_1_theta_val[1,find(x -> x==1, case_1_assumption_holds)],
              case_1_theta_val[2,find(x -> x==1, case_1_assumption_holds)],
              case_1_theta_val[3,find(x -> x==1, case_1_assumption_holds)],
              "*g")
PyPlot.xlabel(L"\log r")
PyPlot.ylabel(L"\log \phi")
PyPlot.zlabel(L"log \sigma")


PyPlot.figure()
PyPlot.plot3D(case_2_theta_val[1,find(x -> x==0, case_2_assumption_holds)],
              case_2_theta_val[2,find(x -> x==0, case_2_assumption_holds)],
              case_2_theta_val[3,find(x -> x==0, case_2_assumption_holds)],
              "*r")
PyPlot.plot3D(case_2_theta_val[1,find(x -> x==1, case_2_assumption_holds)],
              case_2_theta_val[2,find(x -> x==1, case_2_assumption_holds)],
              case_2_theta_val[3,find(x -> x==1, case_2_assumption_holds)],
              "*g")
PyPlot.xlabel(L"\log r")
PyPlot.ylabel(L"\log \phi")
PyPlot.zlabel(L"log \sigma")



PyPlot.figure()
PyPlot.plot3D(case_3_theta_val[1,find(x -> x==0, case_3_assumption_holds)],
              case_3_theta_val[2,find(x -> x==0, case_3_assumption_holds)],
              case_3_theta_val[3,find(x -> x==0, case_3_assumption_holds)],
              "*r")
PyPlot.plot3D(case_3_theta_val[1,find(x -> x==1, case_3_assumption_holds)],
              case_3_theta_val[2,find(x -> x==1, case_3_assumption_holds)],
              case_3_theta_val[3,find(x -> x==1, case_3_assumption_holds)],
              "*g")
PyPlot.xlabel(L"\log r")
PyPlot.ylabel(L"\log \phi")
PyPlot.zlabel(L"log \sigma")


PyPlot.figure()
PyPlot.plot3D(case_4_theta_val[1,find(x -> x==0, case_4_assumption_holds)],
              case_4_theta_val[2,find(x -> x==0, case_4_assumption_holds)],
              case_4_theta_val[3,find(x -> x==0, case_4_assumption_holds)],
              "*r")
PyPlot.plot3D(case_4_theta_val[1,find(x -> x==1, case_4_assumption_holds)],
              case_4_theta_val[2,find(x -> x==1, case_4_assumption_holds)],
              case_4_theta_val[3,find(x -> x==1, case_4_assumption_holds)],
              "*g")
PyPlot.xlabel(L"\log r")
PyPlot.ylabel(L"\log \phi")
PyPlot.zlabel(L"log \sigma")




################################################################################
##  Compare assumption using logistic regression model                                                                             ##
################################################################################



# fit logistic regression model


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

noisy_pred = problem.alg_param.noisy_est


for i = 1:n
  (loglik_est_star, var_pred_ml, prediction_sample_ml_star) = predict(data_training[1:dim,i],gp,noisy_pred)
  (loglik_est_old, var_pred_ml, prediction_sample_ml_old) = predict(res_training[1].Theta_est[:,idx_training_old[i]],gp,noisy_pred)
  data_signs[1,i] = prediction_sample_ml_star[1]
  data_signs[2,i] = prediction_sample_ml_old[1]
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
    data_case_1_and_3 = vcat(data_case_1_and_3, data_training[1:dim,i])
    if data_signs[3,i] > data_signs[4,i]
      append!(targets_case_1_and_3, 1)
      nbr_case_1 += 1
    else
      append!(targets_case_1_and_3, 0)
    end
  elseif data_signs[3,i] > data_signs[4,i]
    data_case_2_and_4 = vcat(data_case_2_and_4, data_training[1:dim,i])
    append!(targets_case_2_and_4, 0)
    nbr_case_4 += 1
  else
    data_case_2_and_4 = vcat(data_case_2_and_4, data_training[1:dim,i])
    append!(targets_case_2_and_4, 1)
  end
end

data_case_1_and_3 = reshape(data_case_1_and_3, (dim, length(targets_case_1_and_3)))
data_case_2_and_4 = reshape(data_case_2_and_4, (dim, length(targets_case_2_and_4)))

nbr_GP_star_led_GP_old = n-nbr_GP_star_geq_GP_old

prob_case_1 = nbr_case_1/nbr_GP_star_geq_GP_old
prob_case_2 = (nbr_GP_star_led_GP_old-nbr_case_4)/nbr_GP_star_led_GP_old
prob_case_3 = 1-prob_case_1
prob_case_4 = nbr_case_4/nbr_GP_star_led_GP_old

prob_cases = [prob_case_1;prob_case_2;prob_case_3;prob_case_4]


# finding structure for logistic regression model

PyPlot.figure()
PyPlot.plot(data_case_1_and_3[1,:], targets_case_1_and_3, "*")

PyPlot.figure()
PyPlot.plot(data_case_1_and_3[2,:], targets_case_1_and_3, "*")


PyPlot.figure()
PyPlot.plot(data_case_1_and_3[3,:], targets_case_1_and_3, "*")


PyPlot.figure()
PyPlot.plot3D(data_case_1_and_3[1,find(x -> x==0, targets_case_1_and_3)],
              data_case_1_and_3[2,find(x -> x==0, targets_case_1_and_3)],
              data_case_1_and_3[3,find(x -> x==0, targets_case_1_and_3)],
              "*r")
PyPlot.plot3D(data_case_1_and_3[1,find(x -> x==1, targets_case_1_and_3)],
              data_case_1_and_3[2,find(x -> x==1, targets_case_1_and_3)],
              data_case_1_and_3[3,find(x -> x==1, targets_case_1_and_3)],
              "*g")
PyPlot.xlabel(L"\log r")
PyPlot.ylabel(L"\log \phi")
PyPlot.zlabel(L"log \sigma")


PyPlot.figure()
PyPlot.plt[:hist](data_case_1_and_3[1,find(x -> x==1, targets_case_1_and_3)],50, color = "g",alpha=0.3)
PyPlot.plt[:hist](data_case_1_and_3[1,find(x -> x==0, targets_case_1_and_3)],50, color = "r",alpha=0.3)

PyPlot.figure()
PyPlot.plt[:hist](data_case_1_and_3[2,find(x -> x==1, targets_case_1_and_3)],50, color = "g",alpha=0.3)
PyPlot.plt[:hist](data_case_1_and_3[2,find(x -> x==0, targets_case_1_and_3)],50, color = "r",alpha=0.3)

PyPlot.figure()
PyPlot.plt[:hist](data_case_1_and_3[2,find(x -> x==1, targets_case_1_and_3)],50, color = "g",alpha=0.3)
PyPlot.plt[:hist](data_case_1_and_3[2,find(x -> x==0, targets_case_1_and_3)],50, color = "r",alpha=0.3)



PyPlot.figure()
PyPlot.plot(data_case_2_and_4[1,:], targets_case_2_and_4, "*")

PyPlot.figure()
PyPlot.plot(data_case_2_and_4[2,:], targets_case_2_and_4, "*")


PyPlot.figure()
PyPlot.plot(data_case_2_and_4[3,:], targets_case_2_and_4, "*")



PyPlot.figure()
PyPlot.plot3D(data_case_2_and_4[1,find(x -> x==0, targets_case_2_and_4)],
              data_case_2_and_4[2,find(x -> x==0, targets_case_2_and_4)],
              data_case_2_and_4[3,find(x -> x==0, targets_case_2_and_4)],
              "*r")
PyPlot.plot3D(data_case_2_and_4[1,find(x -> x==1, targets_case_2_and_4)],
              data_case_2_and_4[2,find(x -> x==1, targets_case_2_and_4)],
              data_case_2_and_4[3,find(x -> x==1, targets_case_2_and_4)],
              "*g")
PyPlot.xlabel(L"\log r")
PyPlot.ylabel(L"\log \phi")
PyPlot.zlabel(L"log \sigma")


PyPlot.figure()
PyPlot.plt[:hist](data_case_2_and_4[1,find(x -> x==1, targets_case_1_and_3)],50, color = "g",alpha=0.3)
PyPlot.plt[:hist](data_case_2_and_4[1,find(x -> x==0, targets_case_1_and_3)],50, color = "r",alpha=0.3)

PyPlot.figure()
PyPlot.plt[:hist](data_case_2_and_4[2,find(x -> x==1, targets_case_1_and_3)],50, color = "g",alpha=0.3)
PyPlot.plt[:hist](data_case_2_and_4[2,find(x -> x==0, targets_case_1_and_3)],50, color = "r",alpha=0.3)

PyPlot.figure()
PyPlot.plt[:hist](data_case_2_and_4[2,find(x -> x==1, targets_case_1_and_3)],50, color = "g",alpha=0.3)
PyPlot.plt[:hist](data_case_2_and_4[2,find(x -> x==0, targets_case_1_and_3)],50, color = "r",alpha=0.3)




using GLM


mean_posterior = mean(theta_training,2)

input_data_case_1_and_3 = zeros(length(targets_case_1_and_3), dim+2)

input_data_case_1_and_3[:,1] = sqrt((mean_posterior[1] - data_case_1_and_3[1,:]).^2)
input_data_case_1_and_3[:,2] = sqrt((mean_posterior[2] - data_case_1_and_3[2,:]).^2)
input_data_case_1_and_3[:,3] = sqrt((mean_posterior[3] - data_case_1_and_3[3,:]).^2)
input_data_case_1_and_3[:,dim+1] = sqrt(sum((repmat(mean_posterior', size(data_case_1_and_3,2))'-data_case_1_and_3).^2,1))

input_data_case_1_and_3[:,end] = targets_case_1_and_3



PyPlot.figure()
PyPlot.plot3D(input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),1],
              input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),2],
              input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),3],
              "*r")
PyPlot.plot3D(input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),1],
              input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),2],
              input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),3],
              "*g")
PyPlot.xlabel(L"\log r")
PyPlot.ylabel(L"\log \phi")
PyPlot.zlabel(L"log \sigma")


input_data_case_1_and_3 = DataFrame(input_data_case_1_and_3)

log_reg_model_case_1_and_3 = glm(@formula(x5 ~ x1 + x2 + x3 + x4), input_data_case_1_and_3, Binomial(), LogitLink())

# without transform
input_data_case_1_and_3 = zeros(length(targets_case_1_and_3), dim+2)
input_data_case_1_and_3[:,1:dim] = data_case_1_and_3'
input_data_case_1_and_3[:,end] = targets_case_1_and_3
input_data_case_1_and_3 = DataFrame(input_data_case_1_and_3)

log_reg_model_case_1_and_3 = glm(@formula(x4 ~ x1 + x2 + x3), input_data_case_1_and_3, Binomial(), LogitLink())


input_data_case_1_and_3 = DataFrame(input_data_case_1_and_3[:,2:4])


log_reg_model_case_1_and_3_test = glm(@formula(x3 ~ x1 + x2), input_data_case_1_and_3, Binomial(), LogitLink())


# fit model for cases 2 and 4


mean_posterior = mean(theta_training,2)

input_data_case_2_and_4 = zeros(length(targets_case_2_and_4), dim+2)

input_data_case_2_and_4[:,1] = sqrt((mean_posterior[1] - data_case_2_and_4[1,:]).^2)
input_data_case_2_and_4[:,2] = sqrt((mean_posterior[2] - data_case_2_and_4[2,:]).^2)
input_data_case_2_and_4[:,3] = sqrt((mean_posterior[3] - data_case_2_and_4[3,:]).^2)
input_data_case_2_and_4[:,dim+1] = sqrt(sum((repmat(mean_posterior', size(data_case_2_and_4,2))'-data_case_2_and_4).^2,1))

input_data_case_2_and_4[:,end] = targets_case_2_and_4


PyPlot.figure()
PyPlot.plot3D(input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),1],
              input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),2],
              input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),3],
              "*r")
PyPlot.plot3D(input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),1],
              input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),2],
              input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),3],
              "*g")
PyPlot.xlabel(L"\log r")
PyPlot.ylabel(L"\log \phi")
PyPlot.zlabel(L"log \sigma")


input_data_case_2_and_4 = DataFrame(input_data_case_2_and_4)

log_reg_model_case_2_and_4 = glm(@formula(x5 ~ x1 + x2 + x3 + x4), input_data_case_2_and_4, Binomial(), LogitLink())


# without transform
input_data_case_2_and_4 = zeros(length(targets_case_2_and_4), dim+1)
input_data_case_2_and_4[:,1:dim] = data_case_2_and_4'
input_data_case_2_and_4[:,end] = targets_case_2_and_4
input_data_case_2_and_4 = DataFrame(input_data_case_2_and_4)

log_reg_model_case_2_and_4 = glm(@formula(x4 ~ x1 + x2 + x3), input_data_case_2_and_4, Binomial(), LogitLink())



# test on training data

pred_training_data_case_1_and_3 = GLM.predict(log_reg_model_case_1_and_3) #, DataFrame(data_case_1_and_3'))


PyPlot.figure()
PyPlot.plot(data_case_1_and_3[1,:], log(pred_training_data_case_1_and_3./(1-pred_training_data_case_1_and_3)), "*")


PyPlot.figure()
PyPlot.plot(data_case_1_and_3[2,:], log(pred_training_data_case_1_and_3./(1-pred_training_data_case_1_and_3)), "*")


PyPlot.figure()
PyPlot.plot(data_case_1_and_3[3,:], log(pred_training_data_case_1_and_3./(1-pred_training_data_case_1_and_3)), "*")


PyPlot.figure()
PyPlot.plot(data_case_1_and_3[1,:], pred_training_data_case_1_and_3, "*")


PyPlot.figure()
PyPlot.plot(data_case_1_and_3[2,:], pred_training_data_case_1_and_3, "*")


PyPlot.figure()
PyPlot.plot(data_case_1_and_3[3,:], pred_training_data_case_1_and_3, "*")

PyPlot.figure()
PyPlot.plot3D(data_case_1_and_3[1,:],data_case_1_and_3[2,:], pred_training_data_case_1_and_3, "*")


PyPlot.figure()
PyPlot.plot3D(data_case_1_and_3[1,:],data_case_1_and_3[3,:], pred_training_data_case_1_and_3, "*")

PyPlot.figure()
PyPlot.plot3D(data_case_1_and_3[2,:],data_case_1_and_3[3,:], pred_training_data_case_1_and_3, "*")


pred_training_data_case_2_and_4 = GLM.predict(log_reg_model_case_2_and_4)#, #DataFrame(data_case_2_and_4'))




PyPlot.figure()
PyPlot.plot(data_case_2_and_4[1,:], log(pred_training_data_case_2_and_4./(1-pred_training_data_case_2_and_4)), "*")


PyPlot.figure()
PyPlot.plot(data_case_2_and_4[2,:], log(pred_training_data_case_2_and_4./(1-pred_training_data_case_2_and_4)), "*")


PyPlot.figure()
PyPlot.plot(data_case_2_and_4[3,:], log(pred_training_data_case_2_and_4./(1-pred_training_data_case_2_and_4)), "*")



PyPlot.figure()
PyPlot.plot(data_case_2_and_4[1,:], pred_training_data_case_2_and_4, "*")


PyPlot.figure()
PyPlot.plot(data_case_2_and_4[2,:], pred_training_data_case_2_and_4, "*")


PyPlot.figure()
PyPlot.plot(data_case_2_and_4[3,:], pred_training_data_case_2_and_4, "*")


PyPlot.figure()
PyPlot.plot3D(data_case_2_and_4[1,:],data_case_2_and_4[2,:], pred_training_data_case_2_and_4, "*")


PyPlot.figure()
PyPlot.plot3D(data_case_2_and_4[1,:],data_case_2_and_4[3,:], pred_training_data_case_2_and_4, "*")

PyPlot.figure()
PyPlot.plot3D(data_case_2_and_4[2,:],data_case_2_and_4[3,:], pred_training_data_case_2_and_4, "*")



# test model on test data


doc"""
   predict(beta::Vector, theta::Vector)

Prediction for the logistic regression model at theta.
"""
function predict(beta::Vector, theta::Vector)

  p_hat =  exp(theta'*beta)./(1+exp(theta'*beta))
  return p_hat[1]

end

beta_case_1_and_3 = coef(log_reg_model_case_1_and_3)
beta_case_2_and_4 = coef(log_reg_model_case_2_and_4)

n = size(data_test,2)

nbr_case_1_selceted = 0
nbr_case_2_selceted = 0
nbr_case_3_selceted = 0
nbr_case_4_selceted = 0

nbr_case_1_correct = 0
nbr_case_2_correct = 0
nbr_case_3_correct = 0
nbr_case_4_correct = 0

include("pf.jl")
N = 1000
y = problem_traning.data.y

for i = 1:n

  (loglik_est_star, var_pred_ml, prediction_sample_ml_star) = predict(data_test[1:dim,i],gp,noisy_pred)
  (loglik_est_old, var_pred_ml, prediction_sample_ml_old) = predict(res_training[1].Theta_est[:,idx_test_old[i]],gp,noisy_pred)

  theta_old = res_training[1].Theta_est[:,idx_test_old[i]]
  loglik_pf_old = pf(y, theta_old,problem_traning.model_param.theta_known,N,false)

  theta_new = data_test[1:dim,i] #res_training[1].Theta_est[:,i+n_burn_in+size(data_training,2)]
  loglik_pf_new = pf(y, theta_new,problem_traning.model_param.theta_known,N,false)

  # tansformation of theta_new
  # transform theta_new to euclidian distance space
  theta_new_log_reg_mod = zeros(dim+1)

  theta_new_log_reg_mod[1] = sqrt((mean_posterior[1] - theta_new[1]).^2)
  theta_new_log_reg_mod[2] = sqrt((mean_posterior[2] - theta_new[2]).^2)
  theta_new_log_reg_mod[3] = sqrt((mean_posterior[3] - theta_new[3]).^2)
  theta_new_log_reg_mod[4] = sqrt(sum(mean_posterior-theta_new).^2)

  if prediction_sample_ml_star[1] > prediction_sample_ml_old[1]
    #prob_case_1 = predict(beta_case_1_and_3, [1;data_test[1:dim,i]])
    prob_case_1 = predict(beta_case_1_and_3, [1;theta_new_log_reg_mod])

    go_to_case_1 = rand(Bernoulli(prob_case_1)) == 1
    if go_to_case_1

      nbr_case_1_selceted = nbr_case_1_selceted+1


      if loglik_pf_new > loglik_pf_old
        nbr_case_1_correct = nbr_case_1_correct+1
      end

    else
      nbr_case_3_selceted = nbr_case_3_selceted+1

      if loglik_pf_new < loglik_pf_old
        nbr_case_3_correct = nbr_case_3_correct+1
      end

    end

  else

    #prob_case_2 = predict(beta_case_2_and_4, [1;data_test[1:dim,i]])
    prob_case_2 = predict(beta_case_2_and_4, [1;theta_new_log_reg_mod])
    go_to_case_2 = rand(Bernoulli(prob_case_2)) == 1

    if go_to_case_2

      nbr_case_2_selceted = nbr_case_2_selceted+1

      if loglik_pf_new < loglik_pf_old
        nbr_case_2_correct = nbr_case_2_correct+1
      end

    else
      nbr_case_4_selceted = nbr_case_4_selceted+1

      if loglik_pf_new > loglik_pf_old
        nbr_case_4_correct = nbr_case_4_correct+1
      end

    end
  end
end

nbr_case_selceted = [nbr_case_1_selceted nbr_case_2_selceted nbr_case_3_selceted nbr_case_4_selceted]

nbr_case_correct = [nbr_case_1_correct nbr_case_2_correct nbr_case_3_correct nbr_case_4_correct]

prob_correct_given_selected = nbr_case_correct./nbr_case_selceted