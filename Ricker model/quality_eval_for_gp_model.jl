# Script for evaluating the GP model

using PyPlot

include(pwd()*"/utilities/featurescaling.jl")
include(pwd()*"/utilities/normplot.jl")
include(pwd()*"/utilities/posteriorinference.jl")

try
  cd("Ricker model")
catch
 warn("Already in the Ricker model folder")
end

include("rickermodel.jl")

try
  cd("..")
catch
 warn("Already in main folder")
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
problem.alg_param.print_interval = 10000 # problem.alg_param.R#
problem.alg_param.beta_MH = 0.15 # "local_loglik_approx" # "max_loglik"

#problem.data.y = Array(readtable("y.csv"))[:,1]
#problem.data.y = Array(readtable("y_data_set_1.csv"))[:,1]
problem.data.y = Array(readtable("Ricker model/y_data_set_2.csv"))[:,1]

################################################################################
##                         training data                                      ##
################################################################################

problem_training = set_up_problem(ploton = false)

length_training_data = 2000
length_test_data = 2000
burn_in = 2000

problem_training.alg_param.N = 1000 # nbr particels
problem_training.alg_param.R = length_training_data + length_test_data + burn_in # nbr iterations
problem_training.alg_param.burn_in = burn_in # burn_in
problem_training.data.y = Array(readtable("Ricker model/y_data_set_2.csv"))[:,1] #Array(readtable("y.csv"))[:,1]
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

################################################################################
##                generate training data                                     ###
################################################################################

if !load_training_data

    # we can only load the training and test data

else

  #@load "gp_training_$(set_nbr_params)_par.jld"
  #@load "gp_training_$(set_nbr_params)_par.jld"

  @load "Ricker model/gp_training_and_test_data.jld"

end


################################################################################
###            plot training data                                            ###
################################################################################

#export_data(problem_training, res_training[1],"dagpMCMC_training_data"*jobname)

# plot chains for tranining
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

# plot chains after burn-in for tranining

PyPlot.figure(figsize=(10,20))

ax1 = PyPlot.subplot(311)
PyPlot.plot(res_training[1].Theta_est[1,burn_in:end])
PyPlot.plot(problem.model_param.theta_true[1]*ones(size(res_training[1].Theta_est[:,burn_in:end])[2]), "k")
PyPlot.ylabel(L"$\log r$",fontsize=text_size)
ax1[:axes][:get_xaxis]()[:set_ticks]([])

ax2 = PyPlot.subplot(312)
PyPlot.plot(res_training[1].Theta_est[2,burn_in:end])
PyPlot.plot(problem.model_param.theta_true[2]*ones(size(res_training[1].Theta_est[:,burn_in:end])[2]), "k")
PyPlot.ylabel(L"$\log \phi$",fontsize=text_size)
ax2[:axes][:get_xaxis]()[:set_ticks]([])

PyPlot.subplot(313)
PyPlot.plot(res_training[1].Theta_est[3,burn_in:end])
PyPlot.plot(problem.model_param.theta_true[3]*ones(size(res_training[1].Theta_est[:,burn_in:end])[2]), "k")
PyPlot.ylabel(L"$\log \sigma$",fontsize=text_size)
PyPlot.xlabel("Iteration",fontsize=text_size)


for i = 1:3
  PyPlot.figure()
  PyPlot.plot(res_training[1].Theta_est[i,:])
  PyPlot.plot(problem.model_param.theta_true[i]*ones(size(res_training[1].Theta_est)[2]), "k")
end

for i = 1:3
  PyPlot.figure()
  PyPlot.plot(data_training_star[i,:])
  PyPlot.plot(problem.model_param.theta_true[i]*ones(size(data_training_star,2)), "k")
end

for i = 1:3
  #PyPlot.figure()
  #PyPlot.plt[:hist](theta_training[i,:],100)
  PyPlot.figure()
  h = PyPlot.plt[:hist](data_training_star[i,:],100)
  PyPlot.plot((problem.model_param.theta_true[i], problem.model_param.theta_true[i]), (0, maximum(h[1])+5), "k");
end

PyPlot.figure()
PyPlot.plt[:hist](data_training_star[end,:],100)

PyPlot.figure()
PyPlot.plot(res_training[1].loglik_est)


show(mean(data_training_star[1:3,:],2))
show(std(data_training_star[1:3,:],2))


################################################################################
###          Fit GP model                                                     ##
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
##  Plot marginal functions pf as function of parameter value                                                                          ##
################################################################################


y = problem.data.y
theta_true = problem.model_param.theta_true
theta_known = problem.model_param.theta_known
N = 1000  #problem.alg_param.N
prior_parameters = problem.prior_dist.prior_parameters

include("pf.jl")

N = 1000

PyPlot.figure()
r_vec = prior_parameters[1,1]:0.01:prior_parameters[1,2]

loglik_r_non_fixed = zeros(2,length(r_vec))

for i = 1:length(r_vec)
  loglik_r_non_fixed[1,i] = pf(y, [r_vec[i]; theta_true[2:3]],theta_known,N,false)
  (loglik_mean,loglik_std,loglik_sample) = predict([r_vec[i]; theta_true[2:3]], gp, problem.alg_param.noisy_est)
  loglik_r_non_fixed[2,i]  = loglik_sample[1]
end

PyPlot.figure()
PyPlot.plot(r_vec, loglik_r_non_fixed[1,:], "b")
PyPlot.plot(r_vec, loglik_r_non_fixed[2,:], "r")
PyPlot.plot((theta_true[1], theta_true[1]), (minimum(loglik_r_non_fixed[find(!isnan, loglik_r_non_fixed)]), maximum(loglik_r_non_fixed[find(!isnan, loglik_r_non_fixed)])), "k")
PyPlot.plot((maximum(data_training[1,:]), maximum(data_training[1,:])), (minimum(loglik_r_non_fixed[find(!isnan, loglik_r_non_fixed)]), maximum(loglik_r_non_fixed[find(!isnan, loglik_r_non_fixed)])), "k--")
PyPlot.plot((minimum(data_training[1,:]), minimum(data_training[1,:])), (minimum(loglik_r_non_fixed[find(!isnan, loglik_r_non_fixed)]), maximum(loglik_r_non_fixed[find(!isnan, loglik_r_non_fixed)])), "k--")
PyPlot.xlabel(L"$\log r$")
PyPlot.ylabel(L"$\ell$")

# keeping psi non fixed

psi_vec = prior_parameters[2,1]:0.01:prior_parameters[2,2]

loglik_psi_non_fixed = zeros(2,length(psi_vec))

for i = 1:length(psi_vec)
  loglik_psi_non_fixed[1,i] = pf(y, [theta_true[1];psi_vec[i]; theta_true[3]],theta_known,N,false)
  (loglik_mean,loglik_std,loglik_sample) = predict( [theta_true[1];psi_vec[i]; theta_true[3]], gp, problem.alg_param.noisy_est)
  loglik_psi_non_fixed[2,i]  = loglik_sample[1]
end

PyPlot.figure()
PyPlot.plot(psi_vec, loglik_psi_non_fixed[1,:], "b")
PyPlot.plot(psi_vec, loglik_psi_non_fixed[2,:], "r")
PyPlot.plot((theta_true[2], theta_true[2]), (minimum(loglik_psi_non_fixed[find(!isnan, loglik_psi_non_fixed)]), maximum(loglik_r_non_fixed[find(!isnan, loglik_psi_non_fixed)])), "k")
PyPlot.plot((maximum(data_training[2,:]), maximum(data_training[2,:])), (minimum(loglik_psi_non_fixed[find(!isnan, loglik_psi_non_fixed)]), maximum(loglik_psi_non_fixed[find(!isnan, loglik_psi_non_fixed)])), "k--")
PyPlot.plot((minimum(data_training[2,:]), minimum(data_training[2,:])), (minimum(loglik_psi_non_fixed[find(!isnan, loglik_psi_non_fixed)]), maximum(loglik_psi_non_fixed[find(!isnan, loglik_psi_non_fixed)])), "k--")
PyPlot.xlabel(L"$\log \phi$")
PyPlot.ylabel(L"$\ell$")

# keeping psi non fixed

sigma_vec = prior_parameters[3,1]:0.01:prior_parameters[3,2]

loglik_sigma_non_fixed = zeros(2,length(sigma_vec))

for i = 1:length(sigma_vec)
  loglik_sigma_non_fixed[1,i] = pf(y,[theta_true[1:2];sigma_vec[i]],theta_known,N,false)
  (loglik_mean,loglik_std,loglik_sample) = predict([theta_true[1:2];sigma_vec[i]], gp, problem.alg_param.noisy_est)
  loglik_sigma_non_fixed[2,i]  = loglik_sample[1]
end

PyPlot.figure()
PyPlot.plot(sigma_vec, loglik_sigma_non_fixed[1,:], "b")
PyPlot.plot(sigma_vec, loglik_sigma_non_fixed[2,:], "r")
PyPlot.plot((theta_true[3], theta_true[3]), (minimum(loglik_sigma_non_fixed[find(!isnan, loglik_sigma_non_fixed)]), maximum(loglik_sigma_non_fixed[find(!isnan, loglik_sigma_non_fixed)])), "k")
PyPlot.plot((maximum(data_training[3,:]), maximum(data_training[3,:])), (minimum(loglik_sigma_non_fixed[find(!isnan, loglik_sigma_non_fixed)]), maximum(loglik_sigma_non_fixed[find(!isnan, loglik_sigma_non_fixed)])), "k--")
PyPlot.plot((minimum(data_training[3,:]), minimum(data_training[3,:])), (minimum(loglik_sigma_non_fixed[find(!isnan, loglik_sigma_non_fixed)]), maximum(loglik_sigma_non_fixed[find(!isnan, loglik_sigma_non_fixed)])), "k--")
PyPlot.xlabel(L"$\log \sigma$")
PyPlot.ylabel(L"$\ell$")


################################################################################
##  Selecting case problem                                                    ##
################################################################################

# Create features for classification models                                                                            ##

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



# use untransformed features (biased coin model)

# is is not nessesary to construct input_data_case_1_and_3 and input_data_case_2_and_4 to run the biased coin model

input_data_case_1_and_3 = zeros(length(targets_case_1_and_3), dim+1)

input_data_case_1_and_3[:,1] =  data_case_1_and_3[1,:]
input_data_case_1_and_3[:,2] =  data_case_1_and_3[2,:]
input_data_case_1_and_3[:,3] =  data_case_1_and_3[3,:]
input_data_case_1_and_3[:,end] = targets_case_1_and_3

input_data_case_2_and_4 = zeros(length(targets_case_2_and_4), dim+1)

input_data_case_2_and_4[:,1] =  data_case_2_and_4[1,:]
input_data_case_2_and_4[:,2] =  data_case_2_and_4[2,:]
input_data_case_2_and_4[:,3] =  data_case_2_and_4[3,:]
input_data_case_2_and_4[:,end] = targets_case_2_and_4



# use tranformed data + standardization (logistic regression model)

# tansformed data for simple logistic model
mean_posterior = mean(theta_training,2)

input_data_case_1_and_3 = zeros(length(targets_case_1_and_3), dim+2)
input_data_case_1_and_3[:,1] = sqrt((mean_posterior[1] - data_case_1_and_3[1,:]).^2)
input_data_case_1_and_3[:,2] = sqrt((mean_posterior[2] - data_case_1_and_3[2,:]).^2)
input_data_case_1_and_3[:,3] = sqrt((mean_posterior[3] - data_case_1_and_3[3,:]).^2)
input_data_case_1_and_3[:,4] = sqrt(sum((repmat(mean_posterior', size(data_case_1_and_3,2))'-data_case_1_and_3[1:3,:]).^2,1))
input_data_case_1_and_3[:,end] = targets_case_1_and_3

input_data_case_2_and_4 = zeros(length(targets_case_2_and_4), dim+2)
input_data_case_2_and_4[:,1] = sqrt((mean_posterior[1] - data_case_2_and_4[1,:]).^2)
input_data_case_2_and_4[:,2] = sqrt((mean_posterior[2] - data_case_2_and_4[2,:]).^2)
input_data_case_2_and_4[:,3] = sqrt((mean_posterior[3] - data_case_2_and_4[3,:]).^2)
input_data_case_2_and_4[:,4] = data_case_1_and_3[4,:]
input_data_case_2_and_4[:,end] = targets_case_2_and_4

# tansformed data for logistic model with the ratio of gp ests are included

input_data_case_1_and_3 = zeros(length(targets_case_1_and_3), dim+3)
input_data_case_1_and_3[:,1] = sqrt((mean_posterior[1] - data_case_1_and_3[1,:]).^2)
input_data_case_1_and_3[:,2] = sqrt((mean_posterior[2] - data_case_1_and_3[2,:]).^2)
input_data_case_1_and_3[:,3] = sqrt((mean_posterior[3] - data_case_1_and_3[3,:]).^2)
input_data_case_1_and_3[:,4] = sqrt(sum((repmat(mean_posterior', size(data_case_1_and_3,2))'-data_case_1_and_3[1:3,:]).^2,1))
input_data_case_1_and_3[:,5] = data_case_1_and_3[4,:]
input_data_case_1_and_3[:,end] = targets_case_1_and_3

input_data_case_2_and_4 = zeros(length(targets_case_2_and_4), dim+3)
input_data_case_2_and_4[:,1] = sqrt((mean_posterior[1] - data_case_2_and_4[1,:]).^2)
input_data_case_2_and_4[:,2] = sqrt((mean_posterior[2] - data_case_2_and_4[2,:]).^2)
input_data_case_2_and_4[:,3] = sqrt((mean_posterior[3] - data_case_2_and_4[3,:]).^2)
input_data_case_2_and_4[:,4] = sqrt(sum((repmat(mean_posterior', size(data_case_2_and_4,2))'-data_case_2_and_4[1:3,:]).^2,1))
input_data_case_2_and_4[:,5] = data_case_2_and_4[4,:]
input_data_case_2_and_4[:,end] = targets_case_2_and_4


# features for the dt model



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



# plot features in input data


PyPlot.figure()
PyPlot.plot3D(input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),1],
              input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),2],
              input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),3],
              "*r")
PyPlot.plot3D(input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),1],
              input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),2],
              input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),3],
              "*g")
PyPlot.xlabel(L"dist \log r dim")
PyPlot.ylabel(L"dist \log \phi dim")
PyPlot.zlabel(L"dist  \sigma dim")

PyPlot.figure()
PyPlot.plot3D(input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),2],
              input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),3],
              input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),4],
              "*r")
PyPlot.plot3D(input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),2],
              input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),3],
              input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),4],
              "*g")
PyPlot.xlabel(L"dist \log \phi dim")
PyPlot.ylabel(L"dist  \sigma dim")
PyPlot.zlabel(L"joint dist")


# for the model with ratio
PyPlot.figure()
PyPlot.plot3D(input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),3],
              input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),4],
              input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),5],
              "*r")
PyPlot.plot3D(input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),3],
              input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),4],
              input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),5],
              "*g")
PyPlot.xlabel(L"dist  \sigma dim")
PyPlot.ylabel(L"joint dist")
PyPlot.zlabel(L"ratio gp ests")



PyPlot.figure()
PyPlot.plot3D(input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),1],
              input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),2],
              input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),3],
              "*r")
PyPlot.plot3D(input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),1],
              input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),2],
              input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),3],
              "*g")
PyPlot.xlabel(L"dist \log r dim")
PyPlot.ylabel(L"dist \log \phi dim")
PyPlot.zlabel(L"dist  \sigma dim")

PyPlot.figure()
PyPlot.plot3D(input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),2],
              input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),3],
              input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),4],
              "*r")
PyPlot.plot3D(input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),2],
              input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),3],
              input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),4],
              "*g")
PyPlot.xlabel(L"dist \log \phi dim")
PyPlot.ylabel(L"dist  \sigma dim")
PyPlot.zlabel(L"joint dist")



# for the model with ratio
PyPlot.figure()
PyPlot.plot3D(input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),3],
              input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),4],
              input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),5],
              "*r")
PyPlot.plot3D(input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),3],
              input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),4],
              input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),5],
              "*g")
PyPlot.xlabel(L"dist  \sigma dim")
PyPlot.ylabel(L"joint dist")
PyPlot.zlabel(L"ratio gp ests")



# use standadized features (features for decision tree model)

# 5 features model

standardization!(data_case_1_and_3)

input_data_case_1_and_3 = zeros(length(targets_case_1_and_3), dim+3)

input_data_case_1_and_3[:,1] =  data_case_1_and_3[1,:]
input_data_case_1_and_3[:,2] =  data_case_1_and_3[2,:]
input_data_case_1_and_3[:,3] =  data_case_1_and_3[3,:]
input_data_case_1_and_3[:,4] =  data_case_1_and_3[4,:]
input_data_case_1_and_3[:,5] =  data_case_1_and_3[5,:]
input_data_case_1_and_3[:,end] = targets_case_1_and_3

standardization!(data_case_2_and_4)


input_data_case_2_and_4 = zeros(length(targets_case_2_and_4), dim+3)

input_data_case_2_and_4[:,1] =  data_case_2_and_4[1,:]
input_data_case_2_and_4[:,2] =  data_case_2_and_4[2,:]
input_data_case_2_and_4[:,3] =  data_case_2_and_4[3,:]
input_data_case_2_and_4[:,4] =  data_case_2_and_4[4,:]
input_data_case_2_and_4[:,5] =  data_case_2_and_4[5,:]
input_data_case_2_and_4[:,end] = targets_case_2_and_4

# 4 features model


standardization!(data_case_1_and_3)

input_data_case_1_and_3 = zeros(length(targets_case_1_and_3), dim+2)

input_data_case_1_and_3[:,1] =  data_case_1_and_3[1,:]
input_data_case_1_and_3[:,2] =  data_case_1_and_3[2,:]
input_data_case_1_and_3[:,3] =  data_case_1_and_3[3,:]
input_data_case_1_and_3[:,4] =  data_case_1_and_3[4,:]
input_data_case_1_and_3[:,end] = targets_case_1_and_3

standardization!(data_case_2_and_4)


input_data_case_2_and_4 = zeros(length(targets_case_2_and_4), dim+2)

input_data_case_2_and_4[:,1] =  data_case_2_and_4[1,:]
input_data_case_2_and_4[:,2] =  data_case_2_and_4[2,:]
input_data_case_2_and_4[:,3] =  data_case_2_and_4[3,:]
input_data_case_2_and_4[:,4] =  data_case_2_and_4[4,:]
input_data_case_2_and_4[:,end] = targets_case_2_and_4


# plot features in input data


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
PyPlot.zlabel(L"\sigma")

PyPlot.figure()
PyPlot.plot3D(input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),2],
              input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),3],
              input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),4],
              "*r")
PyPlot.plot3D(input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),2],
              input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),3],
              input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),4],
              "*g")
PyPlot.xlabel(L"\log \phi")
PyPlot.ylabel(L"\sigma")
PyPlot.zlabel(L"ratio gp loglik est")


PyPlot.figure()
PyPlot.plot3D(input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),2],
              input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),3],
              input_data_case_1_and_3[find(x -> x==0, targets_case_1_and_3),5],
              "*r")
PyPlot.plot3D(input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),2],
              input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),3],
              input_data_case_1_and_3[find(x -> x==1, targets_case_1_and_3),5],
              "*g")
PyPlot.xlabel(L"\sigma")
PyPlot.ylabel(L"ratio gp loglik est")
PyPlot.zlabel(L"std gp est")


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
PyPlot.zlabel(L"\sigma")

PyPlot.figure()
PyPlot.plot3D(input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),2],
              input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),3],
              input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),4],
              "*r")
PyPlot.plot3D(input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),2],
              input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),3],
              input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),4],
              "*g")
PyPlot.xlabel(L"\log \phi")
PyPlot.ylabel(L"\sigma")
PyPlot.zlabel(L"ratio gp loglik est")


PyPlot.figure()
PyPlot.plot3D(input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),2],
              input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),3],
              input_data_case_2_and_4[find(x -> x==0, targets_case_2_and_4),5],
              "*r")
PyPlot.plot3D(input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),2],
              input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),3],
              input_data_case_2_and_4[find(x -> x==1, targets_case_2_and_4),5],
              "*g")
PyPlot.xlabel(L"\sigma")
PyPlot.ylabel(L"ratio gp loglik est")
PyPlot.zlabel(L"std gp est")

# Plot features

PyPlot.figure()
PyPlot.plot(data_case_1_and_3[1,:], targets_case_1_and_3, "*")

PyPlot.figure()
PyPlot.plot(data_case_1_and_3[2,:], targets_case_1_and_3, "*")


PyPlot.figure()
PyPlot.plot(data_case_1_and_3[3,:], targets_case_1_and_3, "*")


PyPlot.figure()
PyPlot.plot(data_case_1_and_3[4,:], targets_case_1_and_3, "*")


PyPlot.figure()
PyPlot.plot(data_case_1_and_3[4,:], targets_case_1_and_3, "*")

PyPlot.figure()
PyPlot.plot(data_case_1_and_3[5,:], targets_case_1_and_3, "*")


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
PyPlot.plot3D(data_case_1_and_3[1,find(x -> x==0, targets_case_1_and_3)],
              data_case_1_and_3[2,find(x -> x==0, targets_case_1_and_3)],
              data_case_1_and_3[4,find(x -> x==0, targets_case_1_and_3)],
              "*r")
PyPlot.plot3D(data_case_1_and_3[1,find(x -> x==1, targets_case_1_and_3)],
              data_case_1_and_3[2,find(x -> x==1, targets_case_1_and_3)],
              data_case_1_and_3[4,find(x -> x==1, targets_case_1_and_3)],
              "*g")
PyPlot.xlabel(L"\log r")
PyPlot.ylabel(L"\log \phi")
PyPlot.zlabel(L"ratio")


PyPlot.figure()
PyPlot.plot3D(data_case_1_and_3[3,find(x -> x==0, targets_case_1_and_3)],
              data_case_1_and_3[4,find(x -> x==0, targets_case_1_and_3)],
              data_case_1_and_3[5,find(x -> x==0, targets_case_1_and_3)],
              "*r")
PyPlot.plot3D(data_case_1_and_3[3,find(x -> x==1, targets_case_1_and_3)],
              data_case_1_and_3[4,find(x -> x==1, targets_case_1_and_3)],
              data_case_1_and_3[5,find(x -> x==1, targets_case_1_and_3)],
              "*g")
PyPlot.xlabel(L"\log \phi")
PyPlot.ylabel(L"ratio")
PyPlot.zlabel(L"std gp pred")



PyPlot.figure()
PyPlot.plot(data_case_2_and_4[1,:], targets_case_2_and_4, "*")

PyPlot.figure()
PyPlot.plot(data_case_2_and_4[2,:], targets_case_2_and_4, "*")


PyPlot.figure()
PyPlot.plot(data_case_2_and_4[3,:], targets_case_2_and_4, "*")


PyPlot.figure()
PyPlot.plot(data_case_2_and_4[4,:], targets_case_2_and_4, "*")

PyPlot.figure()
PyPlot.plot(data_case_2_and_4[5,:], targets_case_2_and_4, "*")


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
PyPlot.plot3D(data_case_1_and_3[1,find(x -> x==0, targets_case_1_and_3)],
              data_case_1_and_3[2,find(x -> x==0, targets_case_1_and_3)],
              data_case_2_and_4[4,find(x -> x==0, targets_case_1_and_3)],
              "*r")
PyPlot.plot3D(data_case_1_and_3[1,find(x -> x==1, targets_case_1_and_3)],
              data_case_1_and_3[2,find(x -> x==1, targets_case_1_and_3)],
              data_case_2_and_4[4,find(x -> x==1, targets_case_1_and_3)],
              "*g")
PyPlot.xlabel(L"\log r")
PyPlot.ylabel(L"\log \phi")
PyPlot.zlabel(L"ratio")


PyPlot.figure()
PyPlot.plot3D(data_case_1_and_3[3,find(x -> x==0, targets_case_1_and_3)],
              data_case_1_and_3[4,find(x -> x==0, targets_case_1_and_3)],
              data_case_2_and_4[5,find(x -> x==0, targets_case_1_and_3)],
              "*r")
PyPlot.plot3D(data_case_1_and_3[3,find(x -> x==1, targets_case_1_and_3)],
              data_case_1_and_3[4,find(x -> x==1, targets_case_1_and_3)],
              data_case_2_and_4[5,find(x -> x==1, targets_case_1_and_3)],
              "*g")
PyPlot.xlabel(L"\log \phi")
PyPlot.ylabel(L"ratio")
PyPlot.zlabel(L"std gp pred")




################################################################################
##  Biased coin model                                                                          ##
################################################################################

nbr_GP_star_led_GP_old = n-nbr_GP_star_geq_GP_old

prob_case_1 = nbr_case_1/nbr_GP_star_geq_GP_old
prob_case_2 = (nbr_case_2)/nbr_GP_star_led_GP_old
prob_case_3 = 1-prob_case_1
prob_case_4 = (nbr_GP_star_led_GP_old-nbr_case_2)/nbr_GP_star_led_GP_old
prob_cases = [prob_case_1;prob_case_2;prob_case_3;prob_case_4]


# test biased coin model
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
y = problem_training.data.y


case_1_theta_val = []
case_1_assumption_holds = []

case_2_theta_val = []
case_2_assumption_holds = []

case_3_theta_val = []
case_3_assumption_holds = []

case_4_theta_val = []
case_4_assumption_holds = []

for i = 1:n

  theta_new = data_test_star[1:dim,i]
  theta_old = data_test_old[1:dim,i]

  (loglik_est_star, var_pred_ml, prediction_sample_ml_star) = predict(theta_new,gp,noisy_pred)
  (loglik_est_old, var_pred_ml, prediction_sample_ml_old) = predict(theta_old,gp,noisy_pred)

  loglik_pf_old = pf(y, theta_old,problem_training.model_param.theta_known,N,false)
  loglik_pf_new = pf(y, theta_new,problem_training.model_param.theta_known,N,false)

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
##  Logistic regression model                                                                             ##
################################################################################


using GLM

# fit model for cases 2 and 4

input_data_case_1_and_3 = DataFrame(input_data_case_1_and_3)

log_reg_model_case_1_and_3 = glm(@formula(x5 ~ x1 + x2 + x3 + x4), input_data_case_1_and_3, Binomial(), LogitLink())

# extened model
log_reg_model_case_1_and_3 = glm(@formula(x6 ~ x1 + x2 + x3 + x4 + x5), input_data_case_1_and_3, Binomial(), LogitLink())


# fit model for cases 2 and 4


input_data_case_2_and_4 = DataFrame(input_data_case_2_and_4)

log_reg_model_case_2_and_4 = glm(@formula(x5 ~ x1 + x2 + x3 + x4), input_data_case_2_and_4, Binomial(), LogitLink())

# extened model
log_reg_model_case_2_and_4 = glm(@formula(x6 ~ x1 + x2 + x3 + x4 + x5), input_data_case_2_and_4, Binomial(), LogitLink())

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

  (loglik_est_star, var_pred_gp_star, prediction_sample_ml_star) = predict(data_test[1:dim,i],gp,noisy_pred)
  (loglik_est_old, var_pred_ml, prediction_sample_ml_old) = predict(res_training[1].Theta_est[:,idx_test_old[i]],gp,noisy_pred)

  theta_old = res_training[1].Theta_est[:,idx_test_old[i]]
  loglik_pf_old = pf(y, theta_old,problem_traning.model_param.theta_known,N,false)

  theta_new = data_test[1:dim,i] #res_training[1].Theta_est[:,i+n_burn_in+size(data_training,2)]
  loglik_pf_new = pf(y, theta_new,problem_traning.model_param.theta_known,N,false)

  # tansformation of theta_new
  # transform theta_new to euclidian distance space
  #=
  # old model
  theta_new_log_reg_mod = zeros(dim+1)

  theta_new_log_reg_mod[1] = sqrt((mean_posterior[1] - theta_new[1]).^2)
  theta_new_log_reg_mod[2] = sqrt((mean_posterior[2] - theta_new[2]).^2)
  theta_new_log_reg_mod[3] = sqrt((mean_posterior[3] - theta_new[3]).^2)
  theta_new_log_reg_mod[4] = sqrt(sum((mean_posterior - theta_new).^2))

  =#

  # for extened model
  theta_new_log_reg_mod = zeros(dim+2)

  theta_new_log_reg_mod[1] = sqrt((mean_posterior[1] - theta_new[1]).^2)
  theta_new_log_reg_mod[2] = sqrt((mean_posterior[2] - theta_new[2]).^2)
  theta_new_log_reg_mod[3] = sqrt((mean_posterior[3] - theta_new[3]).^2)
  theta_new_log_reg_mod[4] = sqrt(sum((mean_posterior - theta_new).^2))
  theta_new_log_reg_mod[5] = prediction_sample_ml_star[1]/prediction_sample_ml_old[1]


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



################################################################################
##  Classification tree                                                                              ##
################################################################################

# Classification tree

using DecisionTree

# tree based model for case 1 and 3

# 5 feature model
features_1_and_3 = convert(Array, input_data_case_1_and_3[:, 1:dim+2])

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

model_1_and_3 = build_tree(labels_1_and_3, features_1_and_3)

model_1_and_3 = prune_tree(model_1_and_3, 0.9)


print_tree(model_1_and_3, 3)

apply_tree(model_1_and_3, features_1_and_3[1,:])

accuracy = nfoldCV_tree(labels_1_and_3, features_1_and_3, 0.9, 3)

# tree based model for case 2 and 4

# 5 features model
features_case_2_and_4 = convert(Array, input_data_case_2_and_4[:, 1:dim+2])

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
model_2_and_4 = build_tree(labels_case_2_and_4, features_case_2_and_4)

model_2_and_4 = prune_tree(model_2_and_4, 0.9)

print_tree(model_2_and_4, 3)

apply_tree(model_2_and_4, features_case_2_and_4[1,:])

accuracy = nfoldCV_tree(labels_case_2_and_4, features_case_2_and_4, 0.9, 3)

# test model on test data

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
y = problem_training.data.y

for i = 1:n

  theta_new = data_test_star[1:dim,i]
  theta_old = data_test_old[1:dim,i]

  (loglik_est_star, var_pred_ml, prediction_sample_ml_star) = predict(theta_new,gp,noisy_pred)
  (loglik_est_old, var_pred_ml, prediction_sample_ml_old) = predict(theta_old,gp,noisy_pred)

  loglik_pf_old = pf(y, theta_old,problem_training.model_param.theta_known,N,false)
  loglik_pf_new = pf(y, theta_new,problem_training.model_param.theta_known,N,false)

  # tansformation of theta_new
  # transform theta_new to euclidian distance space

  # 5 features model
  #theta_new_tree_model = zeros(dim+2)
  #theta_new_tree_model = [theta_new; prediction_sample_ml_star[1]/prediction_sample_ml_old[1]; sqrt(var_pred_gp_star[1])]

  # 4 features model
  theta_new_tree_model = zeros(dim+1)
  theta_new_tree_model = [theta_new; prediction_sample_ml_star[1]/prediction_sample_ml_old[1]]


  if prediction_sample_ml_star[1] > prediction_sample_ml_old[1]
    #prob_case_1 = predict(beta_case_1_and_3, [1;data_test[1:dim,i]])
    case = apply_tree(model_1_and_3, theta_new_tree_model)

    if case ==  "case 1"

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

    case = apply_tree(model_2_and_4, theta_new_tree_model)

    if case == "case 2"

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
