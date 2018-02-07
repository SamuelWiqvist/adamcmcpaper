

include("set_up.jl")

using JLD
using HDF5


# set parameters
nbr_iterations = 2000
nbr_particels = 25
nbr_of_cores= 4
burn_in = 1
sim_data = true
set_nbr_params = 2
log_scale_prior = false
beta_MH = 0.1
mcmc_alg = "MCWM"  # set MCWM or PMCMC

data_set = "old"
dt = 0.035 # new = 0.5 old = 0.03
dt_U = 1. # new = 1 old = 1

length_training_data = 1000


accelerated_da = false

load_tranining_data = true

################################################################################
##                         set DA problem                               ##
################################################################################

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
##                         training data                                      ##
################################################################################


# set parameters
burn_in = 500
nbr_iterations = burn_in+length_training_data
nbr_particels = 25
nbr_of_cores = 4

################################################################################
##                         set model parameters  for training                               ##
################################################################################

problem_training = set_up_problem(nbr_of_unknown_parameters=set_nbr_params, use_sim_data = sim_data, data_set = data_set)
problem_training.alg_param.alg = mcmc_alg
problem_training.alg_param.R = nbr_iterations
problem_training.alg_param.N = nbr_particels
problem_training.alg_param.burn_in = burn_in
problem_training.alg_param.nbr_of_cores = nbr_of_cores
problem_training.alg_param.dt = dt
problem_training.alg_param.dt_U = dt_U
problem_training.alg_param.alg = mcmc_alg
problem_training.adaptive_update =  AMUpdate_gen(eye(set_nbr_params), 1/sqrt(set_nbr_params), 0.15, 1, 0.8, 25)



################################################################################
##                generate training data                                     ###
################################################################################

if !load_tranining_data

  if !log_scale_prior
    tic()
    res_training, theta_training, loglik_training, cov_matrix = MCMC(problem_training, true, true)
    time_pre_er = toc()
    #export_parameters(res_problem_normal_prior_est_AM_gen[2],jobname)
  else
    tic()
    res_training, theta_training, loglik_training, cov_matrix  = @time MCMC(problem_training_nonlog, true, true)
    time_pre_er = toc()
  end

  save("gp_training.jld", "res_training", res_training, "theta_training", theta_training, "loglik_training", loglik_training,"cov_matrix",cov_matrix)

else

  #@load "gp_training_$(set_nbr_params)_par.jld"
  @load "gp_training_$(set_nbr_params)_par.jld"

end



################################################################################
###            plot training data                                            ###
################################################################################

#export_data(problem_training, res_training[1],"dagpMCMC_training_data"*jobname)

using PyPlot


for i = 1:set_nbr_params
PyPlot.figure()
PyPlot.plot(res_training[1].Theta_est[i,:])
PyPlot.plot(problem.model_param.theta_true[i]*ones(size(res_training[1].Theta_est)[2]))
end

for i = 1:set_nbr_params
PyPlot.figure()
PyPlot.plt[:hist](theta_training[i,:],100)
end

PyPlot.figure()
PyPlot.plt[:hist](loglik_training,100)

PyPlot.figure()
PyPlot.plot(res_training[1].loglik_est)



################################################################################
###          Fit GP model                                                     ##
################################################################################

using GaussianProcesses

#data_training = [theta_training[; loglik_training']
#data_test = [theta_test; loglik_test']
data_training = [theta_training; loglik_training']
data_test = data_training[:, Int(size(data_training)[2]/2+1):end ]
data_training = data_training[:, 1:Int(size(data_training)[2]/2)]

X_training = data_training[1:end-1,:]
Y_training = data_training[end,:]
mean_training = mean(Y_training)
Y_training_stad = Y_training - mean_training

X_test = data_test[1:end-1,:]
Y_test = data_test[end,:]
mean_test = mean(Y_training)
Y_test_stad = Y_training - mean_training


PyPlot.figure()
PyPlot.scatter3D(X_training[1,:],X_training[2,:],Y_training)

PyPlot.figure()
PyPlot.scatter3D(X_training[1,:],X_training[2,:],Y_training_stad)

kernel_function = LinArd(zeros(set_nbr_params))*LinArd(zeros(set_nbr_params)) + SEArd(zeros(set_nbr_params), log(1))
mean_function = MeanZero()

gp_model = GP(X_training, Y_test_stad, mean_function, kernel_function)



(μ_GP, σ_GP) = predict_f(gp_model, X_test)

(μ_GP, σ2_GP) = predict_y(gp_model, X_test)

pred_samples = similar(μ_GP)

for i = 1:length(μ_GP)
  pred_samples[i]  = rand(Normal(μ_GP[i], sqrt(σ2_GP[i])))
end


PyPlot.figure()
PyPlot.scatter3D(X_test[1,:],X_test[2,:],pred_samples+mean_training, "b")

PyPlot.hold(true)
PyPlot.scatter3D(X_test[1,:],X_test[2,:],Y_test, "r")


PyPlot.figure()
PyPlot.scatter3D(X_test[1,:],X_test[2,:],μ_GP+mean_training, "b")

PyPlot.hold(true)
PyPlot.scatter3D(X_test[1,:],X_test[2,:],Y_test, "r")



for i = 1:set_nbr_params
  PyPlot.figure()
  PyPlot.plot(X_test[i,:], Y_test, "*r",alpha=0.3)
  PyPlot.plot(X_test[i,:], pred_samples+mean_training, "*b",alpha=0.3)
end


for i = 1:set_nbr_params
  PyPlot.figure()
  PyPlot.plot(X_test[i,:], Y_test, "*r",alpha=0.3)
  PyPlot.plot(X_test[i,:], μ_GP+mean_training, "*b",alpha=0.3)
end

# fit hyperparameters

optimize!(gp_model)



(μ_GP, σ2_GP) = predict_y(gp_model, X_test)

pred_samples = similar(μ_GP)

for i = 1:length(μ_GP)
  pred_samples[i]  = rand(Normal(μ_GP[i], sqrt(σ2_GP[i])))
end


PyPlot.figure()
PyPlot.scatter3D(X_test[1,:],X_test[2,:],pred_samples+mean_training, "b")

PyPlot.hold(true)
PyPlot.scatter3D(X_test[1,:],X_test[2,:],Y_test, "r")


PyPlot.figure()
PyPlot.scatter3D(X_test[1,:],X_test[2,:],μ_GP+mean_training, "b")

PyPlot.hold(true)
PyPlot.scatter3D(X_test[1,:],X_test[2,:],Y_test, "r")
