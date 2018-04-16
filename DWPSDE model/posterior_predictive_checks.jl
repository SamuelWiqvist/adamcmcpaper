# This file contains a script to run posterior checks

using Plots
using PyPlot
using StatPlots
using KernelDensity
using Distributions
using DataFrames
using StatsBase
# set correct folder

try
  cd("DWPSDE model")
catch
  warn("Already in the DWPSDE model folder")
end

# load all functions
cd("..")
include(pwd()*"\\select case\\selectcase.jl")
cd("DWPSDE model")
include("set_up.jl")

################################################################################
#   Set up models
################################################################################

# parameters                                      					            ##

# nbr of iterations
nbr_iterations = 1000 # should be 10000

# nbr parameters
set_nbr_params = 7  # should be 7

# nbr particels
nbr_particels = 200 # should be 200

# nbr cores
nbr_of_cores= 4 # should be > 8

# brun-in
burn_in = 1

# data
sim_data = true
log_scale_prior = false

# beta_MH
beta_MH = 0.1 # should be 0.1

# algorithm
mcmc_alg = "MCWM"  # set MCWM or PMCMC

# data
data_set = "old"
dt = 0.035 # new = 0.5 old = 0.03
dt_U = 1. # new = 1 old = 1

# length training data
length_training_data = 5000 # thid should ne 5000

# job name
global_jobname = "est7_test_new_code_1000iter_local"

# load stored data
load_tranining_data = true


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
#    load results
################################################################################


load_data_from_files = true # load data from files or form some  workspace
#dagp = "_dagp" #  set to _dagp to load ER-GP file  o.w. use ""
dagp = false
jobname = "mcwm_7_par_real_data_2" # set to jobname string

# mcwm_7_para_realdata
# _dagpest7_real_dataada_gp_mcmc
# _dagpest7_real_datada_gp_mcmc_biased_coin

if load_data_from_files

    data_res = convert(Array,readtable("Results/output_res"*jobname*".csv"))

    M, N = size(data_res)

    data_param = convert(Array,readtable("Results/output_param"*jobname*".csv"))

    theta_true = data_param[1:N-2]
    burn_in = Int64(data_param[N-2+1])

    data_prior_dist = convert(Array,readtable("Results/output_prior_dist"*jobname*".csv"))

    data_prior_dist_type = convert(Array,readtable("Results/output_prior_dist_type"*jobname*".csv"))
    data_prior_dist_type = data_prior_dist_type[2]

    Z = convert(Array,readtable("Results/data_used"*jobname*".csv"))
    Z = Z[:,1]

else

    # this option should be used to load from stored .jld files

end

if dagp
  burn_in = 1
end

Theta = data_res[:,1:N-2]' # stor data in column-major order
Theta = Theta[:, burn_in:end]

################################################################################
##  Posterior predictiv distribtuion
################################################################################


# generate samples from the posterior pred distribtuion
include("run_pf_paralell.jl")

N = 500

N_sample_from_posterior = 100

posterior_dist = Categorical(1/size(Theta,2)*ones(size(Theta,2)))

posterior_pred_samples = zeros(length(Z), N_sample_from_posterior)

for i = 1:N_sample_from_posterior

  # sample from posterior
  idx = rand(posterior_dist)

  # set parameters for pf
  theta_star = Theta[:,idx]
  (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta_star, problem.model_param.theta_known,length(theta_star))
  A = A*A_sign
  # set value for constant b function
  b_const = sqrt(2.*sigma^2 / 2.)
  (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))

  # generate data from the boostrap filter
  Z_star = gen_trejectory_using_pf(Z,theta_star,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

  # store generated data set
  posterior_pred_samples[:,i] = Z_star

end

# analysis

dist_post_pred = Categorical(1/N_sample_from_posterior*ones(N_sample_from_posterior))
samples_dist_post_pred = rand(dist_post_pred,8)


text_size = 15
label_size = 15


PyPlot.figure()
PyPlot.subplot(331)
PyPlot.plot(posterior_pred_samples[:,samples_dist_post_pred[1]])
PyPlot.subplot(332)
PyPlot.plot(posterior_pred_samples[:,samples_dist_post_pred[2]])
PyPlot.subplot(333)
PyPlot.plot(posterior_pred_samples[:,samples_dist_post_pred[3]])
PyPlot.subplot(334)
PyPlot.plot(posterior_pred_samples[:,samples_dist_post_pred[4]])
PyPlot.subplot(335)
PyPlot.plot(posterior_pred_samples[:,samples_dist_post_pred[5]])
PyPlot.subplot(336)
PyPlot.plot(posterior_pred_samples[:,samples_dist_post_pred[6]])
PyPlot.subplot(337)
PyPlot.plot(posterior_pred_samples[:,samples_dist_post_pred[7]])
PyPlot.xlabel("Index",fontsize=text_size)
PyPlot.subplot(338)
PyPlot.plot(posterior_pred_samples[:,samples_dist_post_pred[8]])
PyPlot.xlabel("Index",fontsize=text_size)
PyPlot.subplot(339)
PyPlot.plot(Z, "k")
PyPlot.xlabel("Index",fontsize=text_size)

PyPlot.figure()
PyPlot.subplot(331)
PyPlot.plt[:hist](posterior_pred_samples[:,samples_dist_post_pred[1]],50)
PyPlot.ylabel("Freq",fontsize=text_size)
PyPlot.subplot(332)
PyPlot.plt[:hist](posterior_pred_samples[:,samples_dist_post_pred[2]],50)
PyPlot.subplot(333)
PyPlot.plt[:hist](posterior_pred_samples[:,samples_dist_post_pred[3]],50)
PyPlot.subplot(334)
PyPlot.plt[:hist](posterior_pred_samples[:,samples_dist_post_pred[4]],50)
PyPlot.ylabel("Freq",fontsize=text_size)
PyPlot.subplot(335)
PyPlot.plt[:hist](posterior_pred_samples[:,samples_dist_post_pred[5]],50)
PyPlot.subplot(336)
PyPlot.plt[:hist](posterior_pred_samples[:,samples_dist_post_pred[6]],50)
PyPlot.subplot(337)
PyPlot.plt[:hist](posterior_pred_samples[:,samples_dist_post_pred[7]],50)
PyPlot.ylabel("Freq",fontsize=text_size)
PyPlot.subplot(338)
PyPlot.plt[:hist](posterior_pred_samples[:,samples_dist_post_pred[8]],50)
PyPlot.subplot(339)
PyPlot.plt[:hist](Z,50, color = "k")

PyPlot.figure()
for i = 1:8
  h = kde(posterior_pred_samples[:,samples_dist_post_pred[8]])
  PyPlot.plot(h.x,h.density)
end
h = kde(Z)
PyPlot.plot(h.x,h.density)

PyPlot.figure()
h = PyPlot.plt[:hist](mean(posterior_pred_samples,1)[:],20)
PyPlot.plot((mean(Z), mean(Z)), (0, maximum(h[1])), "k")


PyPlot.figure()
h = PyPlot.plt[:hist](std(posterior_pred_samples,1)[:],20)
PyPlot.plot((std(Z), std(Z)), (0, maximum(h[1])), "k")

posterior_pred_auto_cov = autocov(posterior_pred_samples, 1:5)
data_auto_cov = autocov(Z, 1:5)

PyPlot.figure()
h = PyPlot.plt[:hist](posterior_pred_auto_cov[1,:],20)
PyPlot.plot((data_auto_cov[1], data_auto_cov[1]), (0, maximum(h[1])), "k")

PyPlot.figure()
h = PyPlot.plt[:hist](posterior_pred_auto_cov[2,:],20)
PyPlot.plot((data_auto_cov[2], data_auto_cov[2]), (0, maximum(h[1])), "k")

PyPlot.figure()
h = PyPlot.plt[:hist](posterior_pred_auto_cov[3,:],20)
PyPlot.plot((data_auto_cov[3], data_auto_cov[3]), (0, maximum(h[1])), "k")

PyPlot.figure()
h = PyPlot.plt[:hist](posterior_pred_auto_cov[4,:],20)
PyPlot.plot((data_auto_cov[4], data_auto_cov[4]), (0, maximum(h[1])), "k")

PyPlot.figure()
h = PyPlot.plt[:hist](posterior_pred_auto_cov[5,:],20)
PyPlot.plot((data_auto_cov[5], data_auto_cov[5]), (0, maximum(h[1])), "k")


# calc hist for nbr break points

# print path for real data
print("Results/data_used"*dagp*jobname*".csv")

# write data from posterior pred dist
writetable("./Results/post_pred_data.csv", convert(DataFrame, posterior_pred_samples))

nbr_cp_real_data = length(readdlm("Results/cp_data_real_data.txt")[2:end])

nbr_cp_post_pred = readdlm("Results/cp_data_post_pred.txt")[2:end]

# all 15 for mcwm
