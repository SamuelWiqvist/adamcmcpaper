# Script for evaluating the GP model

# 1) load correct files
# 2) fix the code (remove unnecessary parts)

using JLD
using HDF5

include(pwd()*"/DWPSDE model/set_up.jl")
include(pwd()*"/utilities/posteriorinference.jl")
include(pwd()*"/utilities/normplot.jl")


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
nbr_iterations = 1000 # should be 20000

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
problem.alg_param.beta_MH = beta_MH

#problem.alg_param.print_interval = 500



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

	#=
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
	=#
else

  #@load "gp_training_2_par_training_and_test_data_test_new_code_structure.jld"

  #@load "gp_training_7_par_training_and_test_new_data.jld"

  #@load "gp_training_7_par_training_and_test_lunarc.jld"
  if job == "simdata"
		@load "DWPSDE model/gp_training_7_par_training_and_testsimdatalunarc_new.jld"
	elseif job == "new_data"
		@load "DWPSDE model/gp_training_7_par_training_and_testnew_datalunarc_new.jld"
		#@load "fited_gp_new_data.jld"
	end

end


################################################################################
###            plot training data                                            ###
################################################################################

#export_data(problem_training, res_training[1],"dagpMCMC_training_data"*jobname)

plot_theta_true = true


using PyPlot

text_size = 15

PyPlot.figure(figsize=(10,20))

ax1 = PyPlot.subplot(711)
PyPlot.plot(res_training[1].Theta_est[1,:])
plot_theta_true == true ? PyPlot.plot(problem.model_param.theta_true[1]*ones(size(res_training[1].Theta_est)[2]), "k") :
PyPlot.ylabel(L"$\log \kappa$",fontsize=text_size)
ax1[:axes][:get_xaxis]()[:set_ticks]([])
ax2 = PyPlot.subplot(712)
PyPlot.plot(res_training[1].Theta_est[2,:])
plot_theta_true == true ? PyPlot.plot(problem.model_param.theta_true[2]*ones(size(res_training[1].Theta_est)[2]), "k") :
PyPlot.ylabel(L"$\log \gamma$",fontsize=text_size)
ax2[:axes][:get_xaxis]()[:set_ticks]([])
PyPlot.subplot(713)
PyPlot.plot(res_training[1].Theta_est[3,:])
plot_theta_true == true ? PyPlot.plot(problem.model_param.theta_true[3]*ones(size(res_training[1].Theta_est)[2]), "k") :
PyPlot.ylabel(L"$\log c$",fontsize=text_size)
PyPlot.subplot(714)
PyPlot.plot(res_training[1].Theta_est[4,:])
plot_theta_true == true ? PyPlot.plot(problem.model_param.theta_true[4]*ones(size(res_training[1].Theta_est)[2]), "k") :
PyPlot.ylabel(L"$\log d$",fontsize=text_size)
PyPlot.subplot(715)
PyPlot.plot(res_training[1].Theta_est[5,:])
plot_theta_true == true ? PyPlot.plot(problem.model_param.theta_true[5]*ones(size(res_training[1].Theta_est)[2]), "k") :
PyPlot.ylabel(L"$\log p_1$",fontsize=text_size)
PyPlot.subplot(716)
PyPlot.plot(res_training[1].Theta_est[6,:])
plot_theta_true == true ? PyPlot.plot(problem.model_param.theta_true[6]*ones(size(res_training[1].Theta_est)[2]), "k") :
PyPlot.ylabel(L"$\log p_2$",fontsize=text_size)
PyPlot.subplot(717)
PyPlot.plot(res_training[1].Theta_est[7,:])
plot_theta_true == true ? PyPlot.plot(problem.model_param.theta_true[7]*ones(size(res_training[1].Theta_est)[2]), "k") :
PyPlot.ylabel(L"$\log \sigma$",fontsize=text_size)
PyPlot.xlabel("Iteration",fontsize=text_size)

PyPlot.figure()
PyPlot.plot(res_training[1].loglik_est)

for i = 1:set_nbr_params
  PyPlot.figure()
  PyPlot.plot(theta_training[i,:])
  PyPlot.plot(problem.model_param.theta_true[i]*ones(size(theta_training[i,:])), "k")
end


for i = 1:set_nbr_params
  PyPlot.figure()
  h = PyPlot.plt[:hist](theta_training[i,:],100)
  PyPlot.plot((problem.model_param.theta_true[i], problem.model_param.theta_true[i]), (0, maximum(h[1])+5), "k");
end

PyPlot.figure()
PyPlot.plt[:hist](loglik_training,100)

PyPlot.figure()
PyPlot.plot(loglik_training)


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
#data_test = data_training
#data_training = data_training

# fit GP model
if true #problem.alg_param.est_method == "ml"
  # fit GP model using ml
  #perc_outlier = 0.1 # used when using PMCMC for trainig data 0.05
  #tail_rm = "left"

  perc_outlier = 0.01 # 0.02 for simdata, 0.01 also works for new_data
  tail_rm = "left"
  lasso = false # was true test fitting without lassa, The lassa has a large inpact on the fit of the model, we should use lasso!

  ml_est(gp, data_training,"SE", lasso,perc_outlier,tail_rm)
else
  error("The two stage estimation method is not in use")
  #two_stage_est(gp, data_training)
end

time_fit_gp = toc()


################################################################################
##  Compare loglik predictions                                                                           ##
################################################################################

text_size = 15
loglik_pf = data_test[end,:]

(loglik_mean,loglik_std,loglik_sample) = predict(data_test[1:end-1,:], gp, problem.alg_param.noisy_est)

# plot param vs loglik values
PyPlot.figure(figsize=(10,5))
PyPlot.plot(loglik_pf, "*b",alpha=0.3)
PyPlot.plot(loglik_sample, "*r",alpha=0.3)
PyPlot.ylabel(L"$\ell$",fontsize=text_size)
PyPlot.xlabel(L"Index",fontsize=text_size)

for i = 1:set_nbr_params
  PyPlot.figure()
  PyPlot.plot(data_test[i,:], loglik_pf, "*b",alpha=0.3)
  PyPlot.plot(data_test[i,:], loglik_sample, "*r",alpha=0.3)
end

for i = 1:set_nbr_params
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

# todo

residuals = loglik_pf - loglik_sample

describe(residuals)

# plot residuals
PyPlot.figure(figsize=(10,5))
PyPlot.plot(residuals)
PyPlot.ylabel("Residual",fontsize=text_size)
PyPlot.xlabel("Index",fontsize=text_size)


PyPlot.figure(figsize=(7,5))
PyPlot.plot(data_test[1,:], residuals, "*b")
PyPlot.ylabel("Residual",fontsize=text_size)
PyPlot.xlabel(L"$\log \kappa$",fontsize=text_size)

PyPlot.figure(figsize=(7,5))
PyPlot.plot(data_test[2,:], residuals, "*b")
PyPlot.ylabel("Residual",fontsize=text_size)
PyPlot.xlabel(L"$\log \gamma$",fontsize=text_size)


PyPlot.figure(figsize=(7,5))
PyPlot.plot(data_test[3,:], residuals, "*b")
PyPlot.ylabel("Residual",fontsize=text_size)
PyPlot.xlabel(L"$\log c$",fontsize=text_size)


PyPlot.figure(figsize=(7,5))
PyPlot.plot(data_test[4,:], residuals, "*b")
PyPlot.ylabel("Residual",fontsize=text_size)
PyPlot.xlabel(L"$\log d$",fontsize=text_size)


PyPlot.figure(figsize=(7,5))
PyPlot.plot(data_test[5,:], residuals, "*b")
PyPlot.ylabel("Residual",fontsize=text_size)
PyPlot.xlabel(L"$\log p_1$",fontsize=text_size)


PyPlot.figure(figsize=(7,5))
PyPlot.plot(data_test[6,:], residuals, "*b")
PyPlot.ylabel("Residual",fontsize=text_size)
PyPlot.xlabel(L"$\log p_2$",fontsize=text_size)


PyPlot.figure(figsize=(7,5))
PyPlot.plot(data_test[7,:], residuals, "*b")
PyPlot.ylabel("Residual",fontsize=text_size)
PyPlot.xlabel(L"$\log \sigma$",fontsize=text_size)


PyPlot.figure(figsize=(7,5))
PyPlot.plot(data_test[8,:], residuals, "*b")
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

Z = problem.data.Z
theta_true = problem.model_param.theta_true
theta_known = problem.model_param.theta_known
N = 200  #problem.alg_param.N
prior_parameters = problem.prior_dist.prior_parameters

include("run_pf_paralell.jl")


theta = theta_true
(Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, problem.model_param.theta_known,length(theta))
A = A*A_sign
# set value for constant b function
b_const = sqrt(2.*sigma^2 / 2.)
(subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
(loglik_pf_old, 	~,  ~) = @time run_pf_paralell(Z,theta,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)
(loglik_mean,loglik_std,loglik_sample) = @time predict(theta, gp, problem.alg_param.noisy_est)



# kappa non-fixed

kappa_vec = -1.6:0.01:-0.3

loglik_kappa_non_fixes = zeros(2,length(kappa_vec))

for i = 1:length(kappa_vec)

  theta = [kappa_vec[i];theta_true[2:end]]
  (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, problem.model_param.theta_known,length(theta))
  A = A*A_sign
  # set value for constant b function
  b_const = sqrt(2.*sigma^2 / 2.)
  (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
  (loglik_kappa_non_fixes[1,i], 	~,  ~) = run_pf_paralell(Z,theta,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)
  (loglik_mean,loglik_std,loglik_sample) = predict(theta, gp, problem.alg_param.noisy_est)
  loglik_kappa_non_fixes[2,i] = loglik_sample[1]

end


PyPlot.figure(figsize=(7,5))
PyPlot.plot(kappa_vec, loglik_kappa_non_fixes[1,:], "b")
PyPlot.plot(kappa_vec, loglik_kappa_non_fixes[2,:], "r")
PyPlot.plot((theta_true[1], theta_true[1]), (minimum(loglik_kappa_non_fixes[find(!isnan, loglik_kappa_non_fixes)]), maximum(loglik_kappa_non_fixes[find(!isnan, loglik_kappa_non_fixes)])), "k")
PyPlot.plot((maximum(data_training[1,:]), maximum(data_training[1,:])), (minimum(loglik_kappa_non_fixes[find(!isnan, loglik_kappa_non_fixes)]), maximum(loglik_kappa_non_fixes[find(!isnan, loglik_kappa_non_fixes)])), "k--")
PyPlot.plot((minimum(data_training[1,:]), minimum(data_training[1,:])), (minimum(loglik_kappa_non_fixes[find(!isnan, loglik_kappa_non_fixes)]), maximum(loglik_kappa_non_fixes[find(!isnan, loglik_kappa_non_fixes)])), "k--")
PyPlot.xlabel(L"$\log \kappa$",fontsize=text_size)
PyPlot.ylabel(L"$\ell$",fontsize=text_size)

# gamma non-fixed

gamma_vec = -0.3:0.01:0.1

loglik_gamma_non_fixes = zeros(2,length(gamma_vec))

for i = 1:length(gamma_vec)

  theta = [theta_true[1];gamma_vec[i];theta_true[3:end]]
  (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, problem.model_param.theta_known,length(theta))
  A = A*A_sign
  # set value for constant b function
  b_const = sqrt(2.*sigma^2 / 2.)
  (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
  (loglik_gamma_non_fixes[1,i], 	~,  ~) = run_pf_paralell(Z,theta,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)
  (loglik_mean,loglik_std,loglik_sample) = predict(theta, gp, problem.alg_param.noisy_est)
  loglik_gamma_non_fixes[2,i] = loglik_sample[1]

end


PyPlot.figure(figsize=(7,5))
PyPlot.plot(gamma_vec, loglik_gamma_non_fixes[1,:], "b")
PyPlot.plot(gamma_vec, loglik_gamma_non_fixes[2,:], "r")
PyPlot.plot((theta_true[2], theta_true[2]), (minimum(loglik_gamma_non_fixes[find(!isnan, loglik_gamma_non_fixes)]), maximum(loglik_gamma_non_fixes[find(!isnan, loglik_gamma_non_fixes)])), "k")
PyPlot.plot((maximum(data_training[2,:]), maximum(data_training[2,:])), (minimum(loglik_gamma_non_fixes[find(!isnan, loglik_gamma_non_fixes)]), maximum(loglik_gamma_non_fixes[find(!isnan, loglik_gamma_non_fixes)])), "k--")
PyPlot.plot((minimum(data_training[2,:]), minimum(data_training[2,:])), (minimum(loglik_gamma_non_fixes[find(!isnan, loglik_gamma_non_fixes)]), maximum(loglik_gamma_non_fixes[find(!isnan, loglik_gamma_non_fixes)])), "k--")
PyPlot.xlabel(L"$\log \gamma$",fontsize=text_size)
PyPlot.ylabel(L"$\ell$",fontsize=text_size)

# c non-fixed

c_vec = 3:0.01:4

loglik_c_non_fixes = zeros(2,length(c_vec))

for i = 1:length(c_vec)

  theta = [theta_true[1:2];c_vec[i];theta_true[4:end]]
  (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, problem.model_param.theta_known,length(theta))
  A = A*A_sign
  # set value for constant b function
  b_const = sqrt(2.*sigma^2 / 2.)
  (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
  (loglik_c_non_fixes[1,i], 	~,  ~) = run_pf_paralell(Z,theta,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)
  (loglik_mean,loglik_std,loglik_sample) = predict(theta, gp, problem.alg_param.noisy_est)
  loglik_c_non_fixes[2,i] = loglik_sample[1]

end


PyPlot.figure(figsize=(7,5))
PyPlot.plot(c_vec, loglik_c_non_fixes[1,:], "b")
PyPlot.plot(c_vec, loglik_c_non_fixes[2,:], "r")
PyPlot.plot((theta_true[3], theta_true[3]), (minimum(loglik_c_non_fixes[find(!isnan, loglik_c_non_fixes)]), maximum(loglik_c_non_fixes[find(!isnan, loglik_c_non_fixes)])), "k")
PyPlot.plot((maximum(data_training[3,:]), maximum(data_training[3,:])), (minimum(loglik_c_non_fixes[find(!isnan, loglik_c_non_fixes)]), maximum(loglik_c_non_fixes[find(!isnan, loglik_c_non_fixes)])), "k--")
PyPlot.plot((minimum(data_training[3,:]), minimum(data_training[3,:])), (minimum(loglik_c_non_fixes[find(!isnan, loglik_c_non_fixes)]), maximum(loglik_c_non_fixes[find(!isnan, loglik_c_non_fixes)])), "k--")
PyPlot.xlabel(L"$\log c$",fontsize=text_size)
PyPlot.ylabel(L"$\ell$",fontsize=text_size)

# d non-fixed

d_vec = 0.1:0.01:3

loglik_d_non_fixes = zeros(2,length(d_vec))

for i = 1:length(d_vec)

  theta = [theta_true[1:3];d_vec[i];theta_true[5:end]]
  (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, problem.model_param.theta_known,length(theta))
  A = A*A_sign
  # set value for constant b function
  b_const = sqrt(2.*sigma^2 / 2.)
  (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
  (loglik_d_non_fixes[1,i], 	~,  ~) = run_pf_paralell(Z,theta,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)
  (loglik_mean,loglik_std,loglik_sample) = predict(theta, gp, problem.alg_param.noisy_est)
  loglik_d_non_fixes[2,i] = loglik_sample[1]

end


PyPlot.figure(figsize=(7,5))
PyPlot.plot(d_vec, loglik_d_non_fixes[1,:], "b")
PyPlot.plot(d_vec, loglik_d_non_fixes[2,:], "r")
PyPlot.plot((theta_true[4], theta_true[4]), (minimum(loglik_d_non_fixes[find(!isnan, loglik_d_non_fixes)]), maximum(loglik_d_non_fixes[find(!isnan, loglik_d_non_fixes)])), "k")
PyPlot.plot((maximum(data_training[4,:]), maximum(data_training[4,:])), (minimum(loglik_d_non_fixes[find(!isnan, loglik_d_non_fixes)]), maximum(loglik_d_non_fixes[find(!isnan, loglik_d_non_fixes)])), "k--")
PyPlot.plot((minimum(data_training[4,:]), minimum(data_training[4,:])), (minimum(loglik_d_non_fixes[find(!isnan, loglik_d_non_fixes)]), maximum(loglik_d_non_fixes[find(!isnan, loglik_d_non_fixes)])), "k--")
PyPlot.xlabel(L"$\log d$",fontsize=text_size)
PyPlot.ylabel(L"$\ell$",fontsize=text_size)

# p1 non-fixed

p1_vec = -0.5:0.01:2

loglik_p1_non_fixes = zeros(2,length(p1_vec))

for i = 1:length(p1_vec)

  theta = [theta_true[1:4];p1_vec[i];theta_true[6:end]]
  (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, problem.model_param.theta_known,length(theta))
  A = A*A_sign
  # set value for constant b function
  b_const = sqrt(2.*sigma^2 / 2.)
  (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
  (loglik_p1_non_fixes[1,i], 	~,  ~) = run_pf_paralell(Z,theta,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)
  (loglik_mean,loglik_std,loglik_sample) = predict(theta, gp, problem.alg_param.noisy_est)
  loglik_p1_non_fixes[2,i] = loglik_sample[1]

end


PyPlot.figure(figsize=(7,5))
PyPlot.plot(p1_vec, loglik_p1_non_fixes[1,:], "b")
PyPlot.plot(p1_vec, loglik_p1_non_fixes[2,:], "r")
PyPlot.plot((theta_true[5], theta_true[5]), (minimum(loglik_p1_non_fixes[find(!isnan, loglik_p1_non_fixes)]), maximum(loglik_p1_non_fixes[find(!isnan, loglik_p1_non_fixes)])), "k")
PyPlot.plot((maximum(data_training[5,:]), maximum(data_training[5,:])), (minimum(loglik_p1_non_fixes[find(!isnan, loglik_p1_non_fixes)]), maximum(loglik_p1_non_fixes[find(!isnan, loglik_p1_non_fixes)])), "k--")
PyPlot.plot((minimum(data_training[5,:]), minimum(data_training[5,:])), (minimum(loglik_p1_non_fixes[find(!isnan, loglik_p1_non_fixes)]), maximum(loglik_p1_non_fixes[find(!isnan, loglik_p1_non_fixes)])), "k--")
PyPlot.xlabel(L"$\log p_1$",fontsize=text_size)
PyPlot.ylabel(L"$\ell$",fontsize=text_size)

# p2 non-fixed

p2_vec = -0.5:0.01:2

loglik_p2_non_fixes = zeros(2,length(p2_vec))

for i = 1:length(p2_vec)

  theta = [theta_true[1:4];p1_vec[i];theta_true[6:end]]
  (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, problem.model_param.theta_known,length(theta))
  A = A*A_sign
  # set value for constant b function
  b_const = sqrt(2.*sigma^2 / 2.)
  (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
  (loglik_p2_non_fixes[1,i], 	~,  ~) = run_pf_paralell(Z,theta,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)
  (loglik_mean,loglik_std,loglik_sample) = predict(theta, gp, problem.alg_param.noisy_est)
  loglik_p2_non_fixes[2,i] = loglik_sample[1]

end


PyPlot.figure(figsize=(7,5))
PyPlot.plot(p2_vec, loglik_p2_non_fixes[1,:], "b")
PyPlot.plot(p2_vec, loglik_p2_non_fixes[2,:], "r")
PyPlot.plot((theta_true[6], theta_true[6]), (minimum(loglik_p2_non_fixes[find(!isnan, loglik_p2_non_fixes)]), maximum(loglik_p2_non_fixes[find(!isnan, loglik_p2_non_fixes)])), "k")
PyPlot.plot((maximum(data_training[6,:]), maximum(data_training[6,:])), (minimum(loglik_p2_non_fixes[find(!isnan, loglik_p2_non_fixes)]), maximum(loglik_p2_non_fixes[find(!isnan, loglik_p2_non_fixes)])), "k--")
PyPlot.plot((minimum(data_training[6,:]), minimum(data_training[6,:])), (minimum(loglik_p2_non_fixes[find(!isnan, loglik_p2_non_fixes)]), maximum(loglik_p2_non_fixes[find(!isnan, loglik_p2_non_fixes)])), "k--")
PyPlot.xlabel(L"$\log p_2$",fontsize=text_size)
PyPlot.ylabel(L"$\ell$",fontsize=text_size)

# sigma non-fixed

sigma_vec = 0:0.01:1

loglik_sigma_non_fixes = zeros(2,length(sigma_vec))

for i = 1:length(sigma_vec)

  theta = [theta_true[1:6];sigma_vec[i]]
  (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, problem.model_param.theta_known,length(theta))
  A = A*A_sign
  # set value for constant b function
  b_const = sqrt(2.*sigma^2 / 2.)
  (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
  (loglik_sigma_non_fixes[1,i], 	~,  ~) = run_pf_paralell(Z,theta,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)
  (loglik_mean,loglik_std,loglik_sample) = predict(theta, gp, problem.alg_param.noisy_est)
  loglik_sigma_non_fixes[2,i] = loglik_sample[1]

end


PyPlot.figure(figsize=(7,5))
PyPlot.plot(sigma_vec, loglik_sigma_non_fixes[1,:], "b")
PyPlot.plot(sigma_vec, loglik_sigma_non_fixes[2,:], "r")
PyPlot.plot((theta_true[7], theta_true[7]), (minimum(loglik_sigma_non_fixes[find(!isnan, loglik_sigma_non_fixes)]), maximum(loglik_sigma_non_fixes[find(!isnan, loglik_sigma_non_fixes)])), "k")
PyPlot.plot((maximum(data_training[7,:]), maximum(data_training[7,:])), (minimum(loglik_sigma_non_fixes[find(!isnan, loglik_sigma_non_fixes)]), maximum(loglik_sigma_non_fixes[find(!isnan, loglik_sigma_non_fixes)])), "k--")
PyPlot.plot((minimum(data_training[7,:]), minimum(data_training[7,:])), (minimum(loglik_sigma_non_fixes[find(!isnan, loglik_sigma_non_fixes)]), maximum(loglik_sigma_non_fixes[find(!isnan, loglik_sigma_non_fixes)])), "k--")
PyPlot.xlabel(L"$\log \sigma$",fontsize=text_size)
PyPlot.ylabel(L"$\ell$",fontsize=text_size)


################################################################################
##  Selecting case problem                                                    ##
################################################################################

# Create features for classification models                                                                            ##

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





# use tranformed data + standardization (logistic regression model)

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
#PyPlot.xlabel(L"dist \log r dim")
#PyPlot.ylabel(L"dist \log \phi dim")
#PyPlot.zlabel(L"dist  \sigma dim")

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

# 8 features model


standardization!(data_case_1_and_3)

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

standardization!(data_case_2_and_4)


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

# fit model, i.e. est probabilities
nbr_GP_star_led_GP_old = n-nbr_GP_star_geq_GP_old

prob_case_1 = nbr_case_1/nbr_GP_star_geq_GP_old
prob_case_2 = (nbr_GP_star_led_GP_old-nbr_case_4)/nbr_GP_star_led_GP_old
prob_case_3 = 1-prob_case_1
prob_case_4 = nbr_case_4/nbr_GP_star_led_GP_old
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

include("run_pf_paralell.jl")
N = 25
Z = problem_training.data.Z


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
  (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta_old, problem.model_param.theta_known,length(theta_old))
  A = A*A_sign
  # set value for constant b function
  b_const = sqrt(2.*sigma^2 / 2.)
  (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
  (loglik_pf_old, 	~,  ~) = run_pf_paralell(Z,theta_old,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

  theta_new = data_test[1:dim,i] #res_training[1].Theta_est[:,i+n_burn_in+size(data_training,2)]
  (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta_new, problem.model_param.theta_known,length(theta_old))
  A = A*A_sign
  # set value for constant b function
  b_const = sqrt(2.*sigma^2 / 2.)
  (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
  (loglik_pf_new, 	~,  ~) = run_pf_paralell(Z,theta_new,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)


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

log_reg_model_case_1_and_3 = glm(@formula(x10 ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8  + x9), input_data_case_1_and_3, Binomial(), LogitLink())



# fit model for cases 2 and 4


input_data_case_2_and_4 = DataFrame(input_data_case_2_and_4)

log_reg_model_case_2_and_4 = glm(@formula(x10 ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8  + x9), input_data_case_2_and_4, Binomial(), LogitLink())

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

#theta_new_tree_model = zeros(dim+1)
#theta_new_tree_model = [theta_new; prediction_sample_ml_star[1]/prediction_sample_ml_old[1]]
#case = apply_tree(model_1_and_3, theta_new_tree_model)
#case = apply_tree(model_2_and_4, theta_new_tree_model)


include("run_pf_paralell.jl")
N = 250
Z = problem_training.data.Z


case_1_theta_val = []
case_1_assumption_holds = []

case_2_theta_val = []
case_2_assumption_holds = []

case_3_theta_val = []
case_3_assumption_holds = []

case_4_theta_val = []
case_4_assumption_holds = []

for i = 1:n

  println(i)

  (loglik_est_star, var_pred_ml, prediction_sample_ml_star) = predict(data_test[1:dim,i],gp,noisy_pred)
  (loglik_est_old, var_pred_ml, prediction_sample_ml_old) = predict(res_training[1].Theta_est[:,idx_test_old[i]],gp,noisy_pred)

  theta_old = res_training[1].Theta_est[:,idx_test_old[i]]
  (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta_old, problem.model_param.theta_known,length(theta_old))
  A = A*A_sign
  # set value for constant b function
  b_const = sqrt(2.*sigma^2 / 2.)
  (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
  (loglik_pf_old, 	~,  ~) = run_pf_paralell(Z,theta_old,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

  theta_new = data_test[1:dim,i] #res_training[1].Theta_est[:,i+n_burn_in+size(data_training,2)]
  (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta_new, problem.model_param.theta_known,length(theta_old))
  A = A*A_sign
  # set value for constant b function
  b_const = sqrt(2.*sigma^2 / 2.)
  (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
  (loglik_pf_new, 	~,  ~) = run_pf_paralell(Z,theta_new,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

  theta_new_tree_model = zeros(dim+1)
  theta_new_tree_model = [theta_new; prediction_sample_ml_star[1]/prediction_sample_ml_old[1]]


  if prediction_sample_ml_star[1] > prediction_sample_ml_old[1]
    #go_to_case_1 = rand() < prob_case_1
    case = apply_tree(model_1_and_3, theta_new_tree_model)

    if case ==  "case 1"

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

    case = apply_tree(model_2_and_4, theta_new_tree_model)

    if case == "case 2"

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

nbr_case_selceted = [nbr_case_1_selceted nbr_case_2_selceted nbr_case_3_selceted nbr_case_4_selceted]

nbr_case_correct = [nbr_case_1_correct nbr_case_2_correct nbr_case_3_correct nbr_case_4_correct]

prob_correct_given_selected = nbr_case_correct./nbr_case_selceted
