# Script for running the PMCMC algorithm

include("set_up.jl")

using JLD
using HDF5
using StatPlots

# set correct path
try
  cd("DWPSDE model")
catch
  warn("Already in the Ricker model folder")
end

# load functions to compute posterior inference
if Sys.CPU_CORES == 8
    include("C:\\Users\\samuel\\Dropbox\\Phd Education\\Projects\\project 1 accelerated DA and DWP SDE\\code\\utilities\\posteriorinference.jl")
else
    include("C:\\Users\\samue\\OneDrive\\Documents\\GitHub\\adamcmcpaper\\utilities\\posteriorinference.jl")
end

# set parameters
nbr_iterations = 2000
nbr_particels = 25
nbr_of_cores= 4
burn_in = 1
sim_data = true
set_nbr_params = 2 # should be 7
log_scale_prior = false
beta_MH = 0.1
mcmc_alg = "MCWM"  # set MCWM or PMCMC

data_set = "old"
dt = 0.035 # new = 0.5 old = 0.03
dt_U = 1. # new = 1 old = 1

length_training_data = 5000


accelerated_da = false

load_tranining_data = true


# set parameters
burn_in = 10000 # this should be 2000 when estimating 2 parameters

if set_nbr_params == 2
	burn_in = 1000 # this should be 2000
	length_training_data = 2000
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

  @load "gp_training_2_par_training_and_test_data_test_new_code.jld"


  #@load "gp_training_$(set_nbr_params)_par.jld"
  #@load "gp_training_$(set_nbr_params)_par.jld"
  #@load "gp_training_$(set_nbr_params)_par_training_and_test_data.jld"

  #=
  if set_nbr_params == 7
    @load "gp_training_7_par_training_and_test_data_multiple_cores_fix_logsumexp.jld"
  else
    @load "gp_training_$(set_nbr_params)_par_training_and_test_data.jld"
    #@load "gp_training_7_par_training_and_test_data_multiple_cores_fix_logsumexp.jld"
  end
  =#


end


################################################################################
###            plot training data                                            ###
################################################################################

#export_data(problem_training, res_training[1],"dagpMCMC_training_data"*jobname)


using PyPlot

text_size = 15

PyPlot.figure(figsize=(10,20))

ax1 = PyPlot.subplot(711)
PyPlot.plot(res_training[1].Theta_est[1,:])
PyPlot.plot(problem.model_param.theta_true[1]*ones(size(res_training[1].Theta_est)[2]), "k")
PyPlot.ylabel(L"$\log \kappa$",fontsize=text_size)
ax1[:axes][:get_xaxis]()[:set_ticks]([])

ax2 = PyPlot.subplot(712)
PyPlot.plot(res_training[1].Theta_est[2,:])
PyPlot.plot(problem.model_param.theta_true[2]*ones(size(res_training[1].Theta_est)[2]), "k")
PyPlot.ylabel(L"$\log \gamma$",fontsize=text_size)
ax2[:axes][:get_xaxis]()[:set_ticks]([])

PyPlot.subplot(713)
PyPlot.plot(res_training[1].Theta_est[3,:])
PyPlot.plot(problem.model_param.theta_true[3]*ones(size(res_training[1].Theta_est)[2]), "k")
PyPlot.ylabel(L"$\log c$",fontsize=text_size)
PyPlot.subplot(714)
PyPlot.plot(res_training[1].Theta_est[4,:])
PyPlot.plot(problem.model_param.theta_true[4]*ones(size(res_training[1].Theta_est)[2]), "k")
PyPlot.ylabel(L"$\log d$",fontsize=text_size)
PyPlot.subplot(715)
PyPlot.plot(res_training[1].Theta_est[5,:])
PyPlot.plot(problem.model_param.theta_true[5]*ones(size(res_training[1].Theta_est)[2]), "k")
PyPlot.ylabel(L"$\log p_1$",fontsize=text_size)
PyPlot.subplot(716)
PyPlot.plot(res_training[1].Theta_est[6,:])
PyPlot.plot(problem.model_param.theta_true[6]*ones(size(res_training[1].Theta_est)[2]), "k")
PyPlot.ylabel(L"$\log p_2$",fontsize=text_size)
PyPlot.subplot(717)
PyPlot.plot(res_training[1].Theta_est[7,:])
PyPlot.plot(problem.model_param.theta_true[7]*ones(size(res_training[1].Theta_est)[2]), "k")
PyPlot.ylabel(L"$\log \sigma$",fontsize=text_size)
PyPlot.xlabel("Iteration",fontsize=text_size)



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

  perc_outlier = 0.02
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

StatPlots.qqplot(Normal, residuals)

################################################################################
##  Compare assumption                                                                            ##
################################################################################


# estimate probabilities for assumptions

n = size(data_training,2)
n_burn_in = problem_training.alg_param.burn_in
dim = length(problem.model_param.theta_true)
data_signs = zeros(4,n)
data_signs[3,:] = data_training[dim+1,:]
data_signs[4,:] = res_training[1].loglik_est[end-n+1:end]

noisy_pred = problem.alg_param.noisy_est


for i = 1:n
  (loglik_est_star, var_pred_ml, prediction_sample_ml_star) = predict(data_training[1:dim,i],gp,noisy_pred)
  (loglik_est_old, var_pred_ml, prediction_sample_ml_old) = predict(res_training[1].Theta_est[:,i+n_burn_in],gp,noisy_pred)
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

include("run_pf_paralell.jl")
N = 25
Z = problem_training.data.Z

for i = 1:n
  (loglik_est_star, var_pred_ml, prediction_sample_ml_star) = predict(data_test[1:dim,i],gp,noisy_pred)
  (loglik_est_old, var_pred_ml, prediction_sample_ml_old) = predict(res_training[1].Theta_est[:,i+n_burn_in+size(data_training,2)],gp,noisy_pred)
  if prediction_sample_ml_old[1] > prediction_sample_ml_star[1]
    go_to_case_1 = rand() < prob_case_1
    if go_to_case_1

      nbr_case_1_selceted = nbr_case_1_selceted+1

      theta_old = res_training[1].Theta_est[:,i+n_burn_in+size(data_training,2)]
      (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta_old, problem.model_param.theta_known,length(theta_old))
      A = A*A_sign
      # set value for constant b function
      b_const = sqrt(2.*sigma^2 / 2.)
      (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
      (loglik_pf_old, 	~,  ~) = run_pf_paralell(Z,theta_old,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

      theta_new = res_training[1].Theta_est[:,i+n_burn_in+size(data_training,2)]
      (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta_new, problem.model_param.theta_known,length(theta_old))
      A = A*A_sign
      # set value for constant b function
      b_const = sqrt(2.*sigma^2 / 2.)
      (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
      (loglik_pf_new, 	~,  ~) = run_pf_paralell(Z,theta_new,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

      if loglik_pf_new > loglik_pf_old
        nbr_case_1_correct = nbr_case_1_correct+1

      end

    else
      nbr_case_3_selceted = nbr_case_3_selceted+1


      theta_old = res_training[1].Theta_est[:,i+n_burn_in+size(data_training,2)]
      (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta_old, problem.model_param.theta_known,length(theta_old))
      A = A*A_sign
      # set value for constant b function
      b_const = sqrt(2.*sigma^2 / 2.)
      (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
      (loglik_pf_old, 	~,  ~) = run_pf_paralell(Z,theta_old,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

      theta_new = res_training[1].Theta_est[:,i+n_burn_in+size(data_training,2)]
      (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta_new, problem.model_param.theta_known,length(theta_old))
      A = A*A_sign
      # set value for constant b function
      b_const = sqrt(2.*sigma^2 / 2.)
      (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
      (loglik_pf_new, 	~,  ~) = run_pf_paralell(Z,theta_new,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

      if loglik_pf_new < loglik_pf_old
        nbr_case_3_correct = nbr_case_3_correct+1

      end

    end

  else

    go_to_case_2 = rand() < prob_case_2

    if go_to_case_2

      nbr_case_2_selceted = nbr_case_2_selceted+1

      theta_old = res_training[1].Theta_est[:,i+n_burn_in+size(data_training,2)]
      (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta_old, problem.model_param.theta_known,length(theta_old))
      A = A*A_sign
      # set value for constant b function
      b_const = sqrt(2.*sigma^2 / 2.)
      (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
      (loglik_pf_old, 	~,  ~) = run_pf_paralell(Z,theta_old,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

      theta_new = res_training[1].Theta_est[:,i+n_burn_in+size(data_training,2)]
      (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta_new, problem.model_param.theta_known,length(theta_old))
      A = A*A_sign
      # set value for constant b function
      b_const = sqrt(2.*sigma^2 / 2.)
      (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
      (loglik_pf_new, 	~,  ~) = run_pf_paralell(Z,theta_new,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

      if loglik_pf_new < loglik_pf_old
        nbr_case_2_correct = nbr_case_2_correct+1
      end

    else
      nbr_case_4_selceted = nbr_case_4_selceted+1


      theta_old = res_training[1].Theta_est[:,i+n_burn_in+size(data_training,2)]
      (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta_old, problem.model_param.theta_known,length(theta_old))
      A = A*A_sign
      # set value for constant b function
      b_const = sqrt(2.*sigma^2 / 2.)
      (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
      (loglik_pf_old, 	~,  ~) = run_pf_paralell(Z,theta_old,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

      theta_new = res_training[1].Theta_est[:,i+n_burn_in+size(data_training,2)]
      (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta_new, problem.model_param.theta_known,length(theta_old))
      A = A*A_sign
      # set value for constant b function
      b_const = sqrt(2.*sigma^2 / 2.)
      (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (problem_training.alg_param.subsample_int, problem_training.alg_param.nbr_x0, problem_training.alg_param.nbr_x,N))
      (loglik_pf_new, 	~,  ~) = run_pf_paralell(Z,theta_new,problem.model_param.theta_known, N, N_calc, problem_training.alg_param.dt, problem_training.alg_param.dt_U, problem_training.alg_param.nbr_x0, nbr_x0_calc, problem_training.alg_param.nbr_x, nbr_x_calc, problem_training.alg_param.subsample_int, subsample_interval_calc, false, true, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

      if loglik_pf_new > loglik_pf_old
        nbr_case_4_correct = nbr_case_4_correct+1

      end

    end
  end
end

nbr_case_selceted = [nbr_case_1_selceted nbr_case_2_selceted nbr_case_3_selceted nbr_case_4_selceted]

nbr_case_correct = [nbr_case_1_correct nbr_case_2_correct nbr_case_3_correct nbr_case_4_correct]

prob_correct_given_selected = nbr_case_correct./nbr_case_selceted
