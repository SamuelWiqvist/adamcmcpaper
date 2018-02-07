# Script for running the PMCMC algorithm

include("set_up.jl")

# set parameters
nbr_iterations = 2000
nbr_particels = 25
nbr_of_cores= 4
burn_in = 1
sim_data = true
set_nbr_params = 7
log_scale_prior = false
beta_MH = 0.1
mcmc_alg = "MCWM"  # set MCWM or PMCMC

data_set = "old"
dt = 0.035 # new = 0.5 old = 0.03
dt_U = 1. # new = 1 old = 1

length_training_data = 1000


accelerated_da = true

jobname = "test_ada_gp_mcmc"



################################################################################
##                         set GP problem                               ##
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




################################################################################
###            plot training data                                            ###
################################################################################

export_data(problem_training, res_training[1],"dagpMCMC_training_data")



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
##               Set std_limit                                               ###
################################################################################

theta_gp_test_m = res_training[1].Theta_est[:,problem_training.alg_param.burn_in:end]
noisy_est = problem.alg_param.noisy_est

(mean_pred_ml_vec, std_pred_ml_vec, prediction_sample_ml)  = predict(theta_gp_test_m,gp,noisy_est)

problem.alg_param.std_limit = percentile(std_pred_ml_vec, 50)



problem.alg_param.std_limit = 100
################################################################################
#                Estimate probabilities
################################################################################


n = size(data_training,2)
n_burn_in = problem_training.alg_param.burn_in
dim = length(problem.model_param.theta_true)
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

################################################################################
##               Run A-DA-GP-MCMC                                              ###
################################################################################

# start at mode (median)!
problem.model_param.theta_0 = mean(theta_training,2)


if !log_scale_prior
  # run adaptive PMCMC
  res = adagpMCMC(problem_training, problem, gp, cov_matrix,prob_cases)

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


#=
# run adaptive PMCMC
res2 = @time gpPMCMC(problem2)
res5 = @time gpPMCMC(problem5)

mcmc_results = Result(res[1].Theta_est, res[1].loglik_est, res[1].accept_vec, res[1].prior_vec)

# plot results
export_data(problem5, mcmc_results)
export_parameters(mcmc_results[2])

# est 2 parameters

problem = set_up_gp_problem(nbr_of_unknown_parameters=2)
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.adaptive_update =  AMUpdate_gen(eye(2), 1/sqrt(2), 0.15, 1, 0.8, 25) # was 0.3


problem = set_up_gp_problem(nbr_of_unknown_parameters=2, prior_dist="nonlog")
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.adaptive_update =  AMUpdate_gen(eye(2), 1/sqrt(2), 0.2, 1, 0.8, 25)



# est 3 parameters

problem = set_up_gp_problem(nbr_of_unknown_parameters=3)
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.adaptive_update =  AMUpdate_gen(eye(3), 1/sqrt(3), 0.2, 1, 0.8, 25)

problem = set_up_gp_problem(nbr_of_unknown_parameters=3, prior_dist="nonlog")
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.adaptive_update =  AMUpdate_gen(eye(3), 1/sqrt(3), 0.2, 1, 0.8, 25)


# est 4 parameters


problem = set_up_gp_problem(nbr_of_unknown_parameters=4)
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.alg_param.burn_in = burn_in
problem.adaptive_update =  AMUpdate_gen(eye(4), 1/sqrt(4), 0.2, 1, 0.7, 25)


problem = set_up_gp_problem(nbr_of_unknown_parameters=4, prior_dist="nonlog")
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.adaptive_update =  AMUpdate_gen(eye(4), 1/sqrt(4), 0.2, 1, 0.8, 25)



# est 5 parameters

problem = set_up_gp_problem(nbr_of_unknown_parameters=5)
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.adaptive_update =  AMUpdate_gen(eye(5), 1/sqrt(5), 0.3, 1, 0.8, 25)


problem = set_up_gp_problem(nbr_of_unknown_parameters=5, prior_dist="nonlog")
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.adaptive_update =  AMUpdate_gen(eye(5), 1/sqrt(5), 0.2, 1, 0.8, 25)

# est 6 parameters


problem = set_up_gp_problem(nbr_of_unknown_parameters=6)
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.adaptive_update =  AMUpdate_gen(eye(6), 1/sqrt(6), 0.2, 1, 0.7, 25)

problem = set_up_gp_problem(nbr_of_unknown_parameters=6, prior_dist="nonlog")
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.adaptive_update =  AMUpdate_gen(eye(6), 1/sqrt(6), 0.2, 1, 0.7, 25)


# est 7 parameters


problem = set_up_gp_problem(nbr_of_unknown_parameters=7)
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.adaptive_update =  AMUpdate_gen(eye(7), 1/sqrt(7), 0.2, 1, 0.7, 25)


problem = set_up_gp_problem(nbr_of_unknown_parameters=7, prior_dist="nonlog")
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.adaptive_update =  AMUpdate_gen(eye(7), 1/sqrt(7), 0.2, 1, 0.7, 25)


# est 8 parameters


problem = set_up_gp_problem(nbr_of_unknown_parameters=8)
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.adaptive_update =  AMUpdate_gen(eye(8), 1/sqrt(8), 0.2, 1, 0.7, 25)


problem = set_up_gp_problem(nbr_of_unknown_parameters=8, prior_dist="nonlog")
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.adaptive_update =  AMUpdate_gen(eye(8), 1/sqrt(8), 0.2, 1, 0.7, 25)



# est 9 parameters

problem = set_up_gp_problem (nbr_of_unknown_parameters=9)
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.adaptive_update =  AMUpdate_gen(eye(9), 1/sqrt(9), 0.2, 1, 0.7, 25)

problem = set_up_gp_problem (nbr_of_unknown_parameters=9,prior_dist="nonlog")
problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.alg_param.nbr_of_cores = nbr_of_cores
problem.adaptive_update =  AMUpdate_gen(eye(9), 1/sqrt(9), 0.2, 1, 0.7, 25)

################################################################################
###                        set algorithm parameters                          ###
################################################################################

problem.alg_param.alg = "MCWM"
problem.alg_param.compare_GP_and_PF = false
problem.alg_param.noisy_est = false
problem.alg_param.pred_method = "sample"
problem.alg_param.R = 8000
problem.alg_param.burn_in = 1000
problem.alg_param.length_training_data = 2000
problem.alg_param.nbr_predictions = 1
#problem.alg_param.print_interval = 500
problem.alg_param.selection_method = "max_loglik"  # "local_loglik_approx" # "max_loglik"

# run adaptive PMCMC
res2 = @time gpPMCMC(problem2)
res5 = @time gpPMCMC(problem5)

mcmc_results = Result(res[1].Theta_est, res[1].loglik_est, res[1].accept_vec, res[1].prior_vec)

# plot results
export_data(problem5, mcmc_results)
export_parameters(mcmc_results[2])

# analyse accaptance rates ¨¨
a_prob_gp = exp(res[4][1,:])
a_prob_pf = exp(res[4][2,:])

for i = 1:length(a_prob_gp)
  if a_prob_gp[i] > 1
    a_prob_gp[i] = 1
  end
  if a_prob_pf[i] > 1
    a_prob_pf[i] = 1
  end
end


bins = 100
PyPlot.figure()
h1 = PyPlot.plt[:hist](a_prob_gp,bins)
PyPlot.figure()
h2 = PyPlot.plt[:hist](a_prob_pf,bins)



# analyse gpResults

res[1].nbr_early_rejections


eqzeo(x) = x == 0
eqone(x) = x == 1
eqminusone(x) = x == -1
accpetance_rate_preER = sum(res[1].accept_vec[(problem.alg_param.burn_in+1):(problem.alg_param.burn_in + problem.alg_param.length_training_data)])/(problem.alg_param.length_training_data)
accpetance_rate_ER = sum(res[1].accept_vec[end-size(res[1].compare_GP_PF,2):end])/size(res[1].compare_GP_PF,2)
proc_same_GP_PF = length(find(eqzeo, res[1].compare_GP_PF[1,:] - res[1].compare_GP_PF[2,:]))/size(res[1].compare_GP_PF,2)
percentage_wrong_accept = length(find(eqone, res[1].compare_GP_PF[1,:] - res[1].compare_GP_PF[2,:]))/size(res[1].compare_GP_PF,2)
percentage_wrong_reject = length(find(eqminusone, res[1].compare_GP_PF[1,:] - res[1].compare_GP_PF[2,:]))/size(res[1].compare_GP_PF,2)



PyPlot.figure()
PyPlot.scatter3D(res[1].data_gp_pf[1,:],res[1].data_gp_pf[2,:],res[1].data_gp_pf[4,:], color = "red")
PyPlot.hold(true)
PyPlot.scatter3D(res[1].data_gp_pf[1,:],res[1].data_gp_pf[2,:],res[1].data_gp_pf[3,:], color = "blue")
PyPlot.title("PF (red), GP (blue)")

PyPlot.figure()
PyPlot.plot(res[1].data_gp_pf[3,:],res[1].data_gp_pf[4,:], "*")
PyPlot.xlabel("GP")
PyPlot.ylabel("PF")



RMSE(res[1].data_gp_pf[4,:],res[1].data_gp_pf[3,:])

# compare variance for gp and pf

y = problem.data.y
theta_star = problem.model_param.theta_true
theta_known = problem.model_param.theta_known
N = 500
print_on = false

noisy_pred = problem.alg_param.noisy_est
gp = res[3]

nbr_itr_pf_gp = 1000
loglik_vec_pf = zeros(nbr_itr_pf_gp)
loglik_vec_gp = zeros(nbr_itr_pf_gp)

for i = 1:nbr_itr_pf_gp
  if mod(i,100) == 0
    println(i)
  end
  loglik_vec_pf[i] = pf(y, theta_star,theta_known,N,print_on)
end

for i = 1:nbr_itr_pf_gp
  if mod(i,100) == 0
    println(i)
  end
  (mean_pred_ml, var_pred_ml, prediction_sample_ml) = predict(theta_star,gp,noisy_pred)
  loglik_vec_gp[i] = prediction_sample_ml[1]
end

std(loglik_vec_pf)
std(loglik_vec_gp)

bins = 50
PyPlot.figure()
h1 = PyPlot.plt[:hist](loglik_vec_pf,bins)
PyPlot.figure()
h2 = PyPlot.plt[:hist](loglik_vec_gp,bins)
=#
