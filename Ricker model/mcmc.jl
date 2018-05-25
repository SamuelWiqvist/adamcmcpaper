# This file contains the functions related to the PMCMC, MCWM and (A)DA-GP-MCMC algorithms for the Ricker model

# load particle filter
include("pf.jl")


################################################################################
###                  algorithms                                            #####
################################################################################

# PMCMC/MCWM

doc"""
    MCMC(problem::Problem, store_data::Bool=false, return_cov_matrix::Bool=false) # this function should be merged with the generate_training_test_data function!

Runs PMCMC or MCWM for the Ricker model.
"""
function mcmc(problem::Problem, store_data::Bool=false, return_cov_matrix::Bool=false)

  # data
  y = problem.data.y

  # algorithm parameters
  R = problem.alg_param.R # number of iterations
  N = problem.alg_param.N # number of particels
  burn_in = problem.alg_param.burn_in # burn in
  alg = problem.alg_param.alg # use PMCMC or MCWM
  pf_alg = problem.alg_param.pf_alg # pf algorithm bootsrap of apf
  print_interval = problem.alg_param.print_interval # print accaptance rate and covarince function ever print_interval:th iteration

  # model parameters
  theta_true = problem.model_param.theta_true # [log(r) log(phi) log(sigma)]
  theta_known = problem.model_param.theta_known # NaN
  theta_0 = problem.model_param.theta_0 # [log(r_0) log(phi_0) log(sigma_0)]

  # parameters for prior dist
  dist_type = problem.prior_dist.dist
  prior_parameters = problem.prior_dist.prior_parameters

  # pre-allocate matricies and vectors and variables
  Theta = zeros(length(theta_0),R)
  loglik = zeros(R)
  accept_vec = zeros(R)
  prior_vec = zeros(R)
  theta_star = zeros(length(theta_0),1)
  loglik_star = zeros(Float64)
  a_log = zero(Float64)
  nbr_eval_pf = 0

  # pre-allocate matricies and vectors for storing data
  if store_data
    Theta_star_store = zeros(length(theta_0),R-burn_in)
    loglik_star_store = zeros(R-burn_in)
    Theta_old_store = zeros(length(theta_0),R-burn_in)
    loglik_old_store = zeros(R-burn_in)
  end

  # parameters for adaptive update
  adaptive_update_params = set_adaptive_alg_params(problem.adaptive_update, length(theta_0),Theta[:,1], R)


  # print information at start of algorithm
  @printf "Starting MCMC with adaptive RW estimating %d parameters\n" length(theta_true)
  @printf "MCMC algorithm: %s\n" alg
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist
  @printf "Particel filter: %s\n" pf_alg

  # first iteration
  @printf "Iteration: %d\n" 1
  @printf "Covariance:\n"
  print_covariance(problem.adaptive_update,adaptive_update_params, 1)

  Theta[:,1] = theta_0

  loglik[1] = pf(y, Theta[:,1],theta_known,N,true)

  # print start loglik
  @printf "Loglik: %.4f \n" loglik[1]

  for r = 2:R

    # set print_on to false, only print each print_interval:th iteration
    print_on = false

    # print acceptance rate for the last print_interval iterations
    if mod(r-1,print_interval) == 0
      print_on = true # print ESS and Nbr resample each print_interval:th iteration
      # print progress
      @printf "Percentage done: %.2f %% \n" 100*(r-1)/R
      # print accaptace rate
      @printf "Acceptance rate on iteration %d to %d is %.4f\n" r-print_interval r-1  sum(accept_vec[r-print_interval:r-1])/( r-1 - (r-print_interval) )
      # print covaraince function
      @printf "Covariance:\n"
      print_covariance(problem.adaptive_update,adaptive_update_params, r)
      # print loglik
      @printf "Loglik: %.4f \n" loglik[r-1]
    end

    # Gaussian random walk
    (theta_star, ) = gaussian_random_walk(problem.adaptive_update, adaptive_update_params, Theta[:,r-1], r)

    # calc loglik using proposal
    loglik_star = pf(y, theta_star,theta_known, N,print_on)

    prior_log_star = evaluate_prior(theta_star,prior_parameters, problem.prior_dist.dist)
    prior_log_old = evaluate_prior(Theta[:,r-1],prior_parameters, problem.prior_dist.dist)

    jacobian_log_star = jacobian(theta_star)
    jacobian_log_old = jacobian(Theta[:,r-1])

    if alg == "MCWM"
      a_log = loglik_star + prior_log_star +  jacobian_log_star - (pf(y, Theta[:,r-1],theta_known,N,print_on) +  prior_log_old + jacobian_log_old)
    else
      # calc accaptace probability for the PMCMC algorithm
      a_log = loglik_star + prior_log_star +  jacobian_log_star - (loglik[r-1] +  prior_log_old + jacobian_log_old)
    end

    nbr_eval_pf += 1

    accept = log(rand()) < a_log # calc accaptace decision

    if store_data && r > burn_in # store data
      Theta_star_store[:,r-burn_in] = theta_star
      loglik_star_store[r-burn_in] = loglik_star
      Theta_old_store[:,r-burn_in]  = Theta[:,r-1]
      loglik_old_store[r-burn_in] = loglik[r-1]
    end

    if accept # the proposal is accapted
      Theta[:,r] = theta_star # update chain with new proposals
      loglik[r] = loglik_star
      accept_vec[r] = 1
    else
      Theta[:,r] = Theta[:,r-1] # keep old values
      loglik[r] = loglik[r-1]
    end

    # adaptation of covaraince matrix for the proposal distribution
    adaptation(problem.adaptive_update, adaptive_update_params, Theta, r,a_log)

  end

  @printf "Ending MCMC with adaptive RW estimating %d parameters\n" length(theta_true)
  @printf "Algorithm: %s\n" alg
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist
  @printf "Particel filter: %s\n" pf_alg
  @printf "Total number of evaluations of the particle filter: %d\n" nbr_eval_pf

  # return results
  if store_data && return_cov_matrix
    cov_prop_kernel = get_covariance(problem.adaptive_update,adaptive_update_params, R)
    return return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params), Theta_star_store, loglik_star_store,Theta_old_store, loglik_old_store, cov_prop_kernel
  elseif store_data
    return return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params), Theta_star_store, loglik_star_store,Theta_old_store, loglik_old_store

  else
    return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params)
  end


end

# DA-GP-MCMC

doc"""
    dagpmcmc(problem_traning::Problem, problem::gpProblem, gp::GPModel, cov_matrix::Matrix)

Runs the DA-GP-MCMC algorithm for the Ricker model.
"""
function dagpmcmc(problem_traning::Problem, problem::gpProblem, gp::GPModel, cov_matrix::Matrix, return_run_info::Bool=false)

  # data
  y = problem.data.y

  # algorithm parameters
  R = problem.alg_param.R # number of iterations
  N = problem.alg_param.N # number of particels
  burn_in = problem.alg_param.burn_in # burn in
  length_training_data = problem.alg_param.length_training_data # length of training data
  independet_sampling = problem.alg_param.independet_sampling # length of training data
  compare_GP_and_PF = problem.alg_param.compare_GP_and_PF # compare GP and PF if true
  noisy_est = problem.alg_param.noisy_est # use noisy estimation
  pred_method = problem.alg_param.pred_method # method used for predictions
  est_method = problem.alg_param.est_method # methods used to estimate the parameters of the GP model
  pf_alg = problem.alg_param.pf_alg # pf algorithm
  alg = problem.alg_param.alg # use PMCMC or MCWM
  print_interval = problem.alg_param.print_interval # print the accaptance rate every print_interval:th iteration
  lasso = problem.alg_param.lasso # use Lasso
  beta_MH = problem.alg_param.beta_MH

  # model parameters
  theta_true = problem.model_param.theta_true # [log(r) log(phi) log(sigma)]
  theta_known = problem.model_param.theta_known # NaN
  theta_0 = problem.model_param.theta_0 # [log(r_0) log(phi_0) log(sigma)]

  # pre-allocate matricies and vectors
  Theta = zeros(length(theta_0), R)
  loglik = zeros(R)
  accept_vec = zeros(R)
  prior_vec = zeros(R)
  theta_star = zeros(length(theta_0))
  #compare_GP_PF = zeros(2,R-length_training_data-burn_in)
  compare_GP_PF = zeros(2,R)
  data_gp_pf = zeros(length(theta_0)+2,R-length_training_data-burn_in)
  data_training = zeros(1+length(theta_0), length_training_data)
  accept_prob_log = zeros(2, R) # [gp ; pf]
  kernel_MH_direct = problem.adaptive_update


  loglik_star = zero(Float64)
  loglik_gp = zeros(Float64)
  loglik_gp_old = zero(Float64)
  loglik_gp_new = zero(Float64)
  index_keep_gp_er = zero(Int64)
  nbr_early_rejections = zero(Int64)
  accept_gp = true
  accept = true
  MH_direct = false
  nbr_ordinary_mh = 0
  nbr_ordinary_mh_accapte = 0
  nbr_run_DA = 0
  nbr_second_stage_accepted = 0
  nbr_second_stage = 0
  nbr_eval_pf = 0
  nbr_eval_pf_secound_stage = 0 

  # starting values for times:
  time_pre_er = 0
  time_fit_gp = 0
  time_er_part = 0

  # parameters for prior dist
  dist_type = problem.prior_dist.dist
  prior_parameters = problem.prior_dist.prior_parameters

  # da new
  loglik_gp_new_std = 0

  # set kernels

  # non-adaptive RW for both kernels

  # prop kernl for DA-GP-MCMC
  xi = 1.1
  problem.adaptive_update = noAdaptation(xi^2*cov_matrix)

  adaptive_update_params = set_adaptive_alg_params(problem.adaptive_update, length(theta_0),Theta[:,1], R)

  # prop kernl for secound_stage_direct
  kernel_MH_direct = noAdaptation(cov_matrix)
  adaptive_update_params_MH_direct = set_adaptive_alg_params(kernel_MH_direct, length(theta_0),Theta[:,1], R)

  @printf "Starting DA-GP-MCMC with adaptive RW estimating %d parameters\n" length(theta_true)
  @printf "Covariance - kernel_MH_direct:\n"
  print_covariance(kernel_MH_direct,adaptive_update_params_MH_direct, 1)


  # print information at start of algorithm
  @printf "MCMC algorithm: %s\n" alg
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist
  @printf "Particel filter: %s\n" pf_alg

  # first iteration
  @printf "Iteration: %d\n" 1 # print first iteration
  @printf "Covariance:\n"
  print_covariance(problem.adaptive_update,adaptive_update_params, 1)

  # first iteration
  Theta[:,1] = theta_0
  loglik[1]  = pf(y, Theta[:,1],theta_known,N,true)

  if alg == "MCWM"
    # do nothing
  else
    loglik_gp_old = predict(Theta[:,1], gp, pred_method,est_method,noisy_est)[1]
  end

  # print start loglik
  @printf "Loglik: %.4f \n" loglik[1]

  tic()

  for r = 2:R

    # set print_on to false, only print each print_interval:th iteration
    print_on = false

    # print acceptance rate for the last print_interval iterations
    if mod(r-1,print_interval) == 0
      print_on = true # print ESS and Nbr resample each print_interval:th iteration
      # print progress
      @printf "Percentage done: %.2f %% \n" 100*(r-1)/R
      # print accaptance probability
      @printf "Acceptance rate on iteration %d to %d is %.4f %%\n" r-print_interval r-1  sum(accept_vec[r-print_interval:r-1])/( r-1 - (r-print_interval) )*100
      # print covariance matrix
      @printf "Covariance:\n"
      print_covariance(problem.adaptive_update,adaptive_update_params, r)
      # print loglik
      @printf "Loglik: %.4f \n" loglik[r-1]
    end

    MH_direct = rand() < beta_MH # we always run the early-rejection scheme

    if MH_direct

      # secound stage direct

      nbr_ordinary_mh +=  1

      # Gaussian random walk using secound stage direct kernel
      (theta_star, ) = gaussian_random_walk(kernel_MH_direct, adaptive_update_params_MH_direct, Theta[:,r-1], r)

      # calc loglik using proposed parameters
      loglik_star = pf(y, theta_star,theta_known,N,print_on)

      prior_log_star = evaluate_prior(theta_star,prior_parameters, problem.prior_dist.dist)
      prior_log_old = evaluate_prior(Theta[:,r-1],prior_parameters, problem.prior_dist.dist)

      jacobian_log_star = jacobian(theta_star)
      jacobian_log_old = jacobian(Theta[:,r-1])

      if alg == "MCWM"
        a_log = loglik_star + prior_log_star +  jacobian_log_star - (pf(y, Theta[:,r-1],theta_known,N,print_on) +  prior_log_old + jacobian_log_old)
      else
        # calc accaptace probability for the PMCMC algorithm
        a_log = loglik_star + prior_log_star +  jacobian_log_star - (loglik[r-1] +  prior_log_old + jacobian_log_old)
      end

      nbr_eval_pf += 1


      accept = log(rand()) < a_log

      if accept # the proposal is accapted
        nbr_ordinary_mh_accapte +=  1
        Theta[:,r] = theta_star # update chain with new values
        loglik[r] = loglik_star
        accept_vec[r] = 1
      else
        Theta[:,r] = Theta[:,r-1] # keep old values
        loglik[r] = loglik[r-1]
      end


    else

      # first stage
      nbr_run_DA +=  1

      # set proposal
      (theta_star, ) = gaussian_random_walk(problem.adaptive_update, adaptive_update_params, Theta[:,r-1], r)

      (loglik_gp_pred,loglik_gp_new_std) = predict(theta_star, gp, pred_method,est_method,noisy_est,true)

      loglik_gp_new = loglik_gp_pred[1]
      loglik_gp_new_std = loglik_gp_new_std[1]

      prior_log_star = evaluate_prior(theta_star,prior_parameters, problem.prior_dist.dist)
      prior_log_old = evaluate_prior(Theta[:,r-1],prior_parameters, problem.prior_dist.dist)

      jacobian_log_star = jacobian(theta_star)
      jacobian_log_old = jacobian(Theta[:,r-1])

      if alg == "MCWM"
        loglik_gp_old = predict(Theta[:,r-1], gp, pred_method,est_method,noisy_est)[1]
        a_gp = loglik_gp_new + prior_log_star +  jacobian_log_star -  (loglik_gp_old - prior_log_old - jacobian_log_old)
      else
        a_gp = loglik_gp_new + prior_log_star +  jacobian_log_star -  (loglik_gp_old - prior_log_old - jacobian_log_old)
      end

      accept = log(rand()) < a_gp # calc accept

      # store accaptance probability
      accept_prob_log[1, r] = a_gp


      if !accept
        # keep old values
        nbr_early_rejections += 1
        Theta[:,r] = Theta[:,r-1]
        loglik[r] = loglik[r-1]
        # adaptation of covaraince matrix for the proposal distribution
        # adaptation(problem.adaptive_update, adaptive_update_params, Theta, r,a_gp)
      else

        # secound stage

        nbr_second_stage += 1

        loglik_star = pf(y, theta_star,theta_known,N,print_on)

        if alg == "MCWM"
          a_log = (loglik_star + loglik_gp_old)  -  (pf(y, Theta[:,r-1],theta_known,N,print_on) + loglik_gp_new)
        else
          a_log = (loglik_star + loglik_gp_old)  -  (loglik[r-1]  + loglik_gp_new)
        end

        nbr_eval_pf += 1
        nbr_eval_pf_secound_stage += 1

        accept = log(rand()) < a_log # calc accaptance decision
        accept_prob_log[2, r] = a_log # store data

        if accept # the proposal is accapted
          nbr_second_stage_accepted += 1
          Theta[:,r] = theta_star # update chain with proposal
          loglik[r] = loglik_star
          accept_vec[r] = 1
          if alg == "MCWM"
            # do nothing
          else
            loglik_gp_old = loglik_gp_new
          end
        else
          Theta[:,r] = Theta[:,r-1] # keep old values
          loglik[r] = loglik[r-1]
        end
      end
    end
  end

  time_da_part = toc()
  times = [time_pre_er time_fit_gp time_da_part]

  @printf "Ending DA-GP-MCMC with adaptive RW estimating %d parameters\n" length(theta_true)
  @printf "Algorithm: %s\n" alg
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist
  @printf "Particel filter: %s\n" pf_alg
  @printf "Time pre-er:  %.0f\n" time_pre_er
  @printf "Time fit GP model:  %.0f\n" time_fit_gp
  @printf "Time er-part:  %.0f\n" time_da_part

  @printf "Number early-rejections: %.d\n"  nbr_early_rejections

  @printf "Number of cases directly to ordinary MH: %d\n"  nbr_ordinary_mh
  @printf "Number of cases directly to ordinary MH accapted: %d\n"  nbr_ordinary_mh_accapte

  @printf "Number cases in secound stage: %d\n"  nbr_second_stage
  @printf "Number accepted in secound stage: %d\n"  nbr_second_stage_accepted

  @printf "Total number of evaluations of the particle filter: %d\n" nbr_eval_pf

  @printf "Acceptance rate for ordinary MH accapted: %.4f\n"  nbr_ordinary_mh_accapte/nbr_ordinary_mh*100
  @printf "Acceptance rate a_1: %.4f\n"  nbr_second_stage/nbr_run_DA*100
  @printf "Acceptance rate a_2: %.4f\n"  nbr_second_stage_accepted/nbr_second_stage*100
  @printf "Acceptance rate a_1*a_2: %.4f\n"  (nbr_second_stage/nbr_run_DA)*(nbr_second_stage_accepted/nbr_second_stage)*100
  @printf "Acceptance rate a (for entier algorithm): %.4f\n" sum(accept_vec)/R*100

  assumption_list = []
  loglik_list = []

  # return resutls
  if return_run_info
    run_info = [nbr_eval_pf;
                nbr_eval_pf_secound_stage;
                nbr_second_stage;
                nbr_ordinary_mh]
    return return_da_results(gp, Theta,loglik,accept_vec,prior_vec, compare_GP_PF, data_gp_pf,nbr_early_rejections, problem, adaptive_update_params,accept_prob_log,times),
            run_info
  else
    return return_da_results(gp, Theta,loglik,accept_vec,prior_vec, compare_GP_PF, data_gp_pf,nbr_early_rejections, problem, adaptive_update_params,accept_prob_log,times)
  end

end


# ADA-GP-MCMC

doc"""
    adagpmcmc(problem_traning::Problem, problem::gpProblem, gp::GPModel, casemodel::CaseModel, cov_matrix::Matrix)

Runs the ADA-GP-MCMC algorithm for the Ricker model.
"""
function adagpmcmc(problem_traning::Problem, problem::gpProblem, gp::GPModel, casemodel::CaseModel, cov_matrix::Matrix, return_run_info::Bool = false)

  # data
  y = problem.data.y

  # algorithm parameters
  R = problem.alg_param.R # number of iterations
  N = problem.alg_param.N # number of particels
  burn_in = problem.alg_param.burn_in # burn in
  length_training_data = problem.alg_param.length_training_data # length of training data
  independet_sampling = problem.alg_param.independet_sampling # length of training data
  compare_GP_and_PF = problem.alg_param.compare_GP_and_PF # compare GP and PF if true
  noisy_est = problem.alg_param.noisy_est # use noisy estimation
  pred_method = problem.alg_param.pred_method # method used for predictions
  est_method = problem.alg_param.est_method # methods used to estimate the parameters of the GP model
  pf_alg = problem.alg_param.pf_alg # pf algorithm
  alg = problem.alg_param.alg # use PMCMC or MCWM
  print_interval = problem.alg_param.print_interval # print the accaptance rate every print_interval:th iteration
  lasso = problem.alg_param.lasso # use Lasso
  beta_MH = problem.alg_param.beta_MH

  # model parameters
  theta_true = problem.model_param.theta_true # [log(r) log(phi) log(sigma)]
  theta_known = problem.model_param.theta_known # NaN
  theta_0 = problem.model_param.theta_0 # [log(r_0) log(phi_0) log(sigma)]

  # pre-allocate matricies and vectors
  Theta = zeros(length(theta_0), R)
  loglik = zeros(R)
  accept_vec = zeros(R)
  prior_vec = zeros(R)
  theta_star = zeros(length(theta_0))
  #compare_GP_PF = zeros(2,R-length_training_data-burn_in)
  compare_GP_PF = zeros(2,R)
  data_gp_pf = zeros(length(theta_0)+2,R-length_training_data-burn_in)
  data_training = zeros(1+length(theta_0), length_training_data)
  accept_prob_log = zeros(2, R) # [gp ; pf]
  kernel_MH_direct = problem.adaptive_update

  loglik_current = zero(Float64)
  loglik_star = zero(Float64)
  loglik_gp = zeros(Float64)
  loglik_gp_old = zero(Float64)
  loglik_gp_new = zero(Float64)
  index_keep_gp_er = zero(Int64)
  nbr_early_rejections = zero(Int64)
  accept_gp = true
  accept = true
  MH_direct = false
  nbr_ordinary_mh = 0
  nbr_ordinary_mh_accapte = 0
  nbr_split_accaptance_region = 0
  nbr_split_accaptance_region_early_accept = 0
  nbr_split_accaptance_region_early_reject = 0
  nbr_run_DA = 0
  nbr_second_stage_accepted = 0
  nbr_second_stage = 0

  nbr_eval_pf = zero(Int64)
  nbr_case_13 = zero(Int64)
  nbr_case_24 = zero(Int64)

  # starting values for times:
  time_pre_er = zero(Float64)
  time_fit_gp = zero(Float64)
  time_er_part = zero(Float64)

  nbr_case_1 = zero(Int64)
  nbr_case_2 = zero(Int64)
  nbr_case_3 = zero(Int64)
  nbr_case_4 = zero(Int64)

  nbr_case_pf_1 = zero(Int64)
  nbr_case_pf_2 = zero(Int64)
  nbr_case_pf_3 = zero(Int64)
  nbr_case_pf_4 = zero(Int64)

  # parameters for prior dist
  dist_type = problem.prior_dist.dist
  prior_parameters = problem.prior_dist.prior_parameters

  # set kernels

  # non-adaptive RW for both kernels

  # prop kernl for DA-GP-MCMC
  xi = 1.1
  problem.adaptive_update = noAdaptation(xi^2*cov_matrix)

  adaptive_update_params = set_adaptive_alg_params(problem.adaptive_update, length(theta_0),Theta[:,1], R)

  # prop kernl for secound_stage_direct
  kernel_MH_direct = noAdaptation(cov_matrix)
  adaptive_update_params_MH_direct = set_adaptive_alg_params(kernel_MH_direct, length(theta_0),Theta[:,1], R)

  @printf "Covariance - kernel_secound_stage_direct:\n"
  print_covariance(kernel_MH_direct,adaptive_update_params_MH_direct, 1)


  # print information at start of algorithm
  @printf "Starting ADA-GP-MCMC with adaptive RW estimating %d parameters\n" length(theta_true)
  @printf "MCMC algorithm: %s\n" alg
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Select case model: %s\n" typeof(casemodel)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist
  @printf "Particel filter: %s\n" pf_alg

  # first iteration
  @printf "Iteration: %d\n" 1 # print first iteration
  @printf "Covariance:\n"
  print_covariance(problem.adaptive_update,adaptive_update_params, 1)

  # first iteration
  Theta[:,1] = theta_0
  loglik[1]  = pf(y, Theta[:,1],theta_known,N,true)

  if alg == "MCWM"
    # do nothing
  else
    loglik_gp_old = predict(Theta[:,1], gp, pred_method,est_method,noisy_est)[1]
  end

  # print start loglik
  @printf "Loglik: %.4f \n" loglik[1]

  tic()

  for r = 2:R

    # set print_on to false, only print each print_interval:th iteration
    print_on = false

    # print acceptance rate for the last print_interval iterations
    if mod(r-1,print_interval) == 0
      print_on = true # print ESS and Nbr resample each print_interval:th iteration
      # print progress
      @printf "Percentage done: %.2f %% \n" 100*(r-1)/R
      # print accaptance probability
      @printf "Acceptance rate on iteration %d to %d is %.4f %%\n" r-print_interval r-1  sum(accept_vec[r-print_interval:r-1])/( r-1 - (r-print_interval) )*100
      # print covariance matrix
      @printf "Covariance:\n"
      print_covariance(problem.adaptive_update,adaptive_update_params, r)
      # print loglik
      @printf "Loglik: %.4f \n" loglik[r-1]
    end

    MH_direct = rand() < beta_MH # we always run the early-rejection scheme

    if MH_direct

      # secound stage direct

      nbr_ordinary_mh +=  1

      # Gaussian random walk using secound stage direct kernel
      (theta_star, ) = gaussian_random_walk(kernel_MH_direct, adaptive_update_params_MH_direct, Theta[:,r-1], r)

      # calc loglik using proposed parameters
      loglik_star = pf(y, theta_star,theta_known,N,print_on)

      prior_log_star = evaluate_prior(theta_star,prior_parameters, problem.prior_dist.dist)
      prior_log_old = evaluate_prior(Theta[:,r-1],prior_parameters, problem.prior_dist.dist)

      jacobian_log_star = jacobian(theta_star)
      jacobian_log_old = jacobian(Theta[:,r-1])

      if alg == "MCWM"
        loglik_current = pf(y, Theta[:,r-1],theta_known,N,print_on)
      else
        # calc accaptace probability for the PMCMC algorithm
        loglik_current = loglik[r-1]
      end

      a_log = loglik_star + prior_log_star +  jacobian_log_star - (loglik_current +  prior_log_old + jacobian_log_old)

      nbr_eval_pf += 1

      accept = log(rand()) < a_log

      if accept # the proposal is accapted
        Theta[:,r] = theta_star # update chain with new values
        loglik[r] = loglik_star
        accept_vec[r] = 1
        nbr_ordinary_mh_accapte += 1
      else
        Theta[:,r] = Theta[:,r-1] # keep old values
        loglik[r] = loglik[r-1]
      end


    else

      # stage 1
      nbr_run_DA +=  1

      # set proposal
      (theta_star, ) = gaussian_random_walk(problem.adaptive_update, adaptive_update_params, Theta[:,r-1], r)

      (loglik_gp_pred,loglik_gp_new_std) = predict(theta_star, gp, pred_method,est_method,noisy_est,true)

      loglik_gp_new = loglik_gp_pred[1]
      loglik_gp_new_std = loglik_gp_new_std[1]


      prior_log_star = evaluate_prior(theta_star,prior_parameters, problem.prior_dist.dist)
      prior_log_old = evaluate_prior(Theta[:,r-1],prior_parameters, problem.prior_dist.dist)

      jacobian_log_star = jacobian(theta_star)
      jacobian_log_old = jacobian(Theta[:,r-1])

      # should we recompute the loglik_gp_old value here?
      # we currently recompute loglik_gp_old here!
      prediction_sample, sigma_pred = predict(Theta[:,r-1], gp, pred_method,est_method,noisy_est, true)
      loglik_gp_old = prediction_sample[1]
      loglik_gp_old_std = sigma_pred[1]

      a_gp = loglik_gp_new + prior_log_star +  jacobian_log_star -  (loglik_gp_old - prior_log_old - jacobian_log_old)

      accept = log(rand()) < a_gp # calc accept

      # store accaptance probability
      accept_prob_log[1, r] = a_gp


      if !accept
        # keep old values
        nbr_early_rejections += 1
        Theta[:,r] = Theta[:,r-1]
        loglik[r] = loglik[r-1]
      else

        # stage 2 usign A-DA

        nbr_second_stage = nbr_second_stage+1


        u_log_hat = log(rand())

        if loglik_gp_old < loglik_gp_new

          nbr_case_13 = nbr_case_13 + 1


          # select case 1 or 3
          if selectcase1or3(casemodel, theta_star, loglik_gp_new, loglik_gp_old) == 1

            # case 1

            nbr_case_1 = nbr_case_1 + 1

            if u_log_hat < loglik_gp_old - loglik_gp_new

              # early-accept
              nbr_split_accaptance_region_early_accept = nbr_split_accaptance_region_early_accept+1
              Theta[:,r] = theta_star # update chain with proposal
              loglik[r] = NaN
              accept_vec[r] = 1

            else

              # run ordinary stage 2

              loglik_star = pf(y, theta_star,theta_known,N,print_on)

              # calc accaptance probability using PF
              # can only run MCWM in this case
              loglik_old = pf(y, Theta[:,r-1],theta_known,N,print_on)
              a_log = (loglik_star + loglik_gp_old)  -  (loglik_old + loglik_gp_new)
              accept = u_log_hat < a_log # calc accaptance decision
              accept_prob_log[2, r] = a_log # store data

              nbr_eval_pf += 1
              nbr_case_pf_1 += 1

              if accept # the proposal is accapted
                nbr_second_stage_accepted = nbr_second_stage_accepted+1
                Theta[:,r] = theta_star # update chain with proposal
                loglik[r] = NaN
                accept_vec[r] = 1
              else
                Theta[:,r] = Theta[:,r-1] # keep old values
                loglik[r] = loglik[r-1]
              end
            end

          else

            # case 3
            nbr_case_3 = nbr_case_3 + 1

            if u_log_hat > loglik_gp_old - loglik_gp_new

              # early-reject
              nbr_split_accaptance_region_early_reject = nbr_split_accaptance_region_early_reject+1
              Theta[:,r] = Theta[:,r-1] # keep old values
              loglik[r] = loglik[r-1]
              #accept_vec[r] = 1

            else

              # run ordinary stage 2
              loglik_star = pf(y, theta_star,theta_known,N,print_on)

              # calc accaptance probability using PF
              # can only run MCWM in this case
              loglik_old = pf(y, Theta[:,r-1],theta_known,N,print_on)
              a_log = (loglik_star + loglik_gp_old)  -  (loglik_old + loglik_gp_new)
              accept = u_log_hat < a_log # calc accaptance decision
              accept_prob_log[2, r] = a_log # store data

              nbr_eval_pf += 1
              nbr_case_pf_3 += 1

              if accept # the proposal is accapted
                nbr_second_stage_accepted = nbr_second_stage_accepted+1
                Theta[:,r] = theta_star # update chain with proposal
                loglik[r] = NaN
                accept_vec[r] = 1
              else
                Theta[:,r] = Theta[:,r-1] # keep old values
                loglik[r] = loglik[r-1]
              end
            end
          end

        else

          # select case 2 or 4
          nbr_case_24 = nbr_case_24 + 1

          if selectcase2or4(casemodel, theta_star, loglik_gp_new, loglik_gp_old) == 1

            # case 2

            nbr_case_2 = nbr_case_2 + 1

            # run ordinary stage 2

            loglik_star = pf(y, theta_star,theta_known,N,print_on)

            # calc accaptance probability using PF
            # can only run MCWM in this case
            loglik_old = pf(y, Theta[:,r-1],theta_known,N,print_on)
            a_log = (loglik_star + loglik_gp_old)  -  (loglik_old + loglik_gp_new)
            accept = u_log_hat < a_log # calc accaptance decision
            accept_prob_log[2, r] = a_log # store data

            nbr_eval_pf += 1
            nbr_case_pf_2 += 1

            if accept # the proposal is accapted
              nbr_second_stage_accepted = nbr_second_stage_accepted+1
              Theta[:,r] = theta_star # update chain with proposal
              loglik[r] = NaN
              accept_vec[r] = 1
            else
              Theta[:,r] = Theta[:,r-1] # keep old values
              loglik[r] = loglik[r-1]
            end

          else

            # case 4
            nbr_case_4 = nbr_case_4 + 1

            # accept directly
            nbr_split_accaptance_region_early_accept = nbr_split_accaptance_region_early_accept+1

            Theta[:,r] = theta_star # update chain with proposal
            loglik[r] = NaN
            accept_vec[r] = 1

          end
        end
      end
    end
  end

  time_da_part = toc()
  times = [time_pre_er time_fit_gp time_da_part]

  @printf "Ending ADA-GP-MCMC with adaptive RW estimating %d parameters\n" length(theta_true)
  @printf "Algorithm: %s\n" alg
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist
  @printf "Particel filter: %s\n" pf_alg
  @printf "Time consumption:"
  @printf "Time pre-er:  %.0f\n" time_pre_er
  @printf "Time fit GP model:  %.0f\n" time_fit_gp
  @printf "Time er-part:  %.0f\n" time_da_part
  @printf "Number early-rejections: %d\n"  nbr_early_rejections

  @printf "Number of cases directly to ordinary MH: %d\n"  nbr_ordinary_mh
  @printf "Number of cases directly to ordinary MH accapted: %d\n"  nbr_ordinary_mh_accapte

  @printf "Number split accaptance region early-accept: %d\n"  nbr_split_accaptance_region_early_accept
  @printf "Number split accaptance region early-reject: %d\n"  nbr_split_accaptance_region_early_reject

  @printf "Number cases in secound stage: %d\n"  nbr_second_stage
  @printf "Number accepted in secound stage: %d\n"  nbr_second_stage_accepted

  @printf "Acceptance rate for ordinary MH accapted: %.4f\n"  nbr_ordinary_mh_accapte/nbr_ordinary_mh*100
  @printf "Acceptance rate a_1: %.4f\n"  nbr_second_stage/nbr_run_DA*100
  @printf "Acceptance rate a_2: %.4f\n"  nbr_second_stage_accepted/nbr_second_stage*100
  @printf "Acceptance rate a_1*a_2: %.4f\n"  (nbr_second_stage/nbr_run_DA)*(nbr_second_stage_accepted/nbr_second_stage)*100
  @printf "Acceptance rate a (for entier algorithm): %.4f\n" sum(accept_vec)/R*100

  @printf "Total number of evaluations of the particle filter: %d\n" nbr_eval_pf

  @printf "Number case 1 or 3: %d\n"  nbr_case_13
  @printf "Number case 2 or 4: %d\n"  nbr_case_24

  @printf "Number case 1: %d, %.4f %% of all cases\n"  nbr_case_1 nbr_case_1/nbr_case_13*100
  @printf "Number case 2: %d, %.4f %% of all cases\n"  nbr_case_2 nbr_case_2/nbr_case_24*100
  @printf "Number case 3: %d, %.4f %% of all cases\n"  nbr_case_3 nbr_case_3/nbr_case_13*100
  @printf "Number case 4: %d, %.4f %% of all cases\n"  nbr_case_4 nbr_case_4/nbr_case_24*100

  @printf "Number pf runs in case 1: %d, prob pf given case 1 %.4f %%\n"  nbr_case_pf_1 nbr_case_pf_1/nbr_case_1*100
  @printf "Number pf runs in case 2: %d, prob pf given case 2 %.4f %%\n"  nbr_case_pf_2 nbr_case_pf_2/nbr_case_2*100
  @printf "Number pf runs in case 3: %d, prob pf given case 3 %.4f %%\n"  nbr_case_pf_3 nbr_case_pf_3/nbr_case_3*100
  @printf "Number pf runs in case 4: %d, prob pf given case 4 %.4f %%\n"  nbr_case_pf_4 nbr_case_pf_4/nbr_case_4*100


  assumption_list = []
  loglik_list = []

  # return resutls
  if return_run_info
    run_info =  [nbr_eval_pf;
                nbr_ordinary_mh;
                nbr_case_13;
                nbr_case_24;
                nbr_case_1;
                nbr_case_2;
                nbr_case_3;
                nbr_case_4;
                nbr_case_pf_1;
                nbr_case_pf_2;
                nbr_case_pf_3;
                nbr_case_pf_4]
    return return_da_results(gp, Theta,loglik,accept_vec,prior_vec, compare_GP_PF, data_gp_pf,nbr_early_rejections, problem, adaptive_update_params,accept_prob_log,times), run_info
  else
    return return_da_results(gp, Theta,loglik,accept_vec,prior_vec, compare_GP_PF, data_gp_pf,nbr_early_rejections, problem, adaptive_update_params,accept_prob_log,times)
  end

end


################################################################################
######               help functions                                        #####
################################################################################

doc"""
    return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params)

Constructs the return type of the resutls from the MCWM and PMCMC algorithm.
"""
function return_results(Theta,loglik,accept_vec,prior_vec)
  return Result(Theta, loglik, accept_vec, prior_vec)
end

doc"""
    return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params)

Constructs the return type of the resutls from the MCWH and PMCMC algorithm.
  """
function return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params)
  if typeof(problem.adaptive_update) == AMUpdate_gen
    return (Result(Theta, loglik, accept_vec, prior_vec), adaptive_update_params[6])
  else
    return Result(Theta, loglik, accept_vec, prior_vec)
  end
end

doc"""
    return_gp_results(gp, Theta,loglik,accept_vec,prior_vec, compare_GP_PF, data_gp_pf,nbr_early_rejections, problem, adaptive_update_params,accept_prob_log,times)

Constructs the return type of the resutls from the gpPMCMC algorithm.
"""
function return_da_results(gp, Theta,loglik,accept_vec,prior_vec, compare_GP_PF,
  data_gp_pf,nbr_early_rejections, problem, adaptive_update_params,accept_prob_log,times)
  if typeof(problem.adaptive_update) == AMUpdate_gen
    return (gpResult(Theta, loglik, accept_vec, prior_vec,compare_GP_PF,data_gp_pf,nbr_early_rejections), adaptive_update_params[6],gp,accept_prob_log,times)
  else
    return gpResult(Theta, loglik, accept_vec, prior_vec,compare_GP_PF,data_gp_pf,nbr_early_rejections),gp,accept_prob_log,times
  end
end

doc"""
    evaluate_prior(theta_star, Theta_bounds)

Calculates the `log-prior` value for the prior distribution for the parameters `theta_star`.

# Inputs
* `theta_star`: the proposal for theta.
* `Theta_bounds`: the bonds for the uniform distributions for the different model parameters.

# Inputs
* `log_prior`: log P(theta_star)
"""
function  evaluate_prior(theta_star, prior_parameters, dist_type = "Uniform")

  # set start value for loglik
  log_prior = 0.

  if dist_type == "Uniform"
    for i = 1:length(theta_star)
      # Update loglik, i.e. add the loglik for each model paramter in theta
      log_prior = log_prior + log_unifpdf( theta_star[i], prior_parameters[i,1], prior_parameters[i,2] )
    end
  else
    # add other priors
  end

  return log_prior # return log_lik

end


doc"""
    log_unifpdf(x::Float64, a::Float64,b::Float64)

Computes log(unifpdf(x,a,b)).
"""
function log_unifpdf(x::Real, a::Real, b::Real)
  if  x >= a && x<= b
    return -log(b-a);
  else
    return log(0);
  end
end



doc"""
    jacobian(theta::Vector, parameter_transformation::String)

Returnes log-Jacobian for transformation of parameter space.
"""
function jacobian(theta::Vector)

  return sum(theta)

end
