# contains the code for the PCMCM, MCWM and particel filters with assiciated help functions


################################################################################
###                       PCMMC/gpPMCMC                                     ####
################################################################################

doc"""
    MCMC(problem::Problem, store_data::Bool=false)

Runs the MCWM algorithms with an adaptive gaussian random walk.

# Inputs
* `Problem`: type that discribes the problem
* `store_data`: return proposals and corresponding loglik values

# Outputs
* `Results`: type with the results
"""
function MCMC(problem::Problem, store_data::Bool=false, return_cov_matrix::Bool=false)

  # data
  Z = problem.data.Z

  # algorithm parameters
  R = problem.alg_param.R
  N = problem.alg_param.N
  burn_in = problem.alg_param.burn_in
  alg = problem.alg_param.alg
  pf_alg = problem.alg_param.pf_alg
  nbr_of_cores = problem.alg_param.nbr_of_cores
  nbr_x0  = problem.alg_param.nbr_x0
  nbr_x = problem.alg_param.nbr_x
  subsample_interval = problem.alg_param.subsample_int
  dt = problem.alg_param.dt
  dt_U = problem.alg_param.dt_U

  # model parameters
  theta_true = problem.model_param.theta_true
  theta_known = problem.model_param.theta_known
  theta_0 = problem.model_param.theta_0

  # pre-allocate matricies and vectors
  Theta = zeros(length(theta_0),R)
  loglik = zeros(R)
  accept_vec = zeros(R)
  prior_vec = zeros(R)
  theta_star = zeros(length(theta_0),1)
  loglik_current = zero(Float64)
  loglik_star = zero(Float64)
  a_log = zero(Float64)

  if store_data
    Theta_val = zeros(length(theta_0),R-burn_in)
    loglik_val = zeros(R-burn_in)
  end

  # draw u's for checking if u < a
  u_log = log(rand(1,R))

  # parameters for adaptive update
  adaptive_update_params = set_adaptive_alg_params(problem.adaptive_update, length(theta_0),Theta[:,1], R)


  # parameters for prior dist
  dist_type = problem.prior_dist.dist
  Theta_parameters = problem.prior_dist.Theta_parameters

  @printf "#####################################################################"


  # print information at start of algorithm
  @printf "Starting MCMC with adaptive RW estimating %d parameters\n" length(theta_true)
  @printf "Algorithm: %s\n" alg
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist
  @printf "Particel filter: %s, on %d cores\n" pf_alg nbr_of_cores

  nbr_of_proc = set_nbr_cores(nbr_of_cores, pf_alg)
  loglik_vec = SharedArray(Float64, 5)

  # print acceptance rate each print_interval:th iteration
  print_interval = 50

  # first iteration
  @printf "Iteration: %d\n" 1 # print first iteration
  @printf "Covariance:\n"
  print_covariance(problem.adaptive_update,adaptive_update_params, 1)

  Theta[:,1] = theta_0

  if pf_alg == "parallel_apf"
    loglik[1] = apf_paralell(Z, Theta[:,1],theta_known,N,dt,nbr_x0, nbr_x,subsample_interval,true,false, nbr_of_proc)
  elseif pf_alg == "parallel_bootstrap"
    loglik[1] = pf_paralell(Z, Theta[:,1],theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,true,false, nbr_of_proc,loglik_vec)
  end

  # print start loglik
  @printf "Loglik: %.4f \n" loglik[1]

  for r = 2:R

    # set print_on to false, only print each print_interval:th iteration
    print_on = false

    # Gaussian random walk
    (theta_star,Z_proposal) = gaussian_random_walk(problem.adaptive_update, adaptive_update_params, Theta[:,r-1], r)


    # print acceptance rate for the last print_interval iterations
    if mod(r-1,print_interval) == 0
      print_on = true # print ESS and Nbr resample each print_interval:th iteration
      # print accaptance rate
      @printf "Acceptance rate on iteration %d to %d is %.4f\n" r-print_interval r-1  sum(accept_vec[r-print_interval:r-1])/( r-1 - (r-print_interval) )
      # print covariance matrix
      @printf "Covariance:\n"
      print_covariance(problem.adaptive_update,adaptive_update_params, r)
      # print loglik
      @printf "Loglik: %.4f \n" loglik[r-1]
    end

    # calc loglik using proposed parameters
    if pf_alg == "parallel_apf"
      loglik_star = apf_paralell(Z, theta_star,theta_known, N,dt,nbr_x0, nbr_x,subsample_interval,print_on,false, nbr_of_proc)
    elseif pf_alg == "parallel_bootstrap"
      loglik_star = pf_paralell(Z, theta_star,theta_known, N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,print_on,false, nbr_of_proc,loglik_vec)
    end

    if store_data && r > burn_in # store data
      Theta_val[:,r-burn_in] = theta_star
      loglik_val[r-burn_in] = loglik_star
    end

    # run MCWM or PMCMC
    if alg == "MCWM"
      if pf_alg == "parallel_bootstrap"
        loglik_current =  pf_paralell(Z, Theta[:,r-1],theta_known, N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,print_on, false, nbr_of_proc,loglik_vec)
      elseif pf_alg == "parallel_apf"
        loglik_current =  apf_paralell(Z, Theta[:,r-1],theta_known, N,dt,nbr_x0, nbr_x,subsample_interval,print_on, false, nbr_of_proc)
      end
    else
      loglik_current = loglik[r-1]
    end

    # evaluate prior distribution
    prior_log_star = evaluate_prior(theta_star,Theta_parameters,dist_type)
    prior_log = evaluate_prior(Theta[:,r-1],Theta_parameters,dist_type)

    if dist_type == "Uniform" # uniform priors

      if prior_log_star == -Inf # reject if the proposed theta is outside the prior
        prior_vec[r] = 1;
        accept = false;
      else
        a_log = loglik_star  -  loglik_current
        accept = u_log[r] < a_log # calc accept
      end

    elseif dist_type == "Normal"

      a_log = (loglik_star + prior_log_star) -  (loglik_current + prior_log)
      accept = u_log[r] < a_log # calc accept

    elseif dist_type == "nonlog"

      if prior_log_star == -Inf # reject if the proposed theta is outside the prior

        prior_vec[r] = 1;
        accept = false;

      else

        # add Jacobian contribution since we have the priors on non-log-scale
        a_log = (loglik_star + prior_log_star + sum(theta_star) ) -  ( loglik_current + prior_log + sum(Theta[r-1,:]) )

        accept = u_log[r] < a_log # calc accept
      end

    else
      # add code for some other prior dist
    end

    # update chain
    if accept # the proposal is accapted
      Theta[:,r] = theta_star # update chain with new values
      loglik[r] = loglik_star
      accept_vec[r] = 1
    else
      Theta[:,r] = Theta[:,r-1] # keep old values
      loglik[r] = loglik[r-1]
    end

    # Adaptation
    adaptation(problem.adaptive_update, adaptive_update_params, Theta, r,a_log)

  end

  @printf "Ending MCMC with adaptive RW estimating %d parameters\n" length(theta_true)
  @printf "Algorithm: %s\n" alg
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist
  @printf "Particel filter: %s, on %d cores\n" pf_alg nbr_of_cores

  @printf "#####################################################################"


  if store_data && return_cov_matrix
    cov_prop_kernel = get_covariance(problem.adaptive_update,adaptive_update_params, R)
    return return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params), Theta_val, loglik_val, cov_prop_kernel
  elseif store_data
   return return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params), Theta_val, loglik_val
  else
    return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params)
  end

end


doc"""
    dagpMCMC(problem_traning::Problem, problem::gpProblem, gp::GPModel, cov_matrix::Matrix)

Runs the dagpMCMC algorithms with an adaptive gaussian random walk.

# Inputs
* `Problem`: type that discribes the problem
* `store_data`: return proposals and corresponding loglik values

# Outputs
* `Results`: type with the results
"""
function dagpMCMC(problem_traning::Problem, problem::gpProblem, gp::GPModel, cov_matrix::Matrix)

  # data
  Z = problem.data.Z

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
  #print_interval = problem.alg_param.print_interval # print the accaptance rate every print_interval:th iteration
  nbr_predictions = problem.alg_param.nbr_predictions # number of predictions to compute at each iteration
  selection_method = problem.alg_param.selection_method # selection method
  lasso = problem.alg_param.lasso # use Lasso
  beta_mh = problem.alg_param.beta_MH

  nbr_of_cores = problem.alg_param.nbr_of_cores
  nbr_x0  = problem.alg_param.nbr_x0
  nbr_x = problem.alg_param.nbr_x
  subsample_interval = problem.alg_param.subsample_int
  dt = problem.alg_param.dt

  # model parameters
  theta_true = problem.model_param.theta_true
  theta_known = problem.model_param.theta_known
  theta_0 = problem.model_param.theta_0

  # pre-allocate matricies and vectors
  Theta = zeros(length(theta_0), R)
  loglik = zeros(R)
  accept_vec = zeros(R)
  prior_vec = zeros(R)
  theta_star = zeros(length(theta_0))
  compare_GP_PF = zeros(2,R-length_training_data-burn_in)
  data_gp_pf = zeros(length(theta_0)+2,R-length_training_data-burn_in)
  data_training = zeros(1+length(theta_0), length_training_data)
  theta_gp_predictions = zeros(length(theta_0), nbr_predictions)
  accept_prob_log = zeros(2, R-length_training_data-burn_in) # [gp ; pf]

  loglik_star = zero(Float64)
  loglik_gp = zeros(nbr_predictions)
  loglik_gp_old = zero(Float64)
  loglik_gp_new = zero(Float64)
  index_keep_gp_er = zero(Int64)
  nbr_early_rejections = zero(Int64)
  accept_gp = true
  accept = true
  pdf_indecies_selction = zeros(nbr_predictions)
  secound_stage_direct_limit = zero(Float64)
  secound_stage_direct = false
  nbr_obs_left_tail = 0
  nbr_split_accaptance_region = 0
  nbr_split_accaptance_region_early_accept = 0
  nbr_second_stage_accepted = 0
  nbr_second_stage = 0
  assumption_list = []
  loglik_list = []
  a_log = zero(Float64)

  # starting values for times:
  time_pre_er = zero(Float64)
  time_fit_gp = zero(Float64)
  time_er_part = zero(Float64)



  # draw u's for checking if u < a
  u_log = log(rand(1,R))



  # set start value
  #theta_0 = theta_training[:, end] # start at last value of the chain for the training part
  Theta[:,1] = theta_0

  # parameters for prior dist
  dist_type = problem.prior_dist.dist
  Theta_parameters = problem.prior_dist.Theta_parameters

  # prop kernl for DA-GP-MCMC
  problem.adaptive_update = noAdaptation(cov_matrix)

  adaptive_update_params = set_adaptive_alg_params(problem.adaptive_update, length(theta_0),Theta[:,1], R)

  # prop kernl for secound_stage_direct
  xi = 1.2
  kernel_secound_stage_direct = noAdaptation(xi^2*cov_matrix)
  adaptive_update_params_secound_stage_direct = set_adaptive_alg_params(kernel_secound_stage_direct, length(theta_0),Theta[:,1], R)

  @printf "#####################################################################"

  # print information at start of algorithm
  @printf "Starting DA-GP-MCMC with adaptive RW estimating %d parameters\n" length(theta_true)
  @printf "MCMC algorithm: %s\n" alg
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist
  @printf "Particel filter: %s\n" pf_alg


  @printf "Covariance - kernel_secound_stage_direct:\n"
  print_covariance(kernel_secound_stage_direct,adaptive_update_params_secound_stage_direct, 1)




  # first iteration
  @printf "Iteration: %d\n" 1 # print first iteration
  @printf "Covariance:\n"
  print_covariance(problem.adaptive_update,adaptive_update_params, 1)

  # set nbr of cores to use for parallel pf
  nbr_of_proc = set_nbr_cores(nbr_of_cores, pf_alg)

  # print acceptance rate each print_interval:th iteration
  print_interval = 1000

  # first iteration
  @printf "Iteration: %d\n" 1 # print first iteration
  @printf "Covariance:\n"
  print_covariance(problem.adaptive_update,adaptive_update_params, 1)

  Theta[:,1] = theta_0

  if pf_alg == "parallel_apf"
    #loglik[1] = apf_paralell(Z, Theta[:,1],theta_known,N,dt,nbr_x0, nbr_x,subsample_interval,true,false, nbr_of_proc)
  elseif pf_alg == "parallel_bootstrap"
    loglik[1] = pf_paralell(Z, Theta[:,1],theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,true,false, nbr_of_proc)
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
      # print accaptance rate
      @printf "Acceptance rate on iteration %d to %d is %.4f\n" r-print_interval r-1  sum(accept_vec[r-print_interval:r-1])/( r-1 - (r-print_interval) )
      # print covariance function
      @printf "Covariance:\n"
      print_covariance(problem.adaptive_update,adaptive_update_params, r)
      # print loglik
      @printf "Loglik: %.4f \n" loglik[r-1]
    end

    secound_stage_direct = rand() < beta_MH # we always run the early-rejection scheme



    if secound_stage_direct

      # secound stage direct

      nbr_obs_left_tail = nbr_obs_left_tail + 1


      # Gaussian random walk using secound stage direct kernel
      (theta_star, ) = gaussian_random_walk(kernel_secound_stage_direct, adaptive_update_params_secound_stage_direct, Theta[:,r-1], r)


      # calc loglik using proposed parameters
      if pf_alg == "parallel_apf"
        error("The auxiliary particle filter is not implemented")
      elseif pf_alg == "parallel_bootstrap"
        loglik_star = pf_paralell(Z, theta_star,theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,print_on,false, nbr_of_proc)
      end

      # evaluate prior distribution
      prior_log_star = evaluate_prior(theta_star,Theta_parameters,dist_type)
      prior_log = evaluate_prior(Theta[:,r-1],Theta_parameters,dist_type)


      if dist_type == "Normal" # uniform priors

        # compute accaptance probability
        if alg == "MCWM"
          if pf_alg == "parallel_apf"
            error("The auxiliary particle filter is not implemented")
          elseif pf_alg == "parallel_bootstrap"
            a_log = (loglik_star + prior_log_star) - (pf_paralell(Z, Theta[:,r-1],theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc) + prior_log)
          end
        else
          a_log =  (loglik_star + prior_log_star)  -  (loglik[r-1] + prior_log)
        end

        # calc accaptance decision
        accept = u_log[r] < a_log

      end

      if accept # the proposal is accapted
        Theta[:,r] = theta_star # update chain with new values
        loglik[r] = loglik_star
        accept_vec[r] = 1
      else
        Theta[:,r] = Theta[:,r-1] # keep old values
        loglik[r] = loglik[r-1]
      end


    else

      # Gaussian random walk using DA proposal kernel
      for i = 1:size(theta_gp_predictions,2)
        (theta_gp_predictions[:,i], ) = gaussian_random_walk(problem.adaptive_update, adaptive_update_params, Theta[:,r-1], r)
      end

      # compute theta_star
      if selection_method == "max_loglik"

        # calc estimation of loglik using the max_loglik selection method

        (loglik_gp,std_loglik) = predict(theta_gp_predictions, gp, pred_method,est_method,noisy_est,true)
        index_keep_gp_er = indmax(loglik_gp)
        loglik_gp_new = loglik_gp[index_keep_gp_er]
        loglik_gp_new_std = std_loglik[index_keep_gp_er]

        #secound_stage_direct = std_loglik[index_keep_gp_er] >= secound_stage_direct_limit

        if print_on
          println("Predictions:")
          println(theta_gp_predictions)
          println("loglik values:")
          println(loglik_gp)
          println("best loglik value:")
          println(loglik_gp_new)
          println("index for  best loglik value:")
          println(index_keep_gp_er)
          println("std_loglik:")
          println(std_loglik[index_keep_gp_er])
          println("secound_stage_direct:")
          println(secound_stage_direct)

        end

      elseif selection_method == "local_loglik_approx"

        # calc estimation of loglik using the local_loglik_approx selection method

        (mean_pred_ml, var_pred_ml, prediction_sample_ml) = predict(theta_gp_predictions,gp,noisy_est)

        if est_method == "mean"

          index_keep_gp_er = stratresample(mean_pred_ml/sum(mean_pred_ml),1)[1]
          loglik_gp_new = mean_pred_ml[index_keep_gp_er]
          loglik_gp_new_std = var_pred_ml[index_keep_gp_er]

          #secound_stage_direct = var_pred_ml[index_keep_gp_er] >= secound_stage_direct_limit

          if print_on
            println("Predictions:")
            println(theta_gp_predictions)
            println("loglik values:")
            println(mean_pred_ml)
            println("best loglik value:")
            println(loglik_gp_new)
            println("index for  best loglik value:")
            println(index_keep_gp_er)
          end

        else

          index_keep_gp_er = stratresample(prediction_sample_ml/sum(prediction_sample_ml),1)[1]
          loglik_gp_new = prediction_sample_ml[index_keep_gp_er]
          loglik_gp_new_std = var_pred_ml[index_keep_gp_er]
          #secound_stage_direct = var_pred_ml[index_keep_gp_er] >= secound_stage_direct_limit

          if print_on
            println("Predictions:")
            println(theta_gp_predictions)
            println("loglik values:")
            println(prediction_sample_ml)
            println("best loglik value:")
            println(loglik_gp_new)
            println("index for  best loglik value:")
            println(index_keep_gp_er)
          end

        end

      end

      # set proposal
      theta_star = theta_gp_predictions[:,index_keep_gp_er]

      prior_log_star = evaluate_prior(theta_star,Theta_parameters,dist_type)
      prior_log = evaluate_prior(Theta[:,r-1],Theta_parameters,dist_type)



      # the ordinary DA method

      if dist_type == "Normal" # uniform priors
        # todo:
        # should we recompute the loglik_gp_old value here?
        loglik_gp_old = predict(Theta[:,r-1], gp, pred_method,est_method,noisy_est)[1]
        a_gp = (loglik_gp_new + prior_log_star)  -  (loglik_gp_old + + prior_log)
        accept_gp = u_log[r] < a_gp # calc accept
        # store accaptance probability
        #accept_prob_log[1, r] = a_gp
      end

      if !accept_gp
        # keep old values
        nbr_early_rejections = nbr_early_rejections + 1
        Theta[:,r] = Theta[:,r-1]
        loglik[r] = loglik[r-1]
        # adaptation of covaraince matrix for the proposal distribution
        # adaptation(problem.adaptive_update, adaptive_update_params, Theta, r,a_gp)
      else

        # run PF
        nbr_second_stage = nbr_second_stage+1

        if pf_alg == "parallel_apf"
          error("The auxiliary particle filter is not implemented")
        elseif pf_alg == "parallel_bootstrap"
          loglik_star = pf_paralell(Z, theta_star,theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,print_on,false, nbr_of_proc)
        end

        if dist_type == "Normal" # uniform priors
          # calc accaptance probability using PF

          if alg == "MCWM"
            a_log = (loglik_star + loglik_gp_old)  -  (pf_paralell(Z, Theta[:,r-1],theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc) + loglik_gp_new)
          else
            a_log = (loglik_star + loglik_gp_old)  -  (loglik[r-1]  + loglik_gp_new)
          end

          accept = log(rand()) < a_log # calc accaptance decision
          #accept_prob_log[2, r] = a_log # store data
        end

        if accept # the proposal is accapted
          nbr_second_stage_accepted = nbr_second_stage_accepted+1
          Theta[:,r] = theta_star # update chain with proposal
          loglik[r] = loglik_star
          accept_vec[r] = 1
        else
          Theta[:,r] = Theta[:,r-1] # keep old values
          loglik[r] = loglik[r-1]
        end
      end
    end
  end

  time_da_part = toc()
  times = [time_pre_er time_fit_gp time_da_part]

  @printf "Ending dagpMCMC estimating %d parameters\n" length(theta_true)
  @printf "Algorithm: %s\n" alg
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist
  @printf "Particel filter: %s\n" pf_alg
  @printf "Time pre-da:  %.0f\n" time_pre_er
  @printf "Time fit GP model:  %.0f\n" time_fit_gp
  @printf "Time da-part:  %.0f\n" time_da_part
  @printf "Number early-rejections: %.d\n"  nbr_early_rejections
  @printf "Secound stage direct limit: %.f\n"  secound_stage_direct_limit
  @printf "Number cases in secound stage: %d\n"  nbr_second_stage
  @printf "Number accepted in secound stage: %d\n"  nbr_second_stage_accepted

  @printf "#####################################################################"


  # return resutls
  return return_gp_results(gp, Theta,loglik,accept_vec,prior_vec, compare_GP_PF, data_gp_pf,nbr_early_rejections, problem, adaptive_update_params,accept_prob_log,times), res_training, theta_training, loglik_training,assumption_list,loglik_list

end



doc"""
    adagpMCMC(problem_traning::Problem, problem::gpProblem, gp::GPModel, cov_matrix::Matrix)

Runs the dagpMCMC algorithms with an adaptive gaussian random walk.

# Inputs
* `Problem`: type that discribes the problem
* `store_data`: return proposals and corresponding loglik values

# Outputs
* `Results`: type with the results
"""
function adagpMCMC(problem_traning::Problem, problem::gpProblem, gp::GPModel, cov_matrix::Matrix, prob_cases::Vector)

  # data
  Z = problem.data.Z

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
  #print_interval = problem.alg_param.print_interval # print the accaptance rate every print_interval:th iteration
  nbr_predictions = problem.alg_param.nbr_predictions # number of predictions to compute at each iteration
  selection_method = problem.alg_param.selection_method # selection method
  lasso = problem.alg_param.lasso # use Lasso
  beta_mh = problem.alg_param.beta_MH

  nbr_of_cores = problem.alg_param.nbr_of_cores
  nbr_x0  = problem.alg_param.nbr_x0
  nbr_x = problem.alg_param.nbr_x
  subsample_interval = problem.alg_param.subsample_int
  dt = problem.alg_param.dt

  # model parameters
  theta_true = problem.model_param.theta_true
  theta_known = problem.model_param.theta_known
  theta_0 = problem.model_param.theta_0

  # pre-allocate matricies and vectors
  Theta = zeros(length(theta_0), R)
  loglik = zeros(R)
  accept_vec = zeros(R)
  prior_vec = zeros(R)
  theta_star = zeros(length(theta_0))
  compare_GP_PF = zeros(2,R-length_training_data-burn_in)
  data_gp_pf = zeros(length(theta_0)+2,R-length_training_data-burn_in)
  data_training = zeros(1+length(theta_0), length_training_data)
  theta_gp_predictions = zeros(length(theta_0), nbr_predictions)
  accept_prob_log = zeros(2, R-length_training_data-burn_in) # [gp ; pf]

  loglik_star = zero(Float64)
  loglik_gp = zeros(nbr_predictions)
  loglik_gp_old = zero(Float64)
  loglik_gp_new = zero(Float64)
  index_keep_gp_er = zero(Int64)
  nbr_early_rejections = zero(Int64)
  accept_gp = true
  accept = true
  pdf_indecies_selction = zeros(nbr_predictions)
  secound_stage_direct_limit = zero(Float64)
  secound_stage_direct = false
  nbr_obs_left_tail = 0
  nbr_split_accaptance_region = 0
  nbr_split_accaptance_region_early_accept = 0
  nbr_split_accaptance_region_early_reject = 0
  nbr_second_stage_accepted = 0
  nbr_second_stage = 0
  assumption_list = []
  loglik_list = []
  a_log = zero(Float64)

  # starting values for times:
  time_pre_er = zero(Float64)
  time_fit_gp = zero(Float64)
  time_er_part = zero(Float64)


  nbr_case_1 = 0
  nbr_case_2 = 0
  nbr_case_3 = 0
  nbr_case_4 = 0

  # draw u's for checking if u < a
  u_log = log(rand(1,R))

  # da new
  std_limit = 0
  loglik_gp_new_std = 0

  #(~,std_loglik_traning) = predict(theta_training, gp, pred_method,est_method,noisy_est,true)
  std_limit = problem.alg_param.std_limit# percentile(std_loglik_traning,50)
  loglik_gp_new_std = 0

  # set start value
  #theta_0 = theta_training[:, end] # start at last value of the chain for the training part
  Theta[:,1] = theta_0

  # parameters for prior dist
  dist_type = problem.prior_dist.dist
  Theta_parameters = problem.prior_dist.Theta_parameters

  # prop kernl for DA-GP-MCMC
  problem.adaptive_update = noAdaptation(cov_matrix)

  adaptive_update_params = set_adaptive_alg_params(problem.adaptive_update, length(theta_0),Theta[:,1], R)

  # prop kernl for secound_stage_direct
  xi = 1.2
  kernel_secound_stage_direct = noAdaptation(xi^2*cov_matrix)
  adaptive_update_params_secound_stage_direct = set_adaptive_alg_params(kernel_secound_stage_direct, length(theta_0),Theta[:,1], R)

  @printf "#####################################################################"

  # print information at start of algorithm
  @printf "Starting DA-GP-MCMC with adaptive RW estimating %d parameters\n" length(theta_true)
  @printf "MCMC algorithm: %s\n" alg
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist
  @printf "Particel filter: %s\n" pf_alg


  @printf "Covariance - kernel_secound_stage_direct:\n"
  print_covariance(kernel_secound_stage_direct,adaptive_update_params_secound_stage_direct, 1)




  # first iteration
  @printf "Iteration: %d\n" 1 # print first iteration
  @printf "Covariance:\n"
  print_covariance(problem.adaptive_update,adaptive_update_params, 1)

  # set nbr of cores to use for parallel pf
  nbr_of_proc = set_nbr_cores(nbr_of_cores, pf_alg)

  # print acceptance rate each print_interval:th iteration
  print_interval = 1000

  # first iteration
  @printf "Iteration: %d\n" 1 # print first iteration
  @printf "Covariance:\n"
  print_covariance(problem.adaptive_update,adaptive_update_params, 1)

  Theta[:,1] = theta_0

  if pf_alg == "parallel_apf"
    #loglik[1] = apf_paralell(Z, Theta[:,1],theta_known,N,dt,nbr_x0, nbr_x,subsample_interval,true,false, nbr_of_proc)
  elseif pf_alg == "parallel_bootstrap"
    loglik[1] = pf_paralell(Z, Theta[:,1],theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,true,false, nbr_of_proc)
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
      # print accaptance rate
      @printf "Acceptance rate on iteration %d to %d is %.4f\n" r-print_interval r-1  sum(accept_vec[r-print_interval:r-1])/( r-1 - (r-print_interval) )
      # print covariance function
      @printf "Covariance:\n"
      print_covariance(problem.adaptive_update,adaptive_update_params, r)
      # print loglik
      @printf "Loglik: %.4f \n" loglik[r-1]
    end

    secound_stage_direct = rand() < beta_MH # we always run the early-rejection scheme



    if secound_stage_direct

      # secound stage direct

      nbr_obs_left_tail = nbr_obs_left_tail + 1


      # Gaussian random walk using secound stage direct kernel
      (theta_star, ) = gaussian_random_walk(kernel_secound_stage_direct, adaptive_update_params_secound_stage_direct, Theta[:,r-1], r)


      # calc loglik using proposed parameters
      if pf_alg == "parallel_apf"
        error("The auxiliary particle filter is not implemented")
      elseif pf_alg == "parallel_bootstrap"
        loglik_star = pf_paralell(Z, theta_star,theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,print_on,false, nbr_of_proc)
      end

      # evaluate prior distribution
      prior_log_star = evaluate_prior(theta_star,Theta_parameters,dist_type)
      prior_log = evaluate_prior(Theta[:,r-1],Theta_parameters,dist_type)


      if dist_type == "Normal" # uniform priors

        # compute accaptance probability
        if alg == "MCWM"
          if pf_alg == "parallel_apf"
            error("The auxiliary particle filter is not implemented")
          elseif pf_alg == "parallel_bootstrap"
            a_log = (loglik_star + prior_log_star) - (pf_paralell(Z, Theta[:,r-1],theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc) + prior_log)
          end
        else
          a_log =  (loglik_star + prior_log_star)  -  (loglik[r-1] + prior_log)
        end

        # calc accaptance decision
        accept = u_log[r] < a_log

      end

      if accept # the proposal is accapted
        Theta[:,r] = theta_star # update chain with new values
        loglik[r] = loglik_star
        accept_vec[r] = 1
      else
        Theta[:,r] = Theta[:,r-1] # keep old values
        loglik[r] = loglik[r-1]
      end


    else

      # Gaussian random walk using DA proposal kernel
      for i = 1:size(theta_gp_predictions,2)
        (theta_gp_predictions[:,i], ) = gaussian_random_walk(problem.adaptive_update, adaptive_update_params, Theta[:,r-1], r)
      end

      # compute theta_star
      if selection_method == "max_loglik"

        # calc estimation of loglik using the max_loglik selection method

        (loglik_gp,std_loglik) = predict(theta_gp_predictions, gp, pred_method,est_method,noisy_est,true)
        index_keep_gp_er = indmax(loglik_gp)
        loglik_gp_new = loglik_gp[index_keep_gp_er]
        loglik_gp_new_std = std_loglik[index_keep_gp_er]

        #secound_stage_direct = std_loglik[index_keep_gp_er] >= secound_stage_direct_limit

        if print_on
          println("Predictions:")
          println(theta_gp_predictions)
          println("loglik values:")
          println(loglik_gp)
          println("best loglik value:")
          println(loglik_gp_new)
          println("index for  best loglik value:")
          println(index_keep_gp_er)
          println("std_loglik:")
          println(std_loglik[index_keep_gp_er])
          println("secound_stage_direct:")
          println(secound_stage_direct)

        end

      elseif selection_method == "local_loglik_approx"

        # calc estimation of loglik using the local_loglik_approx selection method

        (mean_pred_ml, var_pred_ml, prediction_sample_ml) = predict(theta_gp_predictions,gp,noisy_est)

        if est_method == "mean"

          index_keep_gp_er = stratresample(mean_pred_ml/sum(mean_pred_ml),1)[1]
          loglik_gp_new = mean_pred_ml[index_keep_gp_er]
          loglik_gp_new_std = var_pred_ml[index_keep_gp_er]

          #secound_stage_direct = var_pred_ml[index_keep_gp_er] >= secound_stage_direct_limit

          if print_on
            println("Predictions:")
            println(theta_gp_predictions)
            println("loglik values:")
            println(mean_pred_ml)
            println("best loglik value:")
            println(loglik_gp_new)
            println("index for  best loglik value:")
            println(index_keep_gp_er)
          end

        else

          index_keep_gp_er = stratresample(prediction_sample_ml/sum(prediction_sample_ml),1)[1]
          loglik_gp_new = prediction_sample_ml[index_keep_gp_er]
          loglik_gp_new_std = var_pred_ml[index_keep_gp_er]
          #secound_stage_direct = var_pred_ml[index_keep_gp_er] >= secound_stage_direct_limit

          if print_on
            println("Predictions:")
            println(theta_gp_predictions)
            println("loglik values:")
            println(prediction_sample_ml)
            println("best loglik value:")
            println(loglik_gp_new)
            println("index for  best loglik value:")
            println(index_keep_gp_er)
          end

        end

      end

      # set proposal
      theta_star = theta_gp_predictions[:,index_keep_gp_er]

      prior_log_star = evaluate_prior(theta_star,Theta_parameters,dist_type)
      prior_log = evaluate_prior(Theta[:,r-1],Theta_parameters,dist_type)

      # stage 1:


      if dist_type == "Normal" # uniform priors
        # todo:
        # should we recompute the loglik_gp_old value here?
        #loglik_gp_old = predict(Theta[:,r-1], gp, pred_method,est_method,noisy_est, true)
        prediction_sample, sigma_pred = predict(Theta[:,r-1], gp, pred_method,est_method,noisy_est, true)
        loglik_gp_old = prediction_sample[1]
        loglik_gp_old_std = sigma_pred[1]
        a_gp = (loglik_gp_new + prior_log_star) -  (loglik_gp_old + prior_log)
        accept_gp = u_log[r] < a_gp # calc accept
        # store accaptance probability
        #accept_prob_log[1, r] = a_gp
      end

      if !accept_gp # early-accept

        # keep old values
        nbr_early_rejections = nbr_early_rejections + 1
        Theta[:,r] = Theta[:,r-1]
        loglik[r] = loglik[r-1]
        # adaptation of covaraince matrix for the proposal distribution
        # adaptation(problem.adaptive_update, adaptive_update_params, Theta, r,a_gp)

      else

        nbr_second_stage = nbr_second_stage+1

        # A-DA
        u_log_hat = log(rand())

        if loglik_gp_old < loglik_gp_new #&& loglik_gp_new_std < std_limit && loglik_gp_old_std < std_limit

          nbr_split_accaptance_region = nbr_split_accaptance_region+1

          if rand(Bernoulli(prob_cases[1])) == 1 #&& loglik_gp_new_std < std_limit && loglik_gp_old_std < std_limit

            nbr_case_1 = nbr_case_1 + 1

            if u_log_hat < loglik_gp_old - loglik_gp_new
              nbr_second_stage_accepted = nbr_second_stage_accepted+1
              nbr_split_accaptance_region_early_accept = nbr_split_accaptance_region_early_accept+1
              Theta[:,r] = theta_star # update chain with proposal
              loglik[r] = NaN
              accept_vec[r] = 1
            else

              if pf_alg == "parallel_pf"
                error("The auxiliary particle filter is not implemented")
              elseif pf_alg == "parallel_bootstrap"
                loglik_star = pf_paralell(Z, theta_star,theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,print_on,false, nbr_of_proc)
              end

              if dist_type == "Normal" # uniform priors
                # calc accaptance probability using PF
                # can only run MCWM in this case
                loglik_old = pf_paralell(Z, Theta[:,r-1],theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc)
                a_log = (loglik_star + loglik_gp_old)  -  (loglik_old + loglik_gp_new)
                assumption = loglik_star > loglik_old
                push!(loglik_list, [loglik_star loglik[r-1] loglik_gp_new loglik_gp_old loglik_gp_new_std])
                push!(assumption_list, assumption)
                accept = u_log_hat < a_log # calc accaptance decision
                #accept_prob_log[2, r] = a_log # store data
              end

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

          nbr_case_3 = nbr_case_3 + 1

          # run case 3
          if u_log_hat > loglik_gp_old - loglik_gp_new

            nbr_split_accaptance_region_early_reject = nbr_split_accaptance_region_early_reject+1
            Theta[:,r] = Theta[:,r-1] # keep old values
            loglik[r] = loglik[r-1]
            accept_vec[r] = 1

          else

            if pf_alg == "parallel_pf"
              error("The auxiliary particle filter is not implemented")
            elseif pf_alg == "parallel_bootstrap"
              loglik_star = pf_paralell(Z, theta_star,theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,print_on,false, nbr_of_proc)
            end

            if dist_type == "Normal" # uniform priors
              # calc accaptance probability using PF
              # can only run MCWM in this case
              loglik_old = pf_paralell(Z, Theta[:,r-1],theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc)
              a_log = (loglik_star + loglik_gp_old)  -  (loglik_old + loglik_gp_new)
              assumption = loglik_star > loglik_old
              push!(loglik_list, [loglik_star loglik[r-1] loglik_gp_new loglik_gp_old loglik_gp_new_std])
              push!(assumption_list, assumption)
              accept = u_log_hat < a_log # calc accaptance decision
              #accept_prob_log[2, r] = a_log # store data
            end

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

        if rand(Bernoulli(prob_cases[2])) == 1

          # case 2
          nbr_case_2 = nbr_case_2 + 1


          if pf_alg == "parallel_pf"
            error("The auxiliary particle filter is not implemented")
          elseif pf_alg == "parallel_bootstrap"
            loglik_star = pf_paralell(Z, theta_star,theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,print_on,false, nbr_of_proc)
          end

          if dist_type == "Normal" # uniform priors
            # calc accaptance probability using PF
            # can only run MCWM in this case
            loglik_old = pf_paralell(Z, Theta[:,r-1],theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc)
            a_log = (loglik_star + loglik_gp_old)  -  (loglik_old + loglik_gp_new)
            assumption = loglik_star > loglik_old
            push!(loglik_list, [loglik_star loglik[r-1] loglik_gp_new loglik_gp_old loglik_gp_new_std])
            push!(assumption_list, assumption)
            accept = u_log_hat < a_log # calc accaptance decision
            #accept_prob_log[2, r] = a_log # store data
          end

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
         nbr_second_stage_accepted = nbr_second_stage_accepted+1
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

  @printf "Ending adagpMCMC with adaptive RW estimating %d parameters\n" length(theta_true)
  @printf "Algorithm: %s\n" alg
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist
  @printf "Particel filter: %s\n" pf_alg
  @printf "Time pre-er:  %.0f\n" time_pre_er
  @printf "Time fit GP model:  %.0f\n" time_fit_gp
  @printf "Time er-part:  %.0f\n" time_da_part
  @printf "Number early-rejections: %.d\n"  nbr_early_rejections
  @printf "Secound stage direct limit: %.f\n"  secound_stage_direct_limit
  @printf "Number of left-tail obs. with direct run of stage 2: %d\n"  nbr_obs_left_tail

  @printf "Number split accaptance region: %d\n" nbr_split_accaptance_region
  @printf "Number split accaptance region early-accept: %d\n"  nbr_split_accaptance_region_early_accept
  @printf "Number split accaptance region early-reject: %d\n"  nbr_split_accaptance_region_early_reject

  @printf "Number cases in secound stage: %d\n"  nbr_second_stage
  @printf "Number accepted in secound stage: %d\n"  nbr_second_stage_accepted


  if length(assumption_list) > 0
    @printf "Nbr assumtion correct: %.f\n"  sum(assumption_list)
    @printf "Proc assumtion correct: %.f\n"  sum(assumption_list)/length(assumption_list)
  end

  @printf "Number case 1: %d\n"  nbr_case_1
  @printf "Number case 2: %d\n"  nbr_case_2
  @printf "Number case 3: %d\n"  nbr_case_3
  @printf "Number case 4: %d\n"  nbr_case_4

  @printf "#####################################################################"

  # return resutls
  return return_gp_results(gp, Theta,loglik,accept_vec,prior_vec, compare_GP_PF, data_gp_pf,nbr_early_rejections, problem, adaptive_update_params,accept_prob_log,times), res_training, theta_training, loglik_training,assumption_list,loglik_list

end





################################################################################
###               Help functions for PMCMC and gpPMCMC                       ###
################################################################################

doc"""
    return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params)

Constructs the return type of the resutls from the MCWH and PMCMC algorithm.
"""
function return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params)
  if typeof(problem.adaptive_update) == AMUpdate_gen
    return (Result(Theta, loglik, accept_vec, prior_vec), adaptive_update_params[6])
  elseif typeof(problem.adaptive_update) == AMUpdate_comp_w
    return (Result(Theta, loglik, accept_vec, prior_vec), adaptive_update_params[7])
  elseif typeof(problem.adaptive_update) == AMUpdate_gen_comp_w
    return (Result(Theta, loglik, accept_vec, prior_vec), adaptive_update_params[7])
  else
    return Result(Theta, loglik, accept_vec, prior_vec)
  end
end

doc"""
    return_gp_results(Theta,loglik,accept_vec,prior_vec, compare_GP_PF, data_gp_pf, problem, adaptive_update_params)

Constructs the return type of the resutls from the gpPMCMC algorithm.
"""
function return_gp_results(gp, Theta,loglik,accept_vec,prior_vec, compare_GP_PF, data_gp_pf,nbr_early_rejections, problem, adaptive_update_params,accept_prob_log,times)
  if typeof(problem.adaptive_update) == AMUpdate_gen
    return (gpResult(Theta, loglik, accept_vec, prior_vec,compare_GP_PF,data_gp_pf,nbr_early_rejections), adaptive_update_params[6],gp,accept_prob_log,times)
#  elseif typeof(problem.adaptive_update) == AMUpdate_comp_w
#    return (gpResult(Theta, loglik, accept_vec, prior_vec,compare_GP_PF,data_gp_pf,nbr_early_rejections), adaptive_update_params[7],gp,accept_prob_log,times)
  else
    return gpResult(Theta, loglik, accept_vec, prior_vec,compare_GP_PF,data_gp_pf,nbr_early_rejections),gp,accept_prob_log,times
  end
end


doc"""
    set_nbr_cores(nbr_of_cores::Int64, pf_alg)

Sets the number of cores and particel algorithm to use.
"""
function set_nbr_cores(nbr_of_cores::Int64, pf_alg::String)
  if length(workers()) == 1
    nbr_of_proc = nbr_of_cores
    addprocs(nbr_of_proc)
    if pf_alg == "parallel_bootstrap"
      @everywhere include("run_pf_paralell.jl")
    else
      @everywhere include("run_apf_paralell.jl")
    end
  else
    nbr_of_proc = length(workers())
  end
  return nbr_of_proc
end

doc"""
    evaluate_prior(theta_star, Theta_bounds)

Calculates the `log-likelihood` for the prior distribution for the parameters `theta_star`.

# Inputs
* `theta_star`: the proposal for theta.
* `Theta_bounds`: the bonds for the uniform distributions for the different model parameters.

# Inputs
* `log_likelihood`: log P(theta_star)
"""
function  evaluate_prior(theta_star, Theta_parameters, dist_type = "Unifrom")

  # set start value for loglik
  log_likelihood = 0.

  if dist_type == "Uniform"
    for i = 1:length(theta_star)
      # Update loglik, i.e. add the loglik for each model paramter in theta
      log_likelihood = log_likelihood + log_unifpdf( theta_star[i], Theta_parameters[i,1], Theta_parameters[i,2] )
    end
  elseif dist_type == "Normal"
    for i = 1:length(theta_star)
      # Update loglik, i.e. add the loglik for each model paramter in theta
      log_likelihood = log_likelihood + log_normpdf(theta_star[i],Theta_parameters[i,1],Theta_parameters[i,2])
    end
  elseif dist_type == "nonlog"
    # add code to handle priors on non-log-scale!
    if length(theta_star) == 2
      for i = 1:length(theta_star)
        # the unknown parameters c and d both have normal prior dists
        log_likelihood = log_likelihood + log_normpdf( exp(theta_star[i]), Theta_parameters[i,1], Theta_parameters[i,2] )
      end
    elseif length(theta_star) == 3
      # the unknown parameter A has a inv-gamma prior dist
      log_likelihood = log_likelihood + log_invgampdf(exp(theta_star[1]), Theta_parameters[1,1], Theta_parameters[1,2])
      for i = 2:3
        # The unknown parameters c and d both have normal prior dists
        log_likelihood = log_likelihood + log_normpdf( exp(theta_star[i]), Theta_parameters[i,1], Theta_parameters[i,2] )
      end
    elseif length(theta_star) == 5
      for i in [1 2 5]
        # The unknown parameters Κ,Γ and sigma both have gamma prior dists
        log_likelihood = log_likelihood + log_gampdf( exp(theta_star[i]), Theta_parameters[i,1], Theta_parameters[i,2] )
      end
      for i = 3:4
        # The unknown parameters c and d both have normal prior dists
        log_likelihood = log_likelihood + log_normpdf( exp(theta_star[i]), Theta_parameters[i,1], Theta_parameters[i,2] )
      end
    elseif length(theta_star) == 6
      # The unknown parameter A has a gaminv prior dist
      log_likelihood = log_likelihood + log_invgampdf(exp(theta_star[1]), Theta_parameters[1,1], Theta_parameters[1,2])
      for i = 2:3
        # The unknown parameters c and d both have normal prior dists
        log_likelihood = log_likelihood + log_normpdf( exp(theta_star[i]), Theta_parameters[i,1], Theta_parameters[i,2] )
      end
      for i = 4:length(theta_star)
        # The unknown parameters p1, p2 and sigma both have gamma prior dists
        log_likelihood = log_likelihood + log_gampdf( exp(theta_star[i]), Theta_parameters[i,1], Theta_parameters[i,2] )
      end
    elseif length(theta_star) == 7
      for i = [1,2,5,6,7]
        # The unknown parameters Κ,Γ,power1,power2 and sigma all have gamma prior dists
        log_likelihood = log_likelihood + log_gampdf( exp(theta_star[i]), Theta_parameters[i,1], Theta_parameters[i,2] )
      end
      for i = [3,4]
        # The unknown parameters c and d have normal prior dists
        log_likelihood = log_likelihood + log_normpdf( exp(theta_star[i]), Theta_parameters[i,1], Theta_parameters[i,2] )
      end

    elseif length(theta_star) == 8
      # the unknown parameter A has a inv-gamma prior dist
      log_likelihood = log_likelihood + log_invgampdf(exp(theta_star[3]), Theta_parameters[3,1], Theta_parameters[3,2])
      for i = [1,2,5,6,7]
        # The unknown parameters Κ,Γ,power1,power2 and sigma all have gamma prior dists
        log_likelihood = log_likelihood + log_gampdf( exp(theta_star[i]), Theta_parameters[i,1], Theta_parameters[i,2] )
      end
      for i = [3,4]
        # The unknown parameters c and d have normal prior dists
        log_likelihood = log_likelihood + log_normpdf( exp(theta_star[i]), Theta_parameters[i,1], Theta_parameters[i,2] )
      end
    elseif length(theta_star) == 4
      for i = 1:2
        # The unknown parameters Κ and Γ both have gammma prior dists
        log_likelihood = log_likelihood + log_invgampdf( exp(theta_star[i]), Theta_parameters[i,1], Theta_parameters[i,2] )
      end
      for i = 3:4
        # The unknown parameters c and d both have normal prior dists
        log_likelihood = log_likelihood + log_normpdf( exp(theta_star[i]), Theta_parameters[i,1], Theta_parameters[i,2] )
      end

    else
      for i in [3 6]
        # The unknown parameters A and g have a gaminv prior dist
        log_likelihood = log_likelihood + log_invgampdf(exp(theta_star[i]), Theta_parameters[i,1], Theta_parameters[i,2])
      end
      for i in [4 5]
        # The unknown parameters c and d both have normal prior dists
        log_likelihood = log_likelihood + log_normpdf( exp(theta_star[i]), Theta_parameters[i,1], Theta_parameters[i,2] )
      end
      for i in [1 2 7 8 9]
        # The unknown parameters Κ,Γ,power1,power2 and sigma all have gamma prior dists
        log_likelihood = log_likelihood + log_gampdf( exp(theta_star[i]), Theta_parameters[i,1], Theta_parameters[i,2] )
      end

    end

  end

  return log_likelihood # return log_lik

end



doc"""
    log_gampdf(x::Float64, mu::Float64, sigma::Float64)

Computes log(normpdf(x,a,b)) without normalizing constant.

"""
function log_gampdf(x::Float64, a::Float64, b::Float64)
  #return -a*log(b) - log(gamma(a)) - (a-1)*log(x) - x/b
  return -(a-1)*log(x) - x/b

end

doc"""
    log_invgampdf(x::Float64, mu::Float64, sigma::Float64)

Computes log(log_invgampdf(x,a,b)) constant.

"""
function log_invgampdf(x::Float64, a::Float64, b::Float64)
  #return a*log(b) - log(gamma(a)) + (-a-1)*log(x) - b/x
  return (-a-1)*log(x) - b/x
end


doc"""
    log_unifpdf(x::Float64, a::Float64,b::Float64)

Computes log(unifpdf(x,a,b)).

"""
function log_unifpdf(x::Float64, a::Float64,b::Float64)
  if  x >= a && x<= b
    return -log(b-a)
  else
    return log(0)
  end
end

doc"""
    log_normpdf(x::Float64, mu::Float64, sigma::Float64)

Computes log(normpdf(x,a,b)) without normalizing constant.

"""
function log_normpdf(x::Float64, mu::Float64, sigma::Float64)
  #return -log(sigma) - (x-mu)^2/(2*sigma^2)
  return -(x-mu)^2/(2*sigma^2)
end


doc"""
    pf_paralell(Z::Array{Float64},theta::Array{Float64},theta_known::Array{Float64}, N::Int64,dt::Float64, nbr_x0::Int64, nbr_x::Int64,subsample_interval::Int64,print_on::Bool, store_weigths::Bool, nbr_of_proc::Int64)

Runs the bootstrap filter to estimimate the `log-likelihood`
log(P(`Z`|`theta`))
"""
function pf_paralell(Z::Array{Float64},theta::Array{Float64},theta_known::Array{Float64},
  N::Int64,dt::Float64, dt_U::Float64, nbr_x0::Int64, nbr_x::Int64,
  subsample_interval::Int64,print_on::Bool, return_weigths_and_particels::Bool,
  nbr_of_proc::Int64, loglik::SharedArray)

  # set parameters
  (Κ, Γ, A, A_sign,B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, theta_known,length(theta))

  # compute value for A
  A = A*A_sign

  # set value for constant b function
  b_const = sqrt(2.*sigma^2 / 2.)

  # set values needed for calculations in Float64
  (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (subsample_interval, nbr_x0, nbr_x,N))

  if !return_weigths_and_particels

    # run nbr_of_proc parallel estimations

    #loglik = SharedArray{Float64}(nbr_of_proc)

    @sync begin

    @parallel for i = 1:nbr_of_proc
      loglik[i] = run_pf_paralell(Z,theta,theta_known, N, N_calc, dt, dt_U, nbr_x0, nbr_x0_calc, nbr_x, nbr_x_calc, subsample_interval, subsample_interval_calc, print_on, return_weigths_and_particels, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)
    end

    end

    #%println(loglik)
    #println(logsumexp(loglik)-log(nbr_of_proc))

    #println(loglik)

    return logsumexp(loglik)-log(nbr_of_proc)

    #=
    loglik = @parallel (+) for i = 1:nbr_of_proc
      run_pf_paralell(Z,theta,theta_known, N, N_calc, dt, dt_U, nbr_x0, nbr_x0_calc, nbr_x, nbr_x_calc, subsample_interval, subsample_interval_calc, print_on, return_weigths_and_particels, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)
    end

    # return the average of the nbr_of_proc estimations
    return loglik/nbr_of_proc
    =#
  else
    # run nbr_of_proc parallel estimations
    (loglik,weigths, particels) = run_pf_paralell(Z,theta,theta_known, N, N_calc, dt, dt_U, nbr_x0, nbr_x0_calc, nbr_x, nbr_x_calc, subsample_interval, subsample_interval_calc, print_on, return_weigths_and_particels, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

    # return the average of the nbr_of_proc estimations
    return loglik, weigths, particels
  end
end


doc"""
    apf_paralell(Z::Array{Float64},theta::Array{Float64},theta_known::Array{Float64}, N::Int64,dt::Float64, nbr_x0::Int64, nbr_x::Int64,subsample_interval::Int64,print_on::Bool, store_weigths::Bool, nbr_of_proc::Int64)

Runs the auxiliar particle filter filter and estiamtes the `log-likelihood`
P(`Z`|`theta`)
"""
function apf_paralell(Z::Array{Float64},theta::Array{Float64},theta_known::Array{Float64}, N::Int64,dt::Float64, nbr_x0::Int64, nbr_x::Int64,subsample_interval::Int64,print_on::Bool, return_weigths_and_particels::Bool, nbr_of_proc::Int64)

  # set parameters
  (Κ, Γ, A,B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, theta_known,length(theta))


  # set value for constant b function
  b_const = sqrt(2.*sigma^2 / 2.)

  # set values needed for calculations in Float64
  (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (subsample_interval, nbr_x0, nbr_x,N))

  if !return_weigths_and_particels
    # run nbr_of_proc parallel estimations
    loglik = @parallel (+) for i = 1:nbr_of_proc
      run_apf_paralell(Z,theta,theta_known, N, N_calc, dt, nbr_x0, nbr_x0_calc, nbr_x, nbr_x_calc, subsample_interval, subsample_interval_calc, print_on, return_weigths_and_particels, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)
    end

    # return the average of the nbr_of_proc estimations
    return loglik/nbr_of_proc
  else
    # run nbr_of_proc parallel estimations
    (loglik,weigths, particels) = run_apf_paralell(Z,theta,theta_known, N, N_calc, dt, nbr_x0, nbr_x0_calc, nbr_x, nbr_x_calc, subsample_interval, subsample_interval_calc, print_on, return_weigths_and_particels, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

    # return the average of the nbr_of_proc estimations
    return loglik, weigths, particels
  end

end


doc"""
    set_parameters(theta::Array{Float64}, theta_known::Array{Float64}, nbr_of_unknown_parameters::Int64)

Sets the model parameters according to the nbr of unknown parameters. Help function for pf.

"""
function set_parameters(theta::Array{Float64}, theta_known::Array{Float64}, nbr_of_unknown_parameters::Int64)

  # set parameter values
  if nbr_of_unknown_parameters == 8
    # estiamte Κ, Γ,A,c,d,power1,power2,sigma
    (Κ,Γ,A,c,d,power1,power2,sigma) = exp(theta)
    (A_sign,B,f,g) = theta_known
  elseif nbr_of_unknown_parameters == 7
    # estimate Κ,Γ,c,d,power1,power2,sigma
    (Κ,Γ,c,d,power1,power2,sigma) = exp(theta)
    (A,A_sign,B,f,g) = theta_known
  elseif nbr_of_unknown_parameters == 6
    # estiamte A,c,d,power1,power2,sigma
    (A,c,d,power1,power2,sigma) = exp(theta)
    (Κ, Γ, A_sign,B,f,g) = theta_known
  elseif nbr_of_unknown_parameters == 5
    # estimate Κ Γ c d sigma
    (Κ,  Γ, c, d, sigma) = exp(theta)
    (A,A_sign, B, f, g, power1, power2) = theta_known
  elseif nbr_of_unknown_parameters == 4
    # estimate Κ Γ c d
    (Κ, Γ,c,d) = exp(theta)
    (A,A_sign,B,f,g,power1,power2, sigma) = theta_known
  elseif nbr_of_unknown_parameters == 2
    # estimating c and d
    (c,d) = exp(theta)
    (Κ, Γ, A, A_sign, B, f, g, power1, power2, sigma) = theta_known
  elseif nbr_of_unknown_parameters == 3
    # estimate A,c,d
    (A, c, d) = exp(theta)
    (Κ, Γ, A_sign, B, f, g, power1, power2, sigma) = theta_known
  else
    # estimate all parameters. i.e. estimate Κ, Γ,A,c,d,g,power1,power2,sigma
    (Κ, Γ, A,c,d,g,power1,power2,sigma) = exp(theta)
    (A_sign,B,f) = theta_known
  end

  return (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma)

end


################################################################################
###            Functions for particle filter diagnostics                    ####
################################################################################

doc"""
    compare_pf_apf(problem::Problem, nbr_iterations::Int64=100)

Runs the boostrap filter and the auxiliar particle filter for nbr_iterations times.

# Inputs

* `problem`: the type decribing the problem
* `nbr_iterations`: number of iterations of the particel filters

# Output

* `loglik_matrix`: A nbr_iterations x nbr_iterations matrix with the computed log-likelihood values

"""
function compare_pf_apf(problem::Problem, nbr_iterations::Int64=100)

  # data
  Z = problem.data.Z

  # algorithm parameters
  R = problem.alg_param.R
  N = problem.alg_param.N
  burn_in = problem.alg_param.burn_in
  pf_alg = problem.alg_param.pf_alg
  nbr_x0  = problem.alg_param.nbr_x0
  nbr_x = problem.alg_param.nbr_x
  subsample_interval = problem.alg_param.subsample_int
  dt = problem.alg_param.dt

  # model parameters
  theta_true = problem.model_param.theta_true
  theta_known = problem.model_param.theta_known
  theta_0 = problem.model_param.theta_0


  # pre-allocate matrix for storing computed loglik values
  loglik_matrix = zeros(nbr_iterations,2) # pf apf


  @printf "Starting compare_pf_apf\n"


  # add processors and set nbr_pf_proc if using parallel pf
  if length(workers()) == 1
    nbr_of_proc = problem.alg_param.nbr_of_cores # set nbr of workers to use
    addprocs(nbr_of_proc)
    #@everywhere include("run_pf_paralell.jl")
    #@everywhere include("run_apf_paralell.jl")
  else
    nbr_of_proc = length(workers())
  end

  ploton = false

  # run pf
  @printf "Runnig pf %d times\n" nbr_iterations
  @everywhere include("run_pf_paralell.jl")
  for j = 1:nbr_iterations
    if ploton
      @printf "Running pf, iteration: %d\n" j
    end
    loglik_matrix[j,1] = pf_paralell(Z, theta_true,theta_known,N,dt,nbr_x0, nbr_x,subsample_interval,true,false, nbr_of_proc)
  end

  # run apf
  @printf "Runnig apf %d times\n" nbr_iterations
  @everywhere include("run_apf_paralell.jl")
  for j = 1:nbr_iterations
    if ploton
      @printf "Running apf, iteration: %d\n" j
    end
    loglik_matrix[j,2] = apf_paralell(Z, theta_true,theta_known,N,dt,nbr_x0, nbr_x,subsample_interval,true,false, nbr_of_proc)
  end

  return loglik_matrix

end

doc"""
    compare_pf_apf(problem::Problem, nbr_iterations::Int64=100, theta_compute::Array)

Runs the boostrap filter and the auxiliar particle filter for nbr_iterations times.

# Inputs

* `problem`: the type decribing the problem
* `nbr_iterations`: number of iterations of the particel filters

# Output

* `loglik_matrix`: A nbr_iterations x nbr_iterations matrix with the computed log-likelihood values

"""
function compare_pf_apf(problem::Problem, nbr_iterations::Int64, theta_compute::Array)

  # data
  Z = problem.data.Z

  # algorithm parameters
  R = problem.alg_param.R
  N = problem.alg_param.N
  burn_in = problem.alg_param.burn_in
  pf_alg = problem.alg_param.pf_alg
  nbr_x0  = problem.alg_param.nbr_x0
  nbr_x = problem.alg_param.nbr_x
  subsample_interval = problem.alg_param.subsample_int
  dt = problem.alg_param.dt

  # model parameters
  theta_true = problem.model_param.theta_true
  theta_known = problem.model_param.theta_known
  theta_0 = problem.model_param.theta_0


  # pre-allocate matrix for storing computed loglik values
  loglik_matrix = zeros(nbr_iterations,2) # pf apf


  @printf "Starting compare_pf_apf\n"


  # add processors and set nbr_pf_proc if using parallel pf
  if length(workers()) == 1
    nbr_of_proc = problem.alg_param.nbr_of_cores # set nbr of workers to use
    addprocs(nbr_of_proc)
    #@everywhere include("run_pf_paralell.jl")
    #@everywhere include("run_apf_paralell.jl")
  else
    nbr_of_proc = length(workers())
  end

  ploton = false

  # run pf
  @printf "Runnig pf %d times\n" nbr_iterations
  @everywhere include("run_pf_paralell.jl")
  for j = 1:nbr_iterations
    if ploton
      @printf "Running pf, iteration: %d\n" j
    end
    loglik_matrix[j,1] = pf_paralell(Z, theta_compute,theta_known,N,dt,nbr_x0, nbr_x,subsample_interval,true,false, nbr_of_proc)
  end

  # run apf
  @printf "Runnig apf %d times\n" nbr_iterations
  @everywhere include("run_apf_paralell.jl")
  for j = 1:nbr_iterations
    if ploton
      @printf "Running apf, iteration: %d\n" j
    end
    loglik_matrix[j,2] = apf_paralell(Z, theta_compute,theta_known,N,dt,nbr_x0, nbr_x,subsample_interval,true,false, nbr_of_proc)
  end

  return loglik_matrix

end


doc"""
    pf_diagnostics(problem::Problem, nbr_iterations::Int64, theta_compute::Array)

Runs the boostrap filter or the auxiliar particle filter for nbr_iterations times.

# Inputs

* `problem`: the type decribing the problem
* `nbr_iterations`: number of iterations of the particel filters

# Output

* `loglik_matrix`: A nbr_iterations x nbr_iterations matrix with the computed log-likelihood values

"""
function pf_diagnostics(problem::Problem, nbr_iterations::Int64, theta_compute::Array)

  # data
  Z = problem.data.Z

  # algorithm parameters
  R = problem.alg_param.R
  N = problem.alg_param.N
  burn_in = problem.alg_param.burn_in
  pf_alg = problem.alg_param.pf_alg
  nbr_x0  = problem.alg_param.nbr_x0
  nbr_x = problem.alg_param.nbr_x
  subsample_interval = problem.alg_param.subsample_int
  dt = problem.alg_param.dt
  nbr_of_cores = problem.alg_param.nbr_of_cores


  # model parameters
  theta_true = problem.model_param.theta_true
  theta_known = problem.model_param.theta_known
  theta_0 = problem.model_param.theta_0


  # pre-allocate matrix for storing computed loglik values
  loglik_vec = zeros(nbr_iterations,1) # pf apf


  @printf "Running the particle filter %d times \n" nbr_iterations


  # set nbr of cores to use
  nbr_of_proc = set_nbr_cores(nbr_of_cores, pf_alg)

  if pf_alg == "parallel_bootstrap"
    for j = 1:nbr_iterations
      loglik_vec[j] = pf_paralell(Z, theta_compute,theta_known,N,dt,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc)
    end

    # run nbr_of_proc parallel estimations
    (loglik, weigths, particels) = pf_paralell(Z, theta_compute,theta_known,N,dt,nbr_x0, nbr_x,subsample_interval,true,true, nbr_of_proc)

  elseif pf_alg == "parallel_apf"
    for j = 1:nbr_iterations
      loglik_vec[j] = apf_paralell(Z, theta_compute,theta_known,N,dt,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc)
    end

    # run nbr_of_proc parallel estimations
    (loglik, weigths, particels) = apf_paralell(Z, theta_compute,theta_known,N,dt,nbr_x0, nbr_x,subsample_interval,true,true, nbr_of_proc)
  end

  return loglik_vec, weigths, particels

end
