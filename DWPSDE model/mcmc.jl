# contains the code for the PCMCM, MCWM and A(DA)-GP-MCMC algorithms with assiciated help functions

################################################################################
######               algorithms                                            #####
################################################################################

# PMCMC/MCWM

doc"""
    mcmc(problem::Problem, store_data::Bool=false, return_cov_matrix::Bool=false)

Runs PMCMC or MCWM with adaptive gaussian random walk.
"""
function mcmc(problem::Problem, store_data::Bool=false, return_cov_matrix::Bool=false)

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


  # parameters for adaptive update
  adaptive_update_params = set_adaptive_alg_params(problem.adaptive_update, length(theta_0),Theta[:,1], R)


  # parameters for prior dist
  dist_type = problem.prior_dist.dist
  prior_parameters = problem.prior_dist.prior_parameters

  @printf "#####################################################################\n"


  # print information at start of algorithm
  @printf "Starting MCMC with adaptive RW estimating %d parameters\n" length(theta_true)
  @printf "Algorithm: %s\n" alg
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist
  @printf "Particel filter: %s, on %d cores\n" pf_alg nbr_of_cores
  @printf "Nbr particles for particel filter: %d\n" N

  nbr_of_proc = set_nbr_cores(nbr_of_cores, pf_alg)
  #nbr_of_proc = nbr_of_cores
  loglik_vec = SharedArray{Float64}(nbr_of_proc)

  # print acceptance rate each print_interval:th iteration
  print_interval = 1000

  # first iteration
  @printf "Iteration: %d\n" 1 # print first iteration
  @printf "Covariance:\n"
  print_covariance(problem.adaptive_update,adaptive_update_params, 1)

  Theta[:,1] = theta_0
  loglik[1] = pf_paralell(Z, Theta[:,1],theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc,loglik_vec)

  # print start loglik
  @printf "Loglik: %.4f \n" loglik[1]

  for r = 2:R

    # set print_on to false, only print each print_interval:th iteration
    print_on = false

    # Gaussian random walk
    (theta_star,) = gaussian_random_walk(problem.adaptive_update, adaptive_update_params, Theta[:,r-1], r)

    # print acceptance rate for the last print_interval iterations
    if mod(r-1,print_interval) == 0
      # print percentage done
      @printf "Percentage done: %.2f %% \n" 100*(r-1)/R
      print_on = true # print ESS and Nbr resample each print_interval:th iteration
      # print accaptance rate
      @printf "Acceptance rate on iteration %d to %d is %.4f\n" r-print_interval r-1  sum(accept_vec[r-print_interval:r-1])/( r-1 - (r-print_interval) )
      # print covariance matrix
      @printf "Covariance:\n"
      print_covariance(problem.adaptive_update,adaptive_update_params, r)
      # print loglik
      @printf "Loglik: %.4f \n" loglik[r-1]
      # print log-lik vector
      @printf "Loglik values on different cores:\n"
      println(loglik_vec)
    end

    # calc loglik using proposed parameters
    loglik_star = pf_paralell(Z, theta_star,theta_known, N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,print_on,false, nbr_of_proc,loglik_vec)

    if store_data && r > burn_in # store data
      Theta_val[:,r-burn_in] = theta_star
      loglik_val[r-burn_in] = loglik_star
    end

    # run MCWM or PMCMC
    if alg == "MCWM"
      loglik_current =  pf_paralell(Z, Theta[:,r-1],theta_known, N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,print_on, false, nbr_of_proc,loglik_vec)
    else
      loglik_current = loglik[r-1]
    end

    prior_log_star = evaluate_prior(theta_star,prior_parameters, dist_type)
    prior_log_old = evaluate_prior(Theta[:,r-1],prior_parameters, dist_type)

    jacobian_log_star = jacobian(theta_star)
    jacobian_log_old = jacobian(Theta[:,r-1])

    a_log = loglik_star + prior_log_star +  jacobian_log_star - (loglik_current +  prior_log_old + jacobian_log_old)
    # generate log(u)

    u_log = log(rand())
    accept = u_log < a_log # calc accaptace decision

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

  @printf "#####################################################################\n"


  if store_data && return_cov_matrix
    cov_prop_kernel = get_covariance(problem.adaptive_update,adaptive_update_params, R)
    return return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params), Theta_val, loglik_val, cov_prop_kernel
  elseif store_data
   return return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params), Theta_val, loglik_val
  else
    return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params)
  end

end

# DA-GP-MCMC

doc"""
    dagpmcmc(problem_traning::Problem, problem::gpProblem, gp::GPModel, cov_matrix::Matrix)

Runs the DA-GP-MCMC algorithm.
"""
function dagpmcmc(problem_traning::Problem, problem::gpProblem, gp::GPModel, cov_matrix::Matrix,return_run_info::Bool=false)

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
  lasso = problem.alg_param.lasso # use Lasso
  beta_MH = problem.alg_param.beta_MH

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
  #compare_GP_PF = zeros(2,R-length_training_data-burn_in)
  compare_GP_PF = zeros(2,R)
  #data_gp_pf = zeros(length(theta_0)+2,R-length_training_data-burn_in)
  data_gp_pf = zeros(length(theta_0)+2,R)
  data_training = zeros(1+length(theta_0), length_training_data)
  #accept_prob_log = zeros(2, R-length_training_data-burn_in) # [gp ; pf]
  accept_prob_log = zeros(2, R) # [gp ; pf]

  loglik_star = zero(Float64)
  loglik_gp = zeros(Float64)
  loglik_gp_old = zero(Float64)
  loglik_gp_new = zero(Float64)
  index_keep_gp_er = zero(Int64)
  nbr_early_rejections = zero(Int64)
  accept = true
  MH_direct = false
  nbr_ordinary_mh = 0
  nbr_ordinary_mh_accapte = 0
  nbr_split_accaptance_region = 0
  nbr_split_accaptance_region_early_accept = 0
  nbr_second_stage_accepted = 0
  nbr_second_stage = 0
  nbr_run_DA = 0
  assumption_list = []
  loglik_list = []
  a_log = zero(Float64)
  loglik_current = zero(Float64)
  nbr_eval_pf = 0
  nbr_eval_pf_secound_stage = 0

  # starting values for times:
  time_pre_er = zero(Float64)
  time_fit_gp = zero(Float64)
  time_er_part = zero(Float64)

  # parameters for prior dist
  dist_type = problem.prior_dist.dist
  prior_parameters = problem.prior_dist.prior_parameters

  # prop kernl for DA-GP-MCMC
  xi = 1.2
  problem.adaptive_update = noAdaptation(xi^2*cov_matrix)

  adaptive_update_params = set_adaptive_alg_params(problem.adaptive_update, length(theta_0),Theta[:,1], R)

  # prop kernl for MH_direct
  kernel_MH_direct = noAdaptation(cov_matrix)
  adaptive_update_params_MH_direct = set_adaptive_alg_params(kernel_MH_direct, length(theta_0),Theta[:,1], R)

  @printf "#####################################################################\n"

  # print information at start of algorithm
  @printf "Starting DA-GP-MCMC estimating %d parameters \n" length(theta_true)
  @printf "MCMC algorithm: %s\n" alg
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist
  @printf "Particle filter: %s, on %d cores\n" pf_alg nbr_of_cores
  @printf "Nbr particles for particle filter: %d\n" N

  @printf "Covariance - kernel_MH_direct:\n"
  print_covariance(kernel_MH_direct,adaptive_update_params_MH_direct, 1)

  # first iteration
  @printf "Iteration: %d\n" 1 # print first iteration
  @printf "Covariance:\n"
  print_covariance(problem.adaptive_update,adaptive_update_params, 1)

  # set nbr of cores to use for parallel pf
  nbr_of_proc = set_nbr_cores(nbr_of_cores, pf_alg)
  #nbr_of_proc = nbr_of_cores
  loglik_vec = SharedArray{Float64}(nbr_of_cores)

  # print acceptance rate each print_interval:th iteration
  print_interval = 1000

  # first iteration
  @printf "Iteration: %d\n" 1 # print first iteration
  @printf "Covariance:\n"
  print_covariance(problem.adaptive_update,adaptive_update_params, 1)

  # first iteration
  Theta[:,1] = theta_0
  loglik[1] = pf_paralell(Z, Theta[:,1],theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc, loglik_vec)

  if alg == "MCWM"
    # do nothing
  else
    loglik_gp_old = predict(Theta[:,1], gp, pred_method,est_method,noisy_est)[1]
  end

  # print start loglik
  @printf "Loglik start: %.4f \n" loglik[1]

  tic()
  for r = 2:R

    # set print_on to false, only print each print_interval:th iteration
    print_on = false

    # print acceptance rate for the last print_interval iterations
    if mod(r-1,print_interval) == 0
      # print percentage done
      @printf "Percentage done: %.2f %% \n" 100*(r-1)/R
      print_on = true # print ESS and Nbr resample each print_interval:th iteration
      # print accaptance rate
      @printf "Acceptance rate on iteration %d to %d is %.4f\n" r-print_interval r-1  sum(accept_vec[r-print_interval:r-1])/( r-1 - (r-print_interval) )
      # print covariance function
      @printf "Covariance:\n"
      print_covariance(problem.adaptive_update,adaptive_update_params, r)
      # print loglik
      @printf "Loglik: %.8f \n" loglik[r-1]
      # print log-lik vector
      @printf "Loglik values on different cores:\n"
      println(loglik_vec)
    end

    MH_direct = rand() < beta_MH # we always run the early-rejection scheme

    if MH_direct

      # secound stage direct

      nbr_ordinary_mh +=  1

      # Gaussian random walk using secound stage direct kernel
      (theta_star, ) = gaussian_random_walk(kernel_MH_direct, adaptive_update_params_MH_direct, Theta[:,r-1], r)

      # calc loglik using proposed parameters
      loglik_star = pf_paralell(Z, theta_star,theta_known, N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc,loglik_vec)

      # run MCWM or PMCMC
      if alg == "MCWM"
        loglik_current =  pf_paralell(Z, Theta[:,r-1],theta_known, N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false, false, nbr_of_proc,loglik_vec)
      else
        loglik_current = loglik[r-1]
      end

      nbr_eval_pf += 1

      prior_log_star = evaluate_prior(theta_star,prior_parameters, dist_type)
      prior_log_old = evaluate_prior(Theta[:,r-1],prior_parameters, dist_type)

      jacobian_log_star = jacobian(theta_star)
      jacobian_log_old = jacobian(Theta[:,r-1])

      a_log = loglik_star + prior_log_star +  jacobian_log_star - (loglik_current +  prior_log_old + jacobian_log_old)

      accept = log(rand()) < a_log # calc accaptace decision

      if accept # the proposal is accapted
        nbr_ordinary_mh_accapte += 1
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

      prior_log_star = evaluate_prior(theta_star,prior_parameters,dist_type)
      prior_log = evaluate_prior(Theta[:,r-1],prior_parameters,dist_type)

      prior_log_star = evaluate_prior(theta_star,prior_parameters,dist_type)
      prior_log_old = evaluate_prior(Theta[:,r-1],prior_parameters,dist_type)

      jacobian_log_star = jacobian(theta_star)
      jacobian_log_old = jacobian(Theta[:,r-1])

      # should we recompute the loglik_gp_old value here?
      # we currently recompute loglik_gp_old here!

      if alg == "MCWM"
        loglik_gp_old = predict(Theta[:,r-1], gp, pred_method,est_method,noisy_est)[1]
      else
      end

      a_gp = loglik_gp_new + prior_log_star +  jacobian_log_star -  (loglik_gp_old - prior_log_old - jacobian_log_old)

      accept = log(rand()) < a_gp # calc accept

      if !accept
        # keep old values
        nbr_early_rejections = nbr_early_rejections + 1
        Theta[:,r] = Theta[:,r-1]
        loglik[r] = loglik[r-1]
      else

        # run pf

        nbr_second_stage += 1

        loglik_star = pf_paralell(Z, theta_star,theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc,loglik_vec)

        if alg == "MCWM"
          loglik_current =  pf_paralell(Z, Theta[:,r-1],theta_known, N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false, false, nbr_of_proc,loglik_vec)
        else
          loglik_current = loglik[r-1]
        end

        nbr_eval_pf += 1
        nbr_eval_pf_secound_stage += 1

        a_log = (loglik_star + loglik_gp_old)  -  (loglik_current + loglik_gp_new)

        accept = log(rand()) < a_log # calc accaptance decision

        if accept # the proposal is accapted
          nbr_second_stage_accepted = nbr_second_stage_accepted+1
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
  @printf "Particle filter: %s\n" pf_alg
  @printf "Time pre-er:  %.0f\n" time_pre_er
  @printf "Time fit GP model:  %.0f\n" time_fit_gp
  @printf "Time er-part:  %.0f\n" time_da_part

  @printf "Number early-rejections: %.d\n"  nbr_early_rejections

  @printf "Number of cases directly to ordinary MH: %d\n"  nbr_ordinary_mh
  @printf "Number of cases directly to ordinary MH accepted: %d\n"  nbr_ordinary_mh_accapte

  @printf "Number cases in in DA part: %d\n"  nbr_run_DA
  @printf "Number cases in second stage: %d\n"  nbr_second_stage
  @printf "Number accepted in second stage: %d\n"  nbr_second_stage_accepted

  @printf "Total number of evaluations of the particle filter: %d\n" nbr_eval_pf

  @printf "Acceptance rate for ordinary MH accepted: %.4f\n"  nbr_ordinary_mh_accapte/nbr_ordinary_mh*100
  @printf "Acceptance rate a_1: %.4f\n"  nbr_second_stage/nbr_run_DA*100
  @printf "Acceptance rate a_2: %.4f\n"  nbr_second_stage_accepted/nbr_second_stage*100
  @printf "Acceptance rate a_1*a_2: %.4f\n"  (nbr_second_stage/nbr_run_DA)*(nbr_second_stage_accepted/nbr_second_stage)*100
  @printf "Acceptance rate a (for entier algorithm): %.4f\n" sum(accept_vec)/R*100

  @printf "#####################################################################\n"


  # return resutls
  if return_run_info
    run_info = [nbr_eval_pf;
                nbr_eval_pf_secound_stage;
                nbr_second_stage;
                nbr_ordinary_mh]
    return return_gp_results(gp, Theta,loglik,accept_vec,prior_vec, compare_GP_PF, data_gp_pf,nbr_early_rejections, problem, adaptive_update_params,accept_prob_log,times), run_info
  else
    return return_gp_results(gp, Theta,loglik,accept_vec,prior_vec, compare_GP_PF, data_gp_pf,nbr_early_rejections, problem, adaptive_update_params,accept_prob_log,times)
  end


end


# ADA-GP_MCMC

doc"""
    problem_traning::Problem, problem::gpProblem, gp::GPModel, casemodel::CaseModel, cov_matrix::Matrix)

Runs the ADA-GP-MCMC algorithm.
"""
function adagpmcmc(problem_traning::Problem, problem::gpProblem, gp::GPModel, casemodel::CaseModel, cov_matrix::Matrix, return_run_info::Bool=false)

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
  lasso = problem.alg_param.lasso # use Lasso
  beta_MH = problem.alg_param.beta_MH

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
  #compare_GP_PF = zeros(2,R-length_training_data-burn_in)
  compare_GP_PF = zeros(2,R)

  #data_gp_pf = zeros(length(theta_0)+2,R-length_training_data-burn_in)
  data_gp_pf = zeros(length(theta_0)+2,R)
  data_training = zeros(1+length(theta_0), length_training_data)
  #accept_prob_log = zeros(2, R-length_training_data-burn_in) # [gp ; pf]
  accept_prob_log = zeros(2, R) # [gp ; pf]

  loglik_star = zero(Float64)
  loglik_gp = zeros(Float64)
  loglik_gp_old = zero(Float64)
  loglik_gp_new = zero(Float64)
  index_keep_gp_er = zero(Int64)
  nbr_early_rejections = zero(Int64)
  accept = true
  MH_direct = false
  nbr_ordinary_mh = 0
  nbr_ordinary_mh_accapte = 0
  nbr_run_DA = 0
  nbr_split_accaptance_region_early_accept = 0
  nbr_split_accaptance_region_early_reject = 0
  nbr_second_stage_accepted = 0
  nbr_second_stage = 0
  loglik_list = []
  a_log = zero(Float64)

  # starting values for times:
  time_pre_er = zero(Float64)
  time_fit_gp = zero(Float64)
  time_er_part = zero(Float64)

  nbr_case_1 = zero(Int64)
  nbr_case_2 = zero(Int64)
  nbr_case_3 = zero(Int64)
  nbr_case_4 = zero(Int64)

  nbr_eval_pf = zero(Int64)
  nbr_case_13 = zero(Int64)
  nbr_case_24 = zero(Int64)

  nbr_case_pf_1 = zero(Int64)
  nbr_case_pf_2 = zero(Int64)
  nbr_case_pf_3 = zero(Int64)
  nbr_case_pf_4 = zero(Int64)

  # da new
  loglik_gp_new_std = 0

  #(~,std_loglik_training) = predict(theta_training, gp, pred_method,est_method,noisy_est,true)
  loglik_gp_new_std = 0

  # parameters for prior dist
  dist_type = problem.prior_dist.dist
  prior_parameters = problem.prior_dist.prior_parameters

  # prop kernl for ADA-GP-MCMC
  xi = 1.2
  problem.adaptive_update = noAdaptation(xi^2*cov_matrix)

  adaptive_update_params = set_adaptive_alg_params(problem.adaptive_update, length(theta_0),Theta[:,1], R)

  # prop kernl for MH_direct
  kernel_MH_direct = noAdaptation(cov_matrix)
  adaptive_update_params_MH_direct = set_adaptive_alg_params(kernel_MH_direct, length(theta_0),Theta[:,1], R)


  @printf "#####################################################################\n"

  # print information at start of algorithm
  @printf "Starting ADA-GP-MCMC with adaptive RW estimating %d parameters\n" length(theta_true)
  @printf "MCMC algorithm: %s\n" alg
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Select case model: %s\n" typeof(casemodel)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist
  @printf "Particle filter: %s, on %d cores\n" pf_alg nbr_of_cores
  @printf "Nbr particles for particle filter: %d\n" N

  @printf "Covariance - kernel_MH_direct:\n"
  print_covariance(kernel_MH_direct,adaptive_update_params_MH_direct, 1)

  # first iteration
  @printf "Iteration: %d\n" 1 # print first iteration
  @printf "Covariance:\n"
  print_covariance(problem.adaptive_update,adaptive_update_params, 1)

  # set nbr of cores to use for parallel pf
  nbr_of_proc = set_nbr_cores(nbr_of_cores, pf_alg)
  #nbr_of_proc = nbr_of_cores
  loglik_vec = SharedArray{Float64}(nbr_of_proc)


  # print acceptance rate each print_interval:th iteration
  print_interval = 1000

  # first iteration
  @printf "Iteration: %d\n" 1 # print first iteration
  @printf "Covariance:\n"
  print_covariance(problem.adaptive_update,adaptive_update_params, 1)

  Theta[:,1] = theta_0
  loglik[1] = pf_paralell(Z, Theta[:,1],theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc,loglik_vec)

  # print start loglik
  @printf "Loglik: %.4f \n" loglik[1]

  tic()
  for r = 2:R

    # set print_on to false, only print each print_interval:th iteration
    print_on = false

    # print acceptance rate for the last print_interval iterations
    if mod(r-1,print_interval) == 0
      # print percentage done
      @printf "Percentage done: %.2f %% \n" 100*(r-1)/R
      print_on = true # print ESS and Nbr resample each print_interval:th iteration
      # print accaptance rate
      @printf "Acceptance rate on iteration %d to %d is %.4f\n" r-print_interval r-1  sum(accept_vec[r-print_interval:r-1])/( r-1 - (r-print_interval) )
      # print covariance function
      @printf "Covariance:\n"
      print_covariance(problem.adaptive_update,adaptive_update_params, r)
      # print loglik
      @printf "Loglik: %.4f \n" loglik[r-1]
      @printf "Loglik values on different cores:\n"
      println(loglik_vec)
    end

    MH_direct = rand() < beta_MH # we always run the early-rejection scheme

    if MH_direct

      # secound stage direct
      nbr_ordinary_mh += 1

      # Gaussian random walk using secound stage direct kernel
      (theta_star, ) = gaussian_random_walk(kernel_MH_direct, adaptive_update_params_MH_direct, Theta[:,r-1], r)

      # calc loglik using proposed parameters
      loglik_star = pf_paralell(Z, theta_star,theta_known, N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc,loglik_vec)

      # run MCWM or PMCMC
      if alg == "MCWM"
        loglik_current =  pf_paralell(Z, Theta[:,r-1],theta_known, N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false, false, nbr_of_proc,loglik_vec)
      else
        loglik_current = loglik[r-1]
      end

      nbr_eval_pf += 1

      prior_log_star = evaluate_prior(theta_star,prior_parameters, dist_type)
      prior_log_old = evaluate_prior(Theta[:,r-1],prior_parameters, dist_type)

      jacobian_log_star = jacobian(theta_star)
      jacobian_log_old = jacobian(Theta[:,r-1])

      a_log = loglik_star + prior_log_star +  jacobian_log_star - (loglik_current +  prior_log_old + jacobian_log_old)

      accept = log(rand()) < a_log # calc accaptace decision

      if accept # the proposal is accapted
        nbr_ordinary_mh_accapte += 1
        Theta[:,r] = theta_star # update chain with new values
        loglik[r] = loglik_star
        accept_vec[r] = 1
      else
        Theta[:,r] = Theta[:,r-1] # keep old values
        loglik[r] = loglik[r-1]
      end

    else

      # stage 1
      nbr_run_DA += 1
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
      #accept_prob_log[1, r] = a_gp

      if !accept # early-accept

        # keep old values
        nbr_early_rejections = nbr_early_rejections + 1
        Theta[:,r] = Theta[:,r-1]
        loglik[r] = loglik[r-1]
        # adaptation of covaraince matrix for the proposal distribution
        # adaptation(problem.adaptive_update, adaptive_update_params, Theta, r,a_gp)

      else

        # stage 2 usign A-DA

        nbr_second_stage += 1

        # A-DA
        u_log_hat = log(rand())

        if loglik_gp_old < loglik_gp_new #&& loglik_gp_new_std < std_limit && loglik_gp_old_std < std_limit

          nbr_case_13 += 1

          # select case 1 or 3
          if selectcase1or3(casemodel, theta_star, loglik_gp_new, loglik_gp_old) == 1

            # case 1

            nbr_case_1 += 1

            if u_log_hat < loglik_gp_old - loglik_gp_new
              nbr_second_stage_accepted = nbr_second_stage_accepted+1
              nbr_split_accaptance_region_early_accept = nbr_split_accaptance_region_early_accept+1
              Theta[:,r] = theta_star # update chain with proposal
              loglik[r] = NaN
              accept_vec[r] = 1
            else

              loglik_star = pf_paralell(Z, theta_star,theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc,loglik_vec)
              loglik_old = pf_paralell(Z, Theta[:,r-1],theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc,loglik_vec)

              nbr_eval_pf += 1
              nbr_case_pf_1 += 1

              a_log = (loglik_star + loglik_gp_old)  -  (loglik_old + loglik_gp_new)
              accept = u_log_hat < a_log # calc accaptance decision


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

            nbr_case_3 += 1

            # run case 3
            if u_log_hat > loglik_gp_old - loglik_gp_new

              nbr_split_accaptance_region_early_reject = nbr_split_accaptance_region_early_reject+1
              Theta[:,r] = Theta[:,r-1] # keep old values
              loglik[r] = loglik[r-1]
              #accept_vec[r] = 1

            else

              loglik_star = pf_paralell(Z, theta_star,theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc,loglik_vec)
              loglik_old = pf_paralell(Z, Theta[:,r-1],theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc,loglik_vec)

              nbr_eval_pf += 1
              nbr_case_pf_3 += 1

              a_log = (loglik_star + loglik_gp_old)  -  (loglik_old + loglik_gp_new)
              accept = u_log_hat < a_log # calc accaptance decision

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
          nbr_case_24 += 1

          if selectcase2or4(casemodel, theta_star, loglik_gp_new, loglik_gp_old) == 1

            # case 2
            nbr_case_2 += 1


            loglik_star = pf_paralell(Z, theta_star,theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc,loglik_vec)
            loglik_old = pf_paralell(Z, Theta[:,r-1],theta_known,N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc,loglik_vec)

            nbr_eval_pf += 1
            nbr_case_pf_2 += 1

            a_log = (loglik_star + loglik_gp_old)  -  (loglik_old + loglik_gp_new)
            accept = u_log_hat < a_log # calc accaptance decision


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
           nbr_case_4 += 1

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

  @printf "Number split acceptance region early-accept: %d\n"  nbr_split_accaptance_region_early_accept
  @printf "Number split acceptance region early-reject: %d\n"  nbr_split_accaptance_region_early_reject

  @printf "Number cases in DA part: %d\n"  nbr_run_DA
  @printf "Number cases in second stage: %d\n"  nbr_second_stage
  @printf "Number accepted in second stage: %d\n"  nbr_second_stage_accepted

  @printf "Total number of evaluations of the particle filter: %d\n" nbr_eval_pf

  @printf "Acceptance rate for ordinary MH accepted: %.4f\n"  nbr_ordinary_mh_accapte/nbr_ordinary_mh*100
  @printf "Acceptance rate a_1: %.4f\n"  nbr_second_stage/nbr_run_DA*100
  @printf "Acceptance rate a_2: %.4f\n"  nbr_second_stage_accepted/nbr_second_stage*100
  @printf "Acceptance rate a_1*a_2: %.4f\n"  (nbr_second_stage/nbr_run_DA)*(nbr_second_stage_accepted/nbr_second_stage)*100
  @printf "Acceptance rate a (for entier algorithm): %.4f\n" sum(accept_vec)/R*100

  @printf "Number case 1 or 3: %d\n"  nbr_case_13
  @printf "Number case 2 or 4: %d\n"  nbr_case_24

  @printf "Number case 1: %d, %.4f %% of all cases\n"  nbr_case_1 nbr_case_1/nbr_case_13
  @printf "Number case 2: %d, %.4f %% of all cases\n"  nbr_case_2 nbr_case_2/nbr_case_24
  @printf "Number case 3: %d, %.4f %% of all cases\n"  nbr_case_3 nbr_case_3/nbr_case_13
  @printf "Number case 4: %d, %.4f %% of all cases\n"  nbr_case_4 nbr_case_4/nbr_case_24


  @printf "Number pf runs in case 1: %d, prob pf given case 1 %.4f %%\n"  nbr_case_pf_1 nbr_case_pf_1/nbr_case_1*100
  @printf "Number pf runs in case 2: %d, prob pf given case 2 %.4f %%\n"  nbr_case_pf_2 nbr_case_pf_2/nbr_case_2*100
  @printf "Number pf runs in case 3: %d, prob pf given case 3 %.4f %%\n"  nbr_case_pf_3 nbr_case_pf_3/nbr_case_3*100
  @printf "Number pf runs in case 4: %d, prob pf given case 4 %.4f %%\n"  nbr_case_pf_4 nbr_case_pf_4/nbr_case_4*100


  @printf "#####################################################################\n"

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
    return return_gp_results(gp, Theta,loglik,accept_vec,prior_vec, compare_GP_PF, data_gp_pf,nbr_early_rejections, problem, adaptive_update_params,accept_prob_log,times), run_info
  else
    return return_gp_results(gp, Theta,loglik,accept_vec,prior_vec, compare_GP_PF, data_gp_pf,nbr_early_rejections, problem, adaptive_update_params,accept_prob_log,times)
  end

end


################################################################################
######               help functions                                        #####
################################################################################

doc"""
    return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params)

Constructs the return type of the resutls from the PMCMC and MCWM algorithm.
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
    return_gp_results(gp, Theta,loglik,accept_vec,prior_vec, compare_GP_PF, data_gp_pf,nbr_early_rejections, problem, adaptive_update_params,accept_prob_log,times)

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
    set_nbr_cores(nbr_of_cores::Int64, pf_alg::String)

Sets the number of cores to use.
"""
function set_nbr_cores(nbr_of_cores::Int64, pf_alg::String)
  if length(workers()) == 1
    nbr_of_proc = nbr_of_cores
    addprocs(nbr_of_proc)
    if pf_alg == "parallel_bootstrap"
      @everywhere include("run_pf_paralell.jl")
    else
      error("The auxiliary particle filter is not implemented.")
    end
  else
    nbr_of_proc = length(workers())
  end
  return nbr_of_proc
end

doc"""
    evaluate_prior(theta_star, prior_parameters, dist_type)

Calculates the `log-prior` value for the prior distribution for the parameters `theta_star`.
"""
function  evaluate_prior(theta_star, prior_parameters, dist_type)

  # set start value for loglik
  log_prior = 0.

  if dist_type == "Uniform"
    for i = 1:length(theta_star)
      # Update loglik, i.e. add the loglik for each model paramter in theta
      log_prior = log_prior + log_unifpdf( theta_star[i], prior_parameters[i,1], prior_parameters[i,2] )
    end
  elseif dist_type == "Normal"
    for i = 1:length(theta_star)
      # Update loglik, i.e. add the loglik for each model paramter in theta
      log_prior = log_prior + log_normpdf(theta_star[i],prior_parameters[i,1],prior_parameters[i,2])
    end
  elseif dist_type == "nonlog"
    # add code to handle priors on non-log-scale!
    if length(theta_star) == 2
      for i = 1:length(theta_star)
        # the unknown parameters c and d both have normal prior dists
        log_prior = log_prior + log_normpdf( exp(theta_star[i]), prior_parameters[i,1], prior_parameters[i,2] )
      end
    elseif length(theta_star) == 3
      # the unknown parameter A has a inv-gamma prior dist
      log_prior = log_prior + log_invgampdf(exp(theta_star[1]), prior_parameters[1,1], prior_parameters[1,2])
      for i = 2:3
        # The unknown parameters c and d both have normal prior dists
        log_prior = log_prior + log_normpdf( exp(theta_star[i]), prior_parameters[i,1], prior_parameters[i,2] )
      end
    elseif length(theta_star) == 5
      for i in [1 2 5]
        # The unknown parameters Κ,Γ and sigma both have gamma prior dists
        log_prior = log_prior + log_gampdf( exp(theta_star[i]), prior_parameters[i,1], prior_parameters[i,2] )
      end
      for i = 3:4
        # The unknown parameters c and d both have normal prior dists
        log_prior = log_prior + log_normpdf( exp(theta_star[i]), prior_parameters[i,1], prior_parameters[i,2] )
      end
    elseif length(theta_star) == 6
      # The unknown parameter A has a gaminv prior dist
      log_prior = log_prior + log_invgampdf(exp(theta_star[1]), prior_parameters[1,1], prior_parameters[1,2])
      for i = 2:3
        # The unknown parameters c and d both have normal prior dists
        log_prior = log_prior + log_normpdf( exp(theta_star[i]), prior_parameters[i,1], prior_parameters[i,2] )
      end
      for i = 4:length(theta_star)
        # The unknown parameters p1, p2 and sigma both have gamma prior dists
        log_prior = log_prior + log_gampdf( exp(theta_star[i]), prior_parameters[i,1], prior_parameters[i,2] )
      end
    elseif length(theta_star) == 7
      for i = [1,2,5,6,7]
        # The unknown parameters Κ,Γ,power1,power2 and sigma all have gamma prior dists
        log_prior = log_prior + log_gampdf( exp(theta_star[i]), prior_parameters[i,1], prior_parameters[i,2] )
      end
      for i = [3,4]
        # The unknown parameters c and d have normal prior dists
        log_prior = log_prior + log_normpdf( exp(theta_star[i]), prior_parameters[i,1], prior_parameters[i,2] )
      end

    elseif length(theta_star) == 8
      # the unknown parameter A has a inv-gamma prior dist
      log_prior = log_prior + log_invgampdf(exp(theta_star[3]), prior_parameters[3,1], prior_parameters[3,2])
      for i = [1,2,5,6,7]
        # The unknown parameters Κ,Γ,power1,power2 and sigma all have gamma prior dists
        log_prior = log_prior + log_gampdf( exp(theta_star[i]), prior_parameters[i,1], prior_parameters[i,2] )
      end
      for i = [3,4]
        # The unknown parameters c and d have normal prior dists
        log_prior = log_prior + log_normpdf( exp(theta_star[i]), prior_parameters[i,1], prior_parameters[i,2] )
      end
    elseif length(theta_star) == 4
      for i = 1:2
        # The unknown parameters Κ and Γ both have gammma prior dists
        log_prior = log_prior + log_invgampdf( exp(theta_star[i]), prior_parameters[i,1], prior_parameters[i,2] )
      end
      for i = 3:4
        # The unknown parameters c and d both have normal prior dists
        log_prior = log_prior + log_normpdf( exp(theta_star[i]), prior_parameters[i,1], prior_parameters[i,2] )
      end

    else
      for i in [3 6]
        # The unknown parameters A and g have a gaminv prior dist
        log_prior = log_prior + log_invgampdf(exp(theta_star[i]), prior_parameters[i,1], prior_parameters[i,2])
      end
      for i in [4 5]
        # The unknown parameters c and d both have normal prior dists
        log_prior = log_prior + log_normpdf( exp(theta_star[i]), prior_parameters[i,1], prior_parameters[i,2] )
      end
      for i in [1 2 7 8 9]
        # The unknown parameters Κ,Γ,power1,power2 and sigma all have gamma prior dists
        log_prior = log_prior + log_gampdf( exp(theta_star[i]), prior_parameters[i,1], prior_parameters[i,2] )
      end

    end

  end

  return log_prior # return log_lik

end



doc"""
    jacobian(theta::Vector, parameter_transformation::String)

Returnes log-Jacobian for transformation of proposal space.
"""
function jacobian(theta::Vector)

  return sum(theta)

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
    set_parameters(theta::Array{Float64}, theta_known::Array{Float64}, nbr_of_unknown_parameters::Int64)

Sets the model parameters according to the nbr of unknown parameters. Help function for pf.

"""
function set_parameters(theta::Array{Float64}, theta_known::Array{Float64}, nbr_of_unknown_parameters::Int64)

  # set parameter values
  if nbr_of_unknown_parameters == 8
    # estiamte Κ, Γ,A,c,d,power1,power2,sigma
    (Κ,Γ,A,c,d,power1,power2,sigma) = exp.(theta)
    (A_sign,B,f,g) = theta_known
  elseif nbr_of_unknown_parameters == 7
    # estimate Κ,Γ,c,d,power1,power2,sigma
    (Κ,Γ,c,d,power1,power2,sigma) = exp.(theta)
    (A,A_sign,B,f,g) = theta_known
  elseif nbr_of_unknown_parameters == 6
    # estiamte A,c,d,power1,power2,sigma
    (A,c,d,power1,power2,sigma) = exp.(theta)
    (Κ, Γ, A_sign,B,f,g) = theta_known
  elseif nbr_of_unknown_parameters == 5
    # estimate Κ Γ c d sigma
    (Κ,  Γ, c, d, sigma) = exp.(theta)
    (A,A_sign, B, f, g, power1, power2) = theta_known
  elseif nbr_of_unknown_parameters == 4
    # estimate Κ Γ c d
    (Κ, Γ,c,d) = exp.(theta)
    (A,A_sign,B,f,g,power1,power2, sigma) = theta_known
  elseif nbr_of_unknown_parameters == 2
    # estimating c and d
    (c,d) = exp.(theta)
    (Κ, Γ, A, A_sign, B, f, g, power1, power2, sigma) = theta_known
  elseif nbr_of_unknown_parameters == 3
    # estimate A,c,d
    (A, c, d) = exp.(theta)
    (Κ, Γ, A_sign, B, f, g, power1, power2, sigma) = theta_known
  else
    # estimate all parameters. i.e. estimate Κ, Γ,A,c,d,g,power1,power2,sigma
    (Κ, Γ, A,c,d,g,power1,power2,sigma) = exp.(theta)
    (A_sign,B,f) = theta_known
  end

  return (Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma)

end


################################################################################
###            Functions for particle filter diagnostics                    ####
################################################################################


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

    error("The auxiliary particle filter is not implemented.")

  end

  return loglik_vec, weigths, particels

end
