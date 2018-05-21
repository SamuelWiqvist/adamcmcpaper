# load functions and packages
include("set_up.jl")
include("run_pf_paralell.jl")
# set parameters
nbr_iterations = 1000 # nbr of iteration of the MC algorithm
nbr_particels = 25  # nbr of particel on each core
burn_in = 500 # nbr of iterations for the burn in

# create a problem type where we estimate twp parameters and use the AM aglorithm for
# the adaptive updating
data_set = "new"
problem = set_up_problem(nbr_of_unknown_parameters=3,data_set=data_set)
problem.alg_param.nbr_of_cores = 4

problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.adaptive_update =  AMUpdate(eye(2), 2.4/sqrt(2), 2, 0.5, 50) # 2.38/sqrt(2)

Z = problem.data.Z
N = 5000
R = problem.alg_param.R
burn_in = problem.alg_param.burn_in
pf_alg = problem.alg_param.pf_alg
nbr_of_cores = problem.alg_param.nbr_of_cores
nbr_x0  = problem.alg_param.nbr_x0
nbr_x = problem.alg_param.nbr_x
subsample_interval = problem.alg_param.subsample_int
dt = 0.5 #problem.alg_param.dt # 0.5 if data new
dt_U = 1. #problem.alg_param.dt # 1 if data new

print_on = true
return_weigths_and_particels = true
# model parameters
theta_true = problem.model_param.theta_true
theta_known = problem.model_param.theta_known
theta_0 = problem.model_param.theta_0

# run pf at theta_true
theta = theta_true
theta = [log(0.1) log(100) log(40)]

nbr_parallel = 1
nbr_itr_pf = 100*nbr_parallel
loglik_vector = zeros(nbr_itr_pf)

N = 25

(\u039a, \u0393, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, theta_known,length(theta))

A = A*A_sign

# set value for constant b function
b_const = sqrt(2.*sigma^2 / 2.)

# set values needed for calculations in Float64
(subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (subsample_interval, nbr_x0, nbr_x,N))

(loglik, w, x) = @time run_pf_paralell(Z,theta,theta_known, N, N_calc, dt, dt_U, nbr_x0, nbr_x0_calc, nbr_x, nbr_x_calc, subsample_interval, subsample_interval_calc, true, return_weigths_and_particels, \u039a, \u0393, A, B, c, d, f, g, power1, power2, b_const)

for i = 1:nbr_itr_pf
  if mod(i,100) == 0
    println(i)
  end
  loglik_vector[i] = run_pf_paralell(Z,theta,theta_known, N, N_calc, dt, dt_U, nbr_x0, nbr_x0_calc, nbr_x, nbr_x_calc, subsample_interval, subsample_interval_calc, false, false, \u039a, \u0393, A, B, c, d, f, g, power1, power2, b_const)
end

std(loglik_vector)
mean(loglik_vector)

N_vec = [25 50 100 500 1000]
loglik_store = zeros(length(N_vec),nbr_itr_pf)
w_stor = []
x_stor = []

# var < 2 for N > 1000



for j = 1:length(N_vec)


  N = N_vec[j]

  println(N)


  (\u039a, \u0393, A,B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, theta_known,length(theta))

  # set value for constant b function
  b_const = sqrt(2.*sigma^2 / 2.)

  # set values needed for calculations in Float64
  (subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (subsample_interval, nbr_x0, nbr_x,N))


  (loglik, w, x) = @time run_pf_paralell(Z,theta,theta_known, N, N_calc, dt, dt_U, nbr_x0, nbr_x0_calc, nbr_x, nbr_x_calc, subsample_interval, subsample_interval_calc, print_on, return_weigths_and_particels, \u039a, \u0393, A, B, c, d, f, g, power1, power2, b_const)

  for i = 1:nbr_itr_pf
    if mod(i,100) == 0
      println(i)
    end
    loglik_vector[i] = run_pf_paralell(Z,theta,theta_known, N, N_calc, dt, dt_U, nbr_x0, nbr_x0_calc, nbr_x, nbr_x_calc, subsample_interval, subsample_interval_calc, false, false, \u039a, \u0393, A, B, c, d, f, g, power1, power2, b_const)
  end

  loglik_store[j,:] = loglik_vector

  push!(w_stor,w)
  push!(x_stor,x)


  #=
  for i = 1:nbr_parallel:nbr_itr_pf
    loglik_vector[i] = 1/nbr_parallel*sum(loglik_vector[i:i+nbr_parallel-1])
  end

  loglik_vector = loglik_vector[1:nbr_parallel:nbr_itr_pf]

  loglik_store[i,:] = loglik_vector
  =#

end


n = 5
w = w_stor[n]
x = x_stor[n]
loglik_vector = loglik_store[n,:]
loglik_vector_print = zeros(1, length(loglik_vector))

for i = 1:length(loglik_vector)
  loglik_vector_print[i] = loglik_vector[i]
end

export_pf_diagnostics(loglik_vector_print, w, x, Z, Z)


#=
res_problem_normal_prior_est_7_AM_gen = @time MCMC(problem_normal_prior_est_7_AM_gen)
res_problem_normal_prior_est_9_AM_gen = @time MCMC(problem_normal_prior_est_9_AM_gen)
