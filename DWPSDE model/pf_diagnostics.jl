# load functions and packages
try
  cd("DWPSDE model")
catch
  warn("Already in the DWPSDE model folder")
end

# load case models
cd("..")
include(pwd()*"\\select case\\selectcase.jl")
cd("DWPSDE model")


include("set_up.jl")
#include("run_pf_paralell.jl")
using PyPlot

# set parameters
nbr_iterations = 1000 # nbr of iteration of the MC algorithm
nbr_particels = 25  # nbr of particel on each core
burn_in = 500 # nbr of iterations for the burn in

# create a problem type where we estimate twp parameters and use the AM aglorithm for
# the adaptive updating
data_set = "old"
problem = set_up_problem(nbr_of_unknown_parameters=7,use_sim_data = true,data_set=data_set)
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
dt = 0.035 #problem.alg_param.dt # 0.5 if data new
dt_U = 1. #problem.alg_param.dt # 1 if data new

print_on = true
return_weigths_and_particels = false
# model parameters
theta_true = problem.model_param.theta_true
theta_known = problem.model_param.theta_known
theta_0 = problem.model_param.theta_0

# run pf at theta_true
theta = theta_true #log([1.00556;2.66301;33.8242;19.5643;1.3644;1.87421;4.10332])
#theta = [log(0.1) log(100) log(40)]

nbr_parallel = 1
nbr_itr_pf = 100*nbr_parallel
loglik_vector = zeros(100)

N = 200

(Κ, Γ, A, A_sign, B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, theta_known,length(theta))

A = A*A_sign

# set value for constant b function
b_const = sqrt(2.*sigma^2 / 2.)

# use long data
problem.data.Z = convert(Array, readtable("data_new_long.csv")[:,1])
dt = 0.5
Z = problem.data.Z



# set values needed for calculations in Float64
(subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (subsample_interval, nbr_x0, nbr_x,N))
loglik = @time run_pf_paralell(Z,theta,theta_known, N, N_calc, dt, dt_U, nbr_x0, nbr_x0_calc, nbr_x, nbr_x_calc, subsample_interval, subsample_interval_calc, true, false, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

nbr_of_proc = set_nbr_cores(4, pf_alg)
loglik_vec = SharedArray{Float64}(nbr_of_proc)

loglik = pf_paralell(Z, theta,theta_known, N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,true,false, nbr_of_proc,loglik_vec)

println(theta)
println(theta_known)
println(N)
println(dt)
println(dt_U)
println(nbr_x0)
println(nbr_x)
println(subsample_interval)
println(nbr_of_proc)
println(loglik_vec)

tic()
for i = 1:100
  if mod(i,10) == 0
    println(i)
    println(loglik_vec)
  end
  #loglik_vector[i] = run_pf_paralell(Z,theta,theta_known, N, N_calc, dt, dt_U, nbr_x0, nbr_x0_calc, nbr_x, nbr_x_calc, subsample_interval, subsample_interval_calc, false, false, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)
  loglik_vector[i] = pf_paralell(Z, theta,theta_known, N,dt,dt_U,nbr_x0, nbr_x,subsample_interval,false,false, nbr_of_proc,loglik_vec)
end
toc()

std(loglik_vector)
mean(loglik_vector)

PyPlot.figure()
h = PyPlot.plt[:hist](loglik_vector,10)


addprocs(4)
if pf_alg == "parallel_bootstrap"
  @everywhere include("run_pf_paralell.jl")
else
  @everywhere include("run_apf_paralell.jl")
end


loglik_vec = SharedArray{Float64}(4)

tic()
@parallel for i = 1:4
  loglik_vec[i] = run_pf_paralell(Z,theta,theta_known, N, N_calc, dt, dt_U, nbr_x0, nbr_x0_calc, nbr_x, nbr_x_calc, subsample_interval, subsample_interval_calc, true, false, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)
end
println(loglik_vec)
logsumexp(loglik_vec) - log(4)
toc()

tic()
for j = 1:nbr_itr_pf
  if mod(j,100) == 0
    println(j)
  end

  @sync begin

  @parallel for i = 1:4
    loglik_vec[i] = run_pf_paralell(Z,theta,theta_known, N, N_calc, dt, dt_U, nbr_x0, nbr_x0_calc, nbr_x, nbr_x_calc, subsample_interval, subsample_interval_calc, false, false, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)
  end

  end

  loglik_vector[j] = logsumexp(loglik_vec) - log(4)
end
toc()

PyPlot.figure()
h = PyPlot.plt[:hist](loglik_vector,10)


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


  #(Κ, Γ, A,B,c,d,g,f,power1,power2,sigma) = set_parameters(theta, theta_known,length(theta))

  # set value for constant b function
  #b_const = sqrt(2.*sigma^2 / 2.)

  # set values needed for calculations in Float64
  #(subsample_interval_calc, nbr_x0_calc, nbr_x_calc,N_calc) = map(Float64, (subsample_interval, nbr_x0, nbr_x,N))


  (loglik, w, x) = @time run_pf_paralell(Z,theta,theta_known, N, N_calc, dt, dt_U, nbr_x0, nbr_x0_calc, nbr_x, nbr_x_calc, subsample_interval, subsample_interval_calc, print_on, return_weigths_and_particels, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)

  for i = 1:nbr_itr_pf
    if mod(i,100) == 0
      println(i)
    end
    loglik_vector[i] = run_pf_paralell(Z,theta,theta_known, N, N_calc, dt, dt_U, nbr_x0, nbr_x0_calc, nbr_x, nbr_x_calc, subsample_interval, subsample_interval_calc, false, false, Κ, Γ, A, B, c, d, f, g, power1, power2, b_const)
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
