# this script contains code to compute the approximate log-likelihood surface
# for the Ricker model as a function of two parameters keeping the third parameter
# fixed
pwd()

# todo

# set parameters

# generate theta value over the entier prior space

# compute in parallel all loglik estimations

# fix NaN values

# plot surface

# load functions
include("rickermodel.jl")

# set up problem
problem = set_up_problem(ploton = false)


problem.data.y = Array(readtable("y_data_set_2.csv"))[:,1] #Array(readtable("y.csv"))[:,1]

y = problem.data.y
theta = problem.model_param.theta_true
theta_known = problem.model_param.theta_known
N = 1000 #problem.alg_param.N
Theta_parameters = problem.prior.Theta_parameters

# set parameter
grid_size = 100
r_hat = linspace(Theta_parameters[1,1], Theta_parameters[1,2], grid_size)
rho_hat = linspace(Theta_parameters[2,1], Theta_parameters[2,2], grid_size)
sigma = theta_true[3]

loglik_m = zeros(length(grid_size), length(grid_size))

for r in r_hat
  for rho in rho_hat
    theta = [r;rho;sigma]
    @time pf(y,theta,theta_known,N,false,false)
  end
end
