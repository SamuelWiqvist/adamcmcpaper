# this file contains code to run the PCMCM and MCMC algorithm using the bootstrap
# particel filter as well as comparing the boostrap filter and the auxiliar particel filter

# load functions and packages
include("set_up.jl")

# set parameters
nbr_iterations = 1000 # nbr of iteration of the MC algorithm
nbr_particels = 25  # nbr of particel on each core
burn_in = 500 # nbr of iterations for the burn in

# create a problem type where we estimate twp parameters and use the AM aglorithm for
# the adaptive updating
problem = set_up_problem(nbr_of_unknown_parameters=2)
problem.alg_param.nbr_of_cores = 4

problem.alg_param.R = nbr_iterations
problem.alg_param.N = nbr_particels
problem.alg_param.burn_in = burn_in
problem.adaptive_update =  AMUpdate(eye(2), 2.4/sqrt(2), 2, 0.5, 50) # 2.38/sqrt(2)


# run PCMCM
res_problem_PCMCM = @time PMCMC(problem)

# export results into the files cvs files
export_data(problem, res_problem_PCMCM)

# run the matlab script analyse_results.m to plot the results

# run MCWH
res_problem_MCWH = @time MCWM(problem)

# export results into the files cvs files
export_data(problem, res_problem_MCWH)

# run the matlab script analyse_results.m to plot the results

# run  the bootstrap filter and the auxiliary particle filter
pf_apf = compare_pf_apf(problem)

# compute mean and variance
mean_pf_apf = mean(pf_apf,1)
std_pf_apf = std(pf_apf,1)
