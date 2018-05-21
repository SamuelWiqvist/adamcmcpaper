# run gp_regrssion to construct the GP model

include("rickermodel.jl")
using JLD
using HDF5


# set up problem
problem = set_up_gp_problem(ploton = false)


# use AM alg for adaptive updating
problem.adaptive_update = AMUpdate(eye(3), 2.4/sqrt(3), 1., 0.7, 50)

problem.adaptive_update = noAdaptation(2.4/sqrt(3)*eye(3))

# or, use AM gen alg for adaptive updating
problem.adaptive_update = AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.3, 1., 0.8, 25)

# set algorithm parameters
problem.alg_param.N = 1000
problem.alg_param.alg = "PMCMC"
problem.alg_param.compare_GP_and_PF = false
problem.alg_param.noisy_est = false
problem.alg_param.pred_method = "sample"
problem.alg_param.R = 10000
problem.alg_param.burn_in = 0
problem.alg_param.length_training_data = 2000
problem.alg_param.nbr_predictions = 1
problem.alg_param.print_interval = 1000
problem.alg_param.selection_method = "max_loglik"  # "local_loglik_approx" # "max_loglik"
problem.alg_param.beta_MH = 0.2 # "local_loglik_approx" # "max_loglik"

#problem.data.y = Array(readtable("y.csv"))[:,1]
#problem.data.y = Array(readtable("y_data_set_1.csv"))[:,1]
problem.data.y = Array(readtable("y_data_set_2.csv"))[:,1]

accelerated_da = false


# set up traning problem


# set up problem
problem_traning = set_up_problem(ploton = false)

problem_traning.alg_param.N = 1000
problem_traning.alg_param.R = 4000
problem_traning.alg_param.burn_in = 2000
problem_traning.data.y = Array(readtable("y_data_set_2.csv"))[:,1] #Array(readtable("y.csv"))[:,1]
problem_traning.alg_param.print_interval = 1000

# test starting at true parameters
#problem.model_param.theta_0 = problem.model_param.theta_true

# PMCMC
problem_traning.alg_param.alg = "PMCMC"


# use AM alg for adaptive updating
#problem.adaptive_update = AMUpdate(eye(3), 2.4/sqrt(3), 1., 0.7, 25)

#problem.adaptive_update = noAdaptation(2.4/sqrt(3)*eye(3))

# or, use AM gen alg for adaptive updating
problem_traning.adaptive_update = AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.2, 1., 0.8, 25)


# run adaptive PMCMC
res, res_traning, theta_training, loglik_training = dagpMCMC(problem_traning, problem, accelerated_da)

mcmc_results = Result(res[1].Theta_est, res[1].loglik_est, res[1].accept_vec, res[1].prior_vec)

y = problem.data.y
theta = problem.model_param.theta_true
theta_known = problem.model_param.theta_known
N = 1000
gp_test = res[2]
noisy_est = false
theta_star = theta_training
loglik_gp = zeros(size(theta_star,2))
loglik_pf = zeros(size(theta_star,2))

#@time predict(theta_star[:,1], gp_ml, pred_method,est_method,noisy_est)
@time prediction(theta_star[:,1], gp_test, noisy_est)


@time pf(y,theta,theta_known,N,false,false)



tic()
for i = 1:1000; prediction(theta_star[:,1], gp_test, noisy_est);end
time_gp_model = toc()

tic()
for i = 1:1000; pf(y,theta_star[:,1],theta_known,N,false,false);end
time_pf = toc()


@printf "Time GP:  %.4f\n" time_gp_model
@printf "Time PF:  %.4f\n" time_pf


Profile.clear()
#@profile predict(theta_star[:,1], gp_ml, pred_method,est_method,noisy_est)
@profile (for i = 1:100; prediction(theta_star[:,1], gp_test, noisy_est);end)
r = Profile.retrieve();
save("profilegp.jld", "data", r)


Profile.clear()
@profile (for i = 1:100;pf(y,theta_star[:,1],theta_known,N,false,false);end)
r = Profile.retrieve();
#save("profilepf.jld", "profiledata", r)
save("profilepf.jld", "data", r)


#=
tic()
for i = 1:size(theta_star,2)
  (~,~,loglik_gp[i]) = prediction(theta_star[:,i], gp_test, noisy_est)
  #loglik_gp[i] = predict(theta_star[:,i], gp_ml, pred_method,est_method,noisy_est)[1]
end
time_gp = toc()
# \sim 7 sec

tic()
for i = 1:size(theta_star,2)
  loglik_pf[i] = pf(y, theta_star[:,i],theta_known,N)
end
time_pf = toc()
# ∼ 380 sec


bins = 100
PyPlot.figure()
h1 = PyPlot.plt[:hist](loglik_pf,bins)
PyPlot.figure()
h2 = PyPlot.plt[:hist](loglik_gp,bins)

PyPlot.figure()
PyPlot.plot(h1[2][1:end-1],h1[1], "b")
PyPlot.hold(true)
PyPlot.plot(h2[2][1:end-1],h2[1], "r")

=#
# red pf
# blue gp

#=
PyPlot.figure()
PyPlot.scatter3D(theta_star[1,:],theta_star[2,:],loglik_gp, color = "blue")
PyPlot.hold(true)
PyPlot.scatter3D(theta_star[1,:],theta_star[2,:],loglik_pf, color = "red")
PyPlot.title("Particle filter (red), Gaussian process model (blue)")
=#

# comments

# usign a data set with 50 observations:
# time_gp = 5.89
# time_pf = 8.96

# usign a data set with 200 observations:
# time_gp = 6.03
# time_pf = 37.3
