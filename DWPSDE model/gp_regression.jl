# this file containes code to run Gaussian processes regression using the GP model

# load files and functions
include("set_up.jl")

# set parameters

burn_in = 1000
length_training_data = 2000
length_test_data = 2000
nbr_iterations = burn_in + length_training_data + length_test_data
nbr_particels = 25
nbr_of_cores = 8

################################################################################
##                         set model parameters                               ##
################################################################################

# est 2 parameters

problem_normal_prior_est_2_AM_gen = set_up_problem(nbr_of_unknown_parameters=2)
problem_normal_prior_est_2_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_2_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_2_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_2_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_2_AM_gen.adaptive_update =  AMUpdate_gen(eye(2), 1/sqrt(2), 0.15, 1, 0.8, 25)

problem_nonlog_prior_est_2_AM_gen = set_up_problem(nbr_of_unknown_parameters=2, prior_dist="nonlog")
problem_nonlog_prior_est_2_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_2_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_2_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_2_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_2_AM_gen.adaptive_update =  AMUpdate_gen(eye(2), 1/sqrt(2), 0.24, 1, 0.8, 25)



# est 3 parameters

problem_normal_prior_est_3_AM_gen = set_up_problem(nbr_of_unknown_parameters=3)
problem_normal_prior_est_3_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_3_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_3_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_3_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_3_AM_gen.adaptive_update =  AMUpdate_gen(eye(3), 1/sqrt(3), 0.15, 1, 0.8, 25)

problem_nonlog_prior_est_3_AM_gen = set_up_problem(nbr_of_unknown_parameters=3, prior_dist="nonlog")
problem_nonlog_prior_est_3_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_3_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_3_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_3_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_3_AM_gen.adaptive_update =  AMUpdate_gen(eye(3), 1/sqrt(3), 0.24, 1, 0.8, 25)



# est 4 parameters

problem_normal_prior_est_4_AM_gen = set_up_problem(nbr_of_unknown_parameters=4)
problem_normal_prior_est_4_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_4_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_4_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_4_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_4_AM_gen.adaptive_update =  AMUpdate_gen(eye(4), 1/sqrt(4), 0.2, 1, 0.7, 25)

problem_nonlog_prior_est_4_AM_gen = set_up_problem(nbr_of_unknown_parameters=4, prior_dist="nonlog")
problem_nonlog_prior_est_4_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_4_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_4_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_4_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_4_AM_gen.adaptive_update =  AMUpdate_gen(eye(4), 1/sqrt(4), 0.24, 1, 0.8, 25)



# est 5 parameters

problem_normal_prior_est_5_AM_gen = set_up_problem(nbr_of_unknown_parameters=5)
problem_normal_prior_est_5_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_5_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_5_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_5_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_5_AM_gen.adaptive_update =  AMUpdate_gen(eye(5), 1/sqrt(5), 0.2, 1, 0.7, 25)

problem_nonlog_prior_est_5_AM_gen = set_up_problem(nbr_of_unknown_parameters=5, prior_dist="nonlog")
problem_nonlog_prior_est_5_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_5_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_5_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_5_AM_gen.adaptive_update =  AMUpdate_gen(eye(5), 1/sqrt(5), 0.2, 1, 0.8, 25)



# est 6 parameters

problem_normal_prior_est_6_AM_gen = set_up_problem(nbr_of_unknown_parameters=6)
problem_normal_prior_est_6_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_6_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_6_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_6_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_6_AM_gen.adaptive_update =  AMUpdate_gen(eye(6), 1/sqrt(6), 0.2, 1, 0.7, 25)

problem_nonlog_prior_est_6_AM_gen = set_up_problem(nbr_of_unknown_parameters=6, prior_dist="nonlog")
problem_nonlog_prior_est_6_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_6_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_6_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_6_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_6_AM_gen.adaptive_update =  AMUpdate_gen(eye(6), 1/sqrt(6), 0.2, 1, 0.7, 25)



# est 7 parameters

problem_normal_prior_est_7_AM_gen = set_up_problem(nbr_of_unknown_parameters=7)
problem_normal_prior_est_7_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_7_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_7_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_7_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_7_AM_gen.adaptive_update =  AMUpdate_gen(eye(7), 1/sqrt(7), 0.2, 1, 0.7, 25)

problem_nonlog_prior_est_7_AM_gen = set_up_problem(nbr_of_unknown_parameters=7, prior_dist="nonlog")
problem_nonlog_prior_est_7_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_7_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_7_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_7_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_7_AM_gen.adaptive_update =  AMUpdate_gen(eye(7), 1/sqrt(7), 0.2, 1, 0.7, 25)



# est 8 parameters

problem_normal_prior_est_8_AM_gen = set_up_problem(nbr_of_unknown_parameters=8)
problem_normal_prior_est_8_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_8_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_8_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_8_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_8_AM_gen.adaptive_update =  AMUpdate_gen(eye(8), 1/sqrt(8), 0.2, 1, 0.7, 25)

problem_nonlog_prior_est_8_AM_gen = set_up_problem(nbr_of_unknown_parameters=8, prior_dist="nonlog")
problem_nonlog_prior_est_8_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_8_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_8_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_8_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_8_AM_gen.adaptive_update =  AMUpdate_gen(eye(8), 1/sqrt(8), 0.2, 1, 0.7, 25)



# est 9 parameters

problem_normal_prior_est_9_AM_gen = set_up_problem(nbr_of_unknown_parameters=9)
problem_normal_prior_est_9_AM_gen.alg_param.R = nbr_iterations
problem_normal_prior_est_9_AM_gen.alg_param.N = nbr_particels
problem_normal_prior_est_9_AM_gen.alg_param.burn_in = burn_in
problem_normal_prior_est_9_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_normal_prior_est_9_AM_gen.adaptive_update =  AMUpdate_gen(eye(9), 1/sqrt(9), 0.2, 1, 0.7, 25)

problem_nonlog_prior_est_9_AM_gen = set_up_problem(nbr_of_unknown_parameters=9,prior_dist="nonlog")
problem_nonlog_prior_est_9_AM_gen.alg_param.R = nbr_iterations
problem_nonlog_prior_est_9_AM_gen.alg_param.N = nbr_particels
problem_nonlog_prior_est_9_AM_gen.alg_param.burn_in = burn_in
problem_nonlog_prior_est_9_AM_gen.alg_param.nbr_of_cores = nbr_of_cores
problem_nonlog_prior_est_9_AM_gen.adaptive_update =  AMUpdate_gen(eye(9), 1/sqrt(9), 0.2, 1, 0.7, 25)


################################################################################
###                       generate data                                    #####
################################################################################

(res2, data2, targets2) = @time MCMC(problem_normal_prior_est_2_AM_gen, true)
#(res, data, targets)  = @time MCMC(problem_nonlog_prior_est_2_AM_gen, true)

# est 3 parameters

#(res, data, targets) = @time MCMC(problem_normal_prior_est_3_AM_gen, true)
#(res, data, targets) = @time MCMC(problem_nonlog_prior_est_3_AM_gen, true)


# est 4 parameters

#(res, data, targets) = @time MCMC(problem_normal_prior_est_4_AM_gen, true)
#(res, data, targets) = @time MCMC(problem_nonlog_prior_est_4_AM_gen, true)

# est 5 parameters
(res5, data5, targets5) = @time MCMC(problem_normal_prior_est_5_AM_gen, true)
#(res, data, targets) = @time MCMC(problem_nonlog_prior_est_5_AM_gen, true)


# est 6 parameters
#(res, data, targets) = @time MCMC(problem_normal_prior_est_6_AM_gen, true)
#(res, data, targets) = @time MCMC(problem_nonlog_prior_est_6_AM_gen, true)


# est 7 parameters
(res7, data7, targets7) = @time MCMC(problem_normal_prior_est_7_AM_gen, true)
#(res7, data7, targets7) = @time MCMC(problem_nonlog_prior_est_7_AM_gen, true)


# est 9 (all) parameters
#(res, data, targets) = @time MCMC(problem_normal_prior_est_9_AM_gen, true)
#(res, data, targets) = @time MCMC(problem_nonlog_prior_est_9_AM_gen, true)

################################################################################
##                    Export results                                        ####
################################################################################


# export data and targets

data_export = zeros(size(data7,1)+1,size(data7,2))
data_export[1:size(data7,1),:] = data7
data_export[end,:] = targets7

writetable("data_gp_regression_7_param.csv", DataFrame(data_export))


# 2 params
export_data(problem_normal_prior_est_2_AM_gen, res[1])
export_parameters(res[2])


export_data(problem_nonlog_prior_est_2_AM_gen, res[1])
export_parameters(res[2])

# 3 params
export_data(problem_normal_prior_est_3_AM_gen, res[1])
export_parameters(res[2])

export_data(problem_nonlog_prior_est_3_AM_gen, res[1])
export_parameters(res[2])


# 4 params
export_data(problem_normal_prior_est_4_AM_gen, res[1])
export_parameters(res[2])

export_data(problem_nonlog_prior_est_4_AM_gen, res[1])
export_parameters(res[2])

# 5 params
export_data(problem_normal_prior_est_5_AM_gen, res[1])
export_parameters(res[2])

export_data(problem_nonlog_prior_est_5_AM_gen, res[1])
export_parameters(res[2])

# 6 params
export_data(problem_normal_prior_est_6_AM_gen, res[1])
export_parameters(res[2])

export_data(problem_nonlog_prior_est_6_AM_gen, res[1])
export_parameters(res[2])


# 7 params
export_data(problem_normal_prior_est_7_AM_gen, res7[1])
export_data(problem_nonlog_prior_est_7_AM_gen, res[1])


# 9 params

export_data(problem_normal_prior_est_9_AM_gen, res[1])
export_data(problem_normal_prior_est_9_AM_gen_cw, res[1])



################################################################################
##                               load data                                   ###
################################################################################

# 2 unknown parameters

# files:
# data_gp_regression_2_param
# data_gp_regression_5_param
# data_gp_regression_7_param

data_import = readtable("data_gp_regression_2_param.csv")
data_mearged = Array(data_import)
data = data_mearged[1:2,:]
targets = data_mearged[end,:]
################################################################################
##                         set traninig and test data                        ###
################################################################################

# set trainin and test data
dim = size(data,1)
training = data[:,1:length_training_data]
training_targets = targets[1:length_training_data]
test = data[:,length_training_data+1:end]
test_targets = targets[length_training_data+1:end]

trainig_data = zeros(dim+1, size(training,2))
trainig_data[1:dim,:] = training
trainig_data[end,:] = training_targets

# plot training data
PyPlot.figure()
PyPlot.plot(training')
PyPlot.figure()
PyPlot.plot(training_targets)


PyPlot.figure()
h1 = PyPlot.plt[:hist](training_targets,20, normed=true)


for i = 1:size(training,1)
  PyPlot.figure()
  h1 = PyPlot.plt[:hist](training[i,:],20, normed=true)
end


# if using estimating only two unknown parameters
PyPlot.figure()
PyPlot.scatter3D(training[1,:],training[2,:],training_targets)
PyPlot.xlabel("log d")
PyPlot.ylabel("log c")
PyPlot.zlabel("loglik")

PyPlot.figure()
PyPlot.plot(test')
PyPlot.figure()
PyPlot.plot(test_targets)

n_col = dim + sum(dim:-1:1)
prec_outlier = 0.05
tail_rm = "left"
(training_targets,training) = removeoutlier(training_targets,training,Int64(length(training_targets)*prec_outlier),tail_rm)
(test_targets,test) = removeoutlier(test_targets,test,Int64(length(test_targets)*prec_outlier), tail_rm)

X = meanfunction(training',false,1:n_col)

corrmatrix_predictors = cor(X[:,1:end])

corrplot(X)


################################################################################
##                         fit GP model                                      ###
################################################################################

kernel = "SE"
gp_test = GPModel("est_method",zeros(10), zeros(4),
eye(length_training_data-20), zeros(length_training_data-20),
zeros(3,length_training_data-20),collect(1:10))
lasso = true
ml_est(gp_test,trainig_data,kernel,lasso,prec_outlier,tail_rm)

################################################################################
##            Predictions and residual analysis                              ###
################################################################################

(mean_pred_ml, var_pred_ml, prediction_sample_ml) = predict(test,gp_test,true)

residuals = test_targets-prediction_sample_ml

PyPlot.figure()
PyPlot.plot(residuals, color = "blue")
for i = 1:size(test,1)
  PyPlot.figure()
  PyPlot.plot(test[i,:],residuals, color = "blue", "*")
end
PyPlot.figure()
PyPlot.plot(test_targets,residuals, color = "blue", "*")


PyPlot.figure()
#PyPlot.plot(meanfunction(test', true, indeices_keep)*theta_hat[1:end-5], color = "blue", "*")
PyPlot.plot(prediction_sample_ml, color = "red", "*")
PyPlot.hold(true)
PyPlot.plot(test_targets, color = "blue", "*")



# plot variances

PyPlot.figure()
h1 = PyPlot.plt[:hist](var_pred_ml,25)
percentile(var_pred_ml,95)

PyPlot.figure()
h1 = PyPlot.plt[:hist](prediction_sample_ml,25)


findlefttail(x) = x >= percentile(var_pred_ml,99)
indx_low = find(findlefttail, var_pred_ml)
indx_high = setdiff(1:length(prediction_sample_ml), indx_low)
idx = indx_low

bins = 50
PyPlot.figure()
h1 = PyPlot.plt[:hist](test_targets[idx],bins)
PyPlot.figure()
h2 = PyPlot.plt[:hist](prediction_sample_ml[idx],bins)

PyPlot.figure()
PyPlot.plot(h1[2][1:end-1],h1[1], "b")
PyPlot.hold(true)
PyPlot.plot(h2[2][1:end-1],h2[1], "r")

bins = 50
PyPlot.figure()
h1 = PyPlot.plt[:hist](var_pred_ml[idx],bins)
PyPlot.figure()
h2 = PyPlot.plt[:hist](var_pred_ml[idx],bins)

PyPlot.figure()
PyPlot.plot(h1[2][1:end-1],h1[1], "b")
PyPlot.hold(true)
PyPlot.plot(h2[2][1:end-1],h2[1], "b--")




# if using estimating only two unknown parameters
text_size = 20
PyPlot.figure()
ax = axes()
PyPlot.scatter3D(test[1,:],test[2,:],test_targets, color = "blue")
PyPlot.xlabel("log d",fontsize=text_size)
PyPlot.ylabel("log c",fontsize=text_size)
PyPlot.zlabel("loglik",fontsize=text_size)
PyPlot.legend(("PF","GP"),fontsize=text_size)
ax[:tick_params]("both",labelsize=20)


text_size = 20
PyPlot.figure()
ax = axes()
PyPlot.scatter3D(test[1,:],test[2,:],test_targets, color = "blue")
PyPlot.hold(true)
PyPlot.scatter3D(test[1,:],test[2,:],prediction_sample_ml, color = "red")
PyPlot.xlabel("log d",fontsize=text_size)
PyPlot.ylabel("log c",fontsize=text_size)
PyPlot.zlabel("loglik",fontsize=text_size)
PyPlot.legend(("PF","GP"),fontsize=text_size)
ax[:tick_params]("both",labelsize=20)


RMSE_ml_mean = RMSE(test_targets, mean_pred_ml)
RMSE_ml_sample = RMSE(test_targets, prediction_sample_ml)
