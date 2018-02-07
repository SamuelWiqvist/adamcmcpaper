# Script for running the GP regression model on the
# training/test data


################################################################################
##            generate training and test data                                ###
################################################################################

include("rickermodel.jl")

# set up problem
problem = set_up_problem(ploton = true)

# set length of data sets
length_training_data = 2000
length_test_data = 2000
burn_in = 2000
length_data = length_test_data + length_training_data + burn_in

# set algorithm parameters
problem.alg_param.alg = "PMCMC"
problem.alg_param.burn_in = burn_in
problem.alg_param.R = length_data
problem.alg_param.N = 1000
problem.data.y = Array(readtable("y.csv"))[:,1]
problem.data.y = Array(readtable("y_data_set_2.csv"))[:,1]

# set AM algorithm

# use AM alg for adaptive updating
problem.adaptive_update = AMUpdate(eye(3), 2.4/sqrt(3), 1., 0.7, 50)

problem.adaptive_update = noAdaptation(2.4/sqrt(3)*eye(3))

# or, use AM gen alg for adaptive updating
problem.adaptive_update = AMUpdate_gen(eye(3), 2.4/sqrt(3), 0.3, 1., 0.8, 25)

# generate training and test data
(res, data, targets) = MCMC(problem, true)

# plot results
analyse_results(problem, res[1])


################################################################################
###                         set traninig and test data                       ###
################################################################################


# set trainin and test data
training = data[:,1:length_training_data]
training_targets = targets[1:length_training_data]
test = data[:,length_training_data+1:end]
test_targets = targets[length_training_data+1:end]

trainig_data = zeros(4, size(training,2))
trainig_data[1:3,:] = training
trainig_data[end,:] = training_targets

prec_remove = 0.05
tail_rm = "left"

(training_targets,training) = removeoutlier(training_targets,training,Int64(length(training_targets)*prec_remove), tail_rm)


(test_targets,test) = removeoutlier(test_targets,test,Int64(length(test_targets)*prec_remove), tail_rm)


# set data
y = training_targets
x = training
X = meanfunction(x',false,1:9)

# plot training data
PyPlot.figure()
PyPlot.plot(training')
PyPlot.figure()
PyPlot.plot(training_targets)

PyPlot.figure()
h1 = PyPlot.plt[:hist](training_targets,20, normed=true)

std(training,2)
scale = 1.5
limit = mean(training_targets)  - scale*std(training_targets)


findlefttail(x) = x <= limit
indx_low = find(findlefttail, training_targets)
indx_high = setdiff(1:length(training_targets), indx_low)
training_targets = training_targets[indx_high]
training = training[:,indx_high]


PyPlot.figure()
PyPlot.plot(training')
PyPlot.figure()
PyPlot.plot(training_targets)

PyPlot.figure()
h1 = PyPlot.plt[:hist](training_targets,20, normed=true)




#PyPlot.hold(true)
#PyPlot.plot(training_targets)

PyPlot.figure()
PyPlot.plot(test')
PyPlot.figure()
PyPlot.plot(test_targets)


X_corr = zeros(size(X,1), size(X,2)+1)
X_corr[:,1] = y
X_corr[:,2:end] = X


corrplot(X_corr, label = [L"$\ell(\theta)$", L"$\beta_1$",L"$\beta_2$",L"$\beta_3$",L"$\beta_4$",L"$\beta_5$",L"$\beta_6$",L"$\beta_7$",L"$\beta_8$",L"$\beta_9$"])


#=
corrmatrix_predictors = cor(X[:,1:end])

X_data = zeros(size(X,1), size(X,2)+1)
X_data[:,1] = y

X_data[:,2:end] = X[:,1:end]

corrplot(X_corr, label = [L"$y$", L"$x_1$","x2","x3","x4","x5","x6","x7","x8","x9"])
=#

y = trainig_data[end,:]
x = trainig_data[1:end-1,:]
(y,x) = removeoutlier(y,x,Int64(length(y)*prec_remove),tail_rm)



@printf "Length of training data: %d" size(x,2)
dim = size(x,1)
n_col = dim + sum(dim:-1:1)

X = meanfunction(x',true,1:n_col+1)
buffer = zeros(3*size(X,1), size(X,1))
indeices_keep = collect(1:n_col+1)
beta_0 = zeros(n_col+1)
logΦ_0 = log([std(y); std(y)/sqrt(2); std(x,2)])
theta_0 = zeros(n_col+1+length(logΦ_0))

if false
  beta_0 = inv(X'*X)*X'*y # start value for the the beta:s should be the regression est of the beta:s
else
  X = meanfunction(x',false,collect(1:n_col))
  lassopath = Lasso.fit(LassoPath,X,y, λminratio = 0.02,nλ = 500,intercept=true)
  ind_best = indmax(lassopath.pct_dev)
  beta_0[1] = lassopath.b0[ind_best]

  for i = 2:length(beta_0)
    beta_0[i] = lassopath.coefs[i-1,ind_best]
  end

  indeices_keep = find(x -> x != 0.00,beta_0)
  beta_0 = beta_0[indeices_keep]
  X = [ones(size(X,1)) X] # add intercept
  X = X[:,indeices_keep]
end


################################################################################
###                         fit GP model                                     ###
################################################################################

# test est_ml and predict
gp_test = GPModel("est_method",zeros(10), zeros(4),
eye(length_training_data-20), zeros(length_training_data-20),
zeros(3,length_training_data-20),collect(1:10))
lasso = true
kernel = "SE"
ml_est(gp_test,trainig_data,kernel,lasso,prec_remove,tail_rm)


################################################################################
##            Predictions and residual analysis                              ###
################################################################################

(mean_pred_ml , var_pred_ml , prediction_sample_ml) = predict(test,gp_test,false)


# plot variances

PyPlot.figure()
h1 = PyPlot.plt[:hist](var_pred_ml,25)
percentile(var_pred_ml,95)

PyPlot.figure()
h1 = PyPlot.plt[:hist](prediction_sample_ml,25)


findlefttail(x) = x >= percentile(var_pred_ml,90)
indx_low = find(findlefttail, var_pred_ml)
indx_high = setdiff(1:length(prediction_sample_ml), indx_low)
idx = indx_high

bins = 50
PyPlot.figure()
h1 = PyPlot.plt[:hist](test_targets,bins)
PyPlot.figure()
h2 = PyPlot.plt[:hist](prediction_sample_ml,bins)

PyPlot.figure()
PyPlot.plot(h1[2][1:end-1],h1[1], "b")
PyPlot.hold(true)
PyPlot.plot(h2[2][1:end-1],h2[1], "r")

bins = 50
PyPlot.figure()
h1 = PyPlot.plt[:hist](var_pred_ml[indx_low],bins)
PyPlot.figure()
h2 = PyPlot.plt[:hist](var_pred_ml[indx_high],bins)

PyPlot.figure()
PyPlot.plot(h1[2][1:end-1],h1[1], "b")
PyPlot.hold(true)
PyPlot.plot(h2[2][1:end-1],h2[1], "b--")


# calc residuals
residuals =test_targets-meanfunction(test',true)*beta_0# test_targets - prediction_sample_ml
residuals =test_targets-prediction_sample_ml
residuals =test_targets-meanfunction(test', true, indeices_keep)*theta_hat[1:end-5]

text_size = 20
label_size = 15

PyPlot.figure()
ax = axes()
PyPlot.plot(residuals, color = "blue")
PyPlot.xlabel(L"Iteration",fontsize=text_size)
PyPlot.ylabel(L"Residual",fontsize=text_size)
ax[:tick_params]("both",labelsize=label_size)

PyPlot.figure()
ax = axes()
PyPlot.plot(test[1,:],residuals, color = "blue", "*")
PyPlot.xlabel(L"log $r$",fontsize=text_size)
PyPlot.ylabel(L"Residual",fontsize=text_size)
ax[:tick_params]("both",labelsize=label_size)

PyPlot.figure()
ax = axes()
PyPlot.plot(test[2,:],residuals, color = "blue", "*")
PyPlot.xlabel(L"log $\phi$",fontsize=text_size)
PyPlot.ylabel(L"Residual",fontsize=text_size)
ax[:tick_params]("both",labelsize=label_size)

PyPlot.figure()
ax = axes()
PyPlot.plot(test[3,:],residuals, color = "blue", "*")
PyPlot.xlabel(L"log $\sigma$",fontsize=text_size)
PyPlot.ylabel(L"Residual",fontsize=text_size)
ax[:tick_params]("both",labelsize=label_size)

PyPlot.figure()
ax = axes()
PyPlot.plot(test_targets,residuals, color = "blue", "*")
PyPlot.ylabel(L"\ell",fontsize=text_size)
PyPlot.xlabel(L"Residual",fontsize=text_size)
ax[:tick_params]("both",labelsize=label_size)

PyPlot.figure()
#PyPlot.plot(meanfunction(test', true, indeices_keep)*theta_hat[1:end-5], color = "blue", "*")
ax = axes()
PyPlot.plot(prediction_sample_ml, color = "red", "*")
PyPlot.hold(true)
PyPlot.plot(test_targets, color = "blue", "*")
PyPlot.ylabel(L"\ell",fontsize=text_size)
PyPlot.xlabel(L"Iteration",fontsize=text_size)
ax[:tick_params]("both",labelsize=label_size)


bins = 50
PyPlot.figure()
h1 = PyPlot.plt[:hist](prediction_sample_ml,bins)
PyPlot.figure()
h2 = PyPlot.plt[:hist](test_targets,bins)

PyPlot.figure()
PyPlot.plot(h1[2][1:end-1],h1[1], "r")
PyPlot.hold(true)
PyPlot.plot(h2[2][1:end-1],h2[1], "b")



res_exp = zeros(length(residuals),2)
res_exp[:,1] = residuals
writetable("output_residuals.csv",DataFrame(res_exp))


# calc RMSE
RMSE_ml_mean = RMSE(test_targets, mean_pred_ml)
RMSE_ml_sample = RMSE(test_targets, prediction_sample_ml)


PyPlot.figure()
PyPlot.scatter3D(test[1,:],test[2,:],test_targets, color = "red")
PyPlot.hold(true)
PyPlot.scatter3D(test[1,:],test[2,:],mean_pred_ml, color = "blue")
PyPlot.title("Targets (red), predictions (blue)")

PyPlot.figure()
PyPlot.scatter3D(test[1,:],test[2,:],test_targets, color = "red")
PyPlot.hold(true)
PyPlot.scatter3D(test[1,:],test[2,:],prediction_sample_ml, color = "blue")
PyPlot.title("Targets (red), predictions (blue)")



# using the GP approch

meanfunc = MeanZero()  # Zero mean function
covfunc = SEArd([0.0,0.0],0.0)
logObsNoise = log(var(y)/sqrt(2)) # log standard deviation of observation noise (this is optional)

# Fit GP model
gp = GaussianProcesses.GP(x_col_major,y_meanremoved,meanfunc,covfunc, logObsNoise)      # Fit the GP

# store calculations...

beta_0 = inv(X'*X)*X'*y # start value for the the beta:s should be the regression est of the beta:s
#beta_0 = [1,1,1,1,1,1]
logΦ_0 = log([std(y)/sqrt(2), 1, 1, 1])

theta_0 = [beta_0; logΦ_0]
buffer = zeros(3*size(X,1), size(X,1))
theta_last = similar(theta_0)

function objectivfunctionclosure(theta,buffer,theta_last)
  return objectivfunction(x,X,y,kernel, theta, buffer, theta_last)
end

function gclosure!(theta,stor,buffer,theta_last)
  return g!(y,x,X, stor,buffer, theta, theta_last)
end


@time objectivfunctionclosure(theta_0,buffer,zeros(10))

stor_test = zeros(10)
@time gclosure!(theta_0,stor_test,buffer,zeros(10))
df = TwiceDifferentiableFunction(theta -> objectivfunctionclosure(theta,buffer,theta_last),
  (theta,stor) -> gclosure!(theta,stor,buffer,theta_last))


options = Optim.Options(store_trace=true, iterations=1000,time_limit=60*10)

opt_res = Optim.optimize(df,
                        theta_0,
                        ConjugateGradient(),
                        options)

theta_hat = Optim.minimizer(opt_res)

# grid search-ish
log_sigma = -10:2:10
log_sigma_kernel = -10:2:10
logl1 =  -10:2:10
logl2 =  -10:2:10

profileML_vec = zeros(length(log_sigma)^4)
ml_vec = zeros(length(log_sigma)^4)
param_comb = zeros(4,length(log_sigma)^4)

i = 1
for log_sigma_temp in log_sigma
  @printf "Starting iteration %d" i
  for log_sigma_kernel_temp in log_sigma_kernel
    for logl1_temp in logl1
      for logl2_temp in logl2
        profileML_vec[i] = profileML(x,X,y,kernel,[log_sigma_temp, log_sigma_kernel_temp, logl1_temp, logl2_temp])
        ml_vec[i] = negML(x,X,y,kernel,[beta_0, log_sigma_temp, log_sigma_kernel_temp, logl1_temp, logl2_temp],)
        param_comb[:,i] = [log_sigma_temp, log_sigma_kernel_temp, logl1_temp, logl2_temp]
        i = i+1
      end
    end
  end
end

min_val = sort(loglik_vec)[1:5]
min_idx = sortperm(loglik_vec)[1:5]

param_comb[:,min_idx]'

(idx_nonsym_cov) = find(loglik_vec .== Inf)

# find 5 best values
PyPlot.figure()
PyPlot.plot(1:length(loglik_vec), loglik_vec, "r.")


# blackbox optimization
bboptimize(profileMLclosure; SearchRange = (-Inf, 5.0), NumDimensions = 4, Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 10.0)


# calc Β_hat
Φ_hat = exp(logΦ_0)
cov_m = covariancefunction(x, Φ_hat, kernel)
Β_hat = (X'*(cov_m\X))\(X'*(cov_m\y))

# do implace operations on a vector
function f!(x)
  x[:] = x.^2
end

# old stuff


# two step estimation

#=
(Beta_hat, gpmodel) = @time twostageestimation(y,x)

(mean_pred, var_pred, pred_samples) = @time  twostageprediction(test, Beta_hat, gpmodel)

RMSE_twostageestimation = RMSE(test_targets, mean_pred)


PyPlot.figure()
PyPlot.scatter3D(test[1,:],test[2,:],test_targets, color = "red")
PyPlot.hold(true)
PyPlot.scatter3D(test[1,:],test[2,:],mean_pred, color = "blue")

logparams_gp = GaussianProcesses.get_params(gpmodel, mean=false)
logΦ_hat = [logparams_gp[1]; logparams_gp[end]; logparams_gp[2:3] ]
Φ_hat = exp(logparams_gp)

#RMSE(test_targets, mean_pred)

gp_twostage = GPModel("twostage",Beta_hat,logparams_gp,inv(covariancefunction(x,exp(logparams_gp),kernel)),
                    y-X*Beta_hat,x)

=#
# fit the GP using the profile likelihood function

# set kernel for covariance function

# set start values for Φ
#cov_m_x = cov(x)
logΦ_0 = log([std(y)/sqrt(2), 1, 1, 1])

# calc loglik for start value

loglik_0 = @time profileML(x,X,y,kernel,logΦ_0)

# create closure

function profileMLclosure(logΦ)
  return profileML(x,X,y,kernel,logΦ)
end

options = Optim.Options(iterations=100,time_limit=60*15)

opt_res = Optim.optimize(profileMLclosure,
                        logΦ_0,
                        Optim.ConjugateGradient(),
                        options)

logPhi_hat = Optim.minimizer(opt_res)
cov_m_opt = covariancefunction(x,exp(logPhi_hat),kernel)
cov_m_opt_inv = inv(cov_m_opt)
Beta_hat = inv(X'*cov_m_opt_inv*X)*X'*cov_m_opt_inv*y

gp_profile = GPModel(Beta_hat,logPhi_hat,cov_m_opt,y-X*Beta_hat,x)


# fit Gp using the standard likelihood function
beta_0 = inv(X'*X)*X'*y # start value for the the beta:s should be the regression est of the beta:s
#beta_0 = [1,1,1,1,1,1]
logΦ_0 = log([std(y)/sqrt(2), 1, 1, 1, 1])

theta_0 = [beta_0; logΦ_0]

loglik_0 = @time negML(x,X,y,kernel,theta_0)


function mlclosure(theta)
  return negML(x,X,y,kernel, theta)
end

options = Optim.Options(store_trace=true, iterations=1000,time_limit=60*10)

opt_res = Optim.optimize(mlclosure,
                        theta_0,
                        ConjugateGradient(),
                        options)

theta_hat = Optim.minimizer(opt_res)

gp_ml = GPModel("ml",theta_hat[1:6], theta_hat[7:end],
              inv(covariancefunction(x,exp(theta_hat[7:end]),kernel)), y-X*theta_hat[1:6],x)



# black box optimization (for comparison)

# blackbox optimization
bbopt_res = bboptimize(profileMLclosure; SearchRange = (-Inf, 5.0), NumDimensions = 4, Method = :adaptive_de_rand_1_bin_radiuslimited, MaxTime = 10.0)
