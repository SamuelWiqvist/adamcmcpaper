# This file contains the functions for the GP model

# load packages
using Yeppp # for fast exp function

# type(s):

"""
GPModel is the type for the Gaussian processes model with qudaric mean function
and some covariance function.
"""
type GPModel
  est_method::String
  Β::Array
  logΦ::Array # log([σ, σ_kernel lengthscales])
  cov_inv::Array
  diff::Array # y-XΒ
  x::Array # training data
  indecies_to_include::Vector
end

# functions:

doc"""
    meanfunction(x::Array, intercept::Bool, indecies_to_include)

meanfunction computes the design matrix for the quadratic mean function.
"""
function meanfunction(x::Array, intercept::Bool, indecies_to_include)

  nbr_obs = size(x,1)
  dim = size(x,2)

  n_col = dim + sum(dim:-1:1)
  X = zeros(nbr_obs,n_col)

  X[:,1:dim] = x
  X[:,dim+1:2*dim] = x.^2

  idx = 2*dim
  for i = 1:dim-1
    for j = i+1:dim
      idx = idx + 1
      X[:,idx] = x[:,i].*x[:,j]
    end
  end

  if intercept
    X = [ones(size(x,1),1) X]
    return X[:,indecies_to_include]
  else
    #indecies_to_include  = indecies_to_include[2:end]
    return X[:,indecies_to_include]
  end

end

doc"""
    predict(x_pred::Array{Float64}, gp::GPModel, noisy_pred::Bool=true)

predict computes the predictions at the values in x_pred.
"""
function predict(x_pred::Array{Float64}, gp::GPModel, noisy_pred::Bool=true)
  mu_pred = zeros(size(x_pred,2))
  sigma_pred = zeros(size(x_pred,2))
  prediction_sample = zeros(size(x_pred,2))

  for i = 1:size(x_pred,2)
    (mu_pred[i], sigma_pred[i],prediction_sample[i]) = prediction(x_pred[:,i], gp,noisy_pred)
  end
  return mu_pred, sigma_pred, prediction_sample
end

doc"""
    predict(x_pred::Array{Float64}, gp::GPModel, pred_method::String, est_method::String,noisy_pred::Bool=true)


predict computes the predictions at the values in x_pred.
"""
function predict(x_pred::Array{Float64}, gp::GPModel, pred_method::String, est_method::String,noisy_pred::Bool=true, return_var::Bool=false)

  mu_pred = zeros(size(x_pred,2))
  sigma_pred = zeros(size(x_pred,2))
  prediction_sample = zeros(size(x_pred,2))

  if est_method == "ml"

    for i = 1:size(x_pred,2)
      (mu_pred[i], sigma_pred[i],prediction_sample[i]) = prediction(x_pred[:,i], gp,noisy_pred)
    end

  else

    error("The two-stage method is not implemented.")

  end

  if !return_var
    if pred_method == "mean"
      return mu_pred
    else
      return prediction_sample
    end
  else
    if pred_method == "mean"
      return mu_pred, sigma_pred
    else
      return prediction_sample, sigma_pred
    end
  end

end


doc"""
    prediction(x_pred::Array{Float64}, gp::GPModel, noisy_pred::Bool = true)


prediction computes the prediction at x_pred.
"""
function prediction(x_pred::Array{Float64}, gp::GPModel, noisy_pred::Bool = true)


    # pre-allocate arrays
    c = zeros(size(gp.x,2))
    covtemp = zeros(size(gp.x,2))
    mu_pred = zeros(1,1)
    sigma_pred = zeros(1,1)
    temp1 = zeros(1,1)
    temp2 = copy(gp.diff)

    # set parameters
    l_vec = -1/2*1./(exp(gp.logΦ[3:end]).^2)


    # calc variance vectro, i.e. calc K(x_pred, X)

    # compute for each dimension
    for i = 1:length(l_vec)
      covtemp = l_vec[i]*(x_pred[i] - gp.x[i,:]).^2
      c = c + vec(covtemp)
    end

    # take exponent
    Yeppp.exp!(c) # still slow.... migth be possible to fix this by calling the c function for exp

    # scale matrix with σ_kernel
    scale!(c, exp(gp.logΦ[2])^2)
    #c = exp(gp.logΦ[2])^2*c

    if noisy_pred
      # this is the slow part...
      LinAlg.A_mul_B!(mu_pred,meanfunction(x_pred',true,gp.indecies_to_include),gp.Β)
      LinAlg.A_mul_B!(temp2,gp.cov_inv,gp.diff)
      LinAlg.A_mul_B!(temp1,c',temp2)
      LinAlg.axpy!(1.,temp1,mu_pred)
      #mu_pred = meanfunction(x_pred',true,gp.indecies_to_include)*gp.Β + c'*gp.cov_inv*gp.diff
      LinAlg.A_mul_B!(temp2,gp.cov_inv,c)
      LinAlg.A_mul_B!(sigma_pred,c',temp2)
      sigma_pred = exp(gp.logΦ[1])^2 + exp(gp.logΦ[2])^2 + sigma_pred
      #LinAlg.axpy!(1.,exp(gp.logΦ[1])^2,sigma_pred)
      #LinAlg.axpy!(1.,exp(gp.logΦ[2])^2,sigma_pred)
      #sigma_pred = exp(gp.logΦ[2])^2 + exp(gp.logΦ[1])^2 - c'*gp.cov_inv*c #+ exp(gp.logΦ[1])^2
    else
      # this is the slow part...
      LinAlg.A_mul_B!(mu_pred,meanfunction(x_pred',true,gp.indecies_to_include),gp.Β)
      LinAlg.A_mul_B!(temp2,gp.cov_inv,gp.diff)
      LinAlg.A_mul_B!(temp1,c',temp2)
      LinAlg.axpy!(1.,temp1,mu_pred)
      #mu_pred = meanfunction(x_pred',true,gp.indecies_to_include)*gp.Β + c'*gp.cov_inv*gp.diff
      LinAlg.A_mul_B!(temp2,gp.cov_inv,c)
      LinAlg.A_mul_B!(temp1,c',temp2)
      LinAlg.axpy!(1.,temp1,sigma_pred)
      sigma_pred =  exp(gp.logΦ[2])^2 + sigma_pred

      #LinAlg.axpy!(1.,exp(gp.logΦ[2])^2,sigma_pred)

      #mu_pred = meanfunction(x_pred',true,gp.indecies_to_include)*gp.Β + c'*gp.cov_inv*gp.diff
      #sigma_pred = exp(gp.logΦ[2])^2 - c'*gp.cov_inv*c #+ exp(gp.logΦ[1])^2
    end
    if sigma_pred[1] < 0
      return mu_pred[1], sigma_pred[1], NaN #rand(Normal(mu_pred[1],sqrt(sigma_pred[1])))
    else
      return mu_pred[1], sigma_pred[1], rand(Normal(mu_pred[1],sqrt(sigma_pred[1])))
    end


  #=
  # pre-allocate arrays
  c = zeros(size(gp.x,2))
  covtemp = zeros(size(gp.x,2))

  # set parameters
  l_vec = -1/2*1./(exp(gp.logΦ[3:end]).^2)


  # calc variance vectro, i.e. calc K(x_pred, X)

  # compute for each dimension
  for i = 1:length(l_vec)
    covtemp = l_vec[i]*(x_pred[i] - gp.x[i,:]).^2
    # problem v0.4 !!!!!!!!!
    # was: c = c + covtempc
    c = c + vec(covtemp)
  end

  # take exponent
  Yeppp.exp!(c) # still slow.... migth be possible to fix this by calling the c function for exp

  # scale matrix with σ_kernel
  c = exp(gp.logΦ[2])^2*c

  # compute prediction
  res = zeros(1)
  res2 = zeros(1)
  d = At_mul_B(c,gp.cov_inv)
  A_mul_B!(res,d,gp.diff)
  A_mul_B!(res2,d,c)

  if noisy_pred
    mu_pred = meanfunction(x_pred',true,gp.indecies_to_include)*gp.Β + res #c'*gp.cov_inv*gp.diff
    sigma_pred = exp(gp.logΦ[2])^2 + exp(gp.logΦ[1])^2 - res2 # c'*gp.cov_inv*c #+ exp(gp.logΦ[1])^2
  else
    mu_pred = meanfunction(x_pred',true,gp.indecies_to_include)*gp.Β + res #c'*gp.cov_inv*gp.diff
    sigma_pred = exp(gp.logΦ[2])^2 - res2 #c'*gp.cov_inv*c #+ exp(gp.logΦ[1])^2
  end
  if sigma_pred[1] < 0
    return mu_pred[1], sigma_pred[1], NaN #rand(Normal(mu_pred[1],sqrt(sigma_pred[1])))
  else
    return mu_pred[1], sigma_pred[1], rand(Normal(mu_pred[1],sqrt(sigma_pred[1])))
  end

  =#


end


doc"""
    covariancefunction(x::Array, Φ::Array, kernel::String)

covariancefunction computes the covariance function using the SE kernel.
"""
function covariancefunction(x::Array, Φ::Array, kernel::String)

  # set parameters
  σ2 = Φ[1]^2
  σ_kernel2  =  Φ[2]^2
  l_vec = -1/2*(1./(Φ[3:end].^2))

  # pre-allocate matricies
  covtemp = zeros(length(x[1,:]),length(x[1,:]))
  cov_m = zeros(length(x[1,:]),length(x[1,:]))
  sigma_m = eye(size(cov_m,1))

  # calc sigma_m
  scale!(sigma_m,σ2)

  # set power2(x) = x^2 function
  power2(x) = x^2

  # compute for each dimension
  for i = 1:length(l_vec)
    broadcast!(-, covtemp, x[i,:], x[i,:]')
    broadcast!(power2, covtemp, covtemp)
    scale!(covtemp,l_vec[i])
    broadcast!(+, cov_m,cov_m,covtemp)
  end

  # take exponent
  Yeppp.exp!(cov_m) # still slow.... migth be possible to fix this by calling the c function for exp

  # scale matrix with σ_kernel
  scale!(cov_m,σ_kernel2)

  # add sigma_m on the main diagonal
  broadcast!(+, cov_m,cov_m,sigma_m)

  return cov_m

end


doc"""
    leastsquaresestimation(X::Matrix{Float64}, y::Vector{Float64})

Comutes regression estiamtins of beta for X*beta = y, also computes the confidence
bands for the beta's.
"""
function leastsquaresestimation(X::Matrix{Float64}, y::Vector{Float64})

  # compute betas
  beta = \(X'*X,X')*y

  # compute hat matrix
  H = X*\(X'*X,X')

  # compute mse
  sse = y'*(eye(size(H,1),size(H,2))-H)*y
  mse = sse/(size(X,1)-size(X,2))

  # compute variance matrix
  s = mse.*\(X'*X,eye(size(X'*X,1),size(X'*X,1)))

  conf_limits = 1.96*diag(s) # approximate t(1-0.05/2,n-p) with 1.96

  return beta, conf_limits

end


# not storing calc

doc"""
    calculate_common!(theta::Array, buffer::Array{Float64}, x::Array{Float64})

Computes the common calculations for the objectiv function, i.e. the negative ml function
and the gradient of the objective function.
"""
function calculate_common!(theta::Array, buffer::Array{Float64}, x::Array{Float64})

  # set parameters
  σ2 = exp(theta[end-size(x,1)-1])^2
  σ_kernel2  =  exp(theta[end-size(x,1)])^2
  l_vec = -1/2./(exp(theta[end-size(x,1)+1:end]).^2)


  # pre-allocate matricies
  sigma_m = eye(length(x[1,:]),length(x[1,:]))
  cov_m = zeros(length(x[1,:]),length(x[1,:]))
  difftemp = zeros(size(cov_m))
  diff_m = zeros(size(cov_m))
  cov_m = zeros(size(cov_m))

  # calc sigma_m
  scale!(sigma_m,σ2)

  # set power2(x) = x^2 function
  power2(x) = x^2

  # compute for each dimension
  for i = 1:length(l_vec)
    broadcast!(-, difftemp, x[i,:], x[i,:]')
    broadcast!(power2, difftemp, difftemp)
    scale!(difftemp,l_vec[i])
    broadcast!(+, diff_m,diff_m,difftemp)
  end

  # take exponent
  Yeppp.exp!(diff_m) # still slow.... migth be possible to fix this by calling the c function for exp

  # set cov matrix
  cov_m = diff_m

  # scale matrix with σ_kernel
  scale!(cov_m,σ_kernel2)

  broadcast!(+, cov_m,cov_m,sigma_m)
  # v0.4 problem
  # issymmetric(cov_m) is not implemented in v4.0
  if cov_m == cov_m'  && isposdef(cov_m) # check that cov matrix is symmetric and positive semi def.
    cov_m_inv = inv(cov_m)
    buffer[:,:] = [diff_m; cov_m; cov_m_inv]
  else
    # set buffer to something here
    cov_m_inv = zeros(size(cov_m))
    buffer[:,:] = [diff_m; cov_m; cov_m_inv]
  end


end


doc"""
    objectivfunction(x::Array{Float64},X::Array{Float64},y::Vector{Float64},
  kernel::String, theta::Array, buffer::Array{Float64})

The negative maximum likelihood function for the GP model.
"""
function  objectivfunction(x::Array{Float64},X::Array{Float64},y::Vector{Float64},
  kernel::String, theta::Array, buffer::Array{Float64})

  # calc common parts of the objective function and gradient
  calculate_common!(theta,buffer,x) # buffer = [diff_m; cov_m; cov_m_inv]

  # set parameters
  beta = theta[1:size(X,2)] # theta = [beta Phi]

  # set matrices that are stored on buffre
  diff_m = buffer[1:length(x[1,:]),:]
  cov_m = buffer[length(x[1,:])+1:2*length(x[1,:]),:]
  cov_m_inv = buffer[2*length(x[1,:])+1:3*length(x[1,:]),:]

  # calc chol factorizations
  if sum(cov_m_inv) != 0 # if the cov matrix is symmetrix and pos semi def compute negloglik

    R = chol(cov_m)
    # calc negloglik
    negloglik =  2*sum(log(diag(R))) + (y-X*beta)'*cov_m_inv*(y-X*beta)
    negloglik = negloglik[1]

  else  # set negloglik to some value

    println("non-symmetric covariance matrix at logΦ:")
    println(theta[end-4:end])
    negloglik = Inf

  end

  return  negloglik

end



doc"""
    g!(y::Array{Float64},x::Array{Float64},X::Array{Float64}, stor::Array,
  buffer::Array{Float64}, theta::Array)

Computes the gradient of the negative log-likelihood function.
"""
function g!(y::Array{Float64},x::Array{Float64},X::Array{Float64}, stor::Array,
  buffer::Array{Float64}, theta::Array)


  # calc common parts of the objective function and gradient
  calculate_common!(theta,buffer,x)

  # set parameters
  σ2 = exp(theta[end-size(x,1)-1])^2
  σ_kernel2  =  exp(theta[end-size(x,1)])^2
  l_vec = 1./(exp(theta[end-size(x,1)+1:end]).^3)
  beta = theta[1:size(X,2)]

  # pre-allocate matricies
  dCov_b_d_Phi_i_val = zeros(1)
  dCov_b_d_Phi_i = zeros(length(x[1,:]),length(x[1,:]))
  dCov_b_d_Phi_i_inner_deriv = zeros(length(x[1,:]),length(x[1,:]))

  # set matrices that are stored on buffre
  diff_m = buffer[1:length(x[1,:]),:]
  cov_m = buffer[length(x[1,:])+1:2*length(x[1,:]),:]
  cov_m_inv = buffer[2*length(x[1,:])+1:3*length(x[1,:]),:]

  power2(x) = x^2

  # calc chol factorizations
  if sum(cov_m_inv) != 0

    # calc gradient for beta
    stor[1:length(beta)] = -2*X'*cov_m_inv*(y-X*beta)

    # gradient for Phi
    for i = length(beta)+1:length(stor)

      if i == length(beta)+1
        # dCov_b_d_\sigma
        dCov_b_d_Phi_i = 2*sqrt(σ2)*eye(length(x[1,:]),length(x[1,:]))
      elseif i == length(beta)+2
        # dCov_b_d_\sigma_kernel
        dCov_b_d_Phi_i = 2*sqrt(σ_kernel2)*diff_m

      else
        # dCov_b_d_l_i
        idx = i - (length(beta)+2)
        dCov_b_d_Phi_i = scale(σ_kernel2, diff_m)
        broadcast!(-, dCov_b_d_Phi_i_inner_deriv, x[idx,:], x[idx,:]')
        broadcast!(power2, dCov_b_d_Phi_i_inner_deriv, dCov_b_d_Phi_i_inner_deriv)
        scale!(dCov_b_d_Phi_i_inner_deriv,l_vec[idx])
        dCov_b_d_Phi_i = broadcast!(*,dCov_b_d_Phi_i,dCov_b_d_Phi_i,dCov_b_d_Phi_i_inner_deriv)

      end

      trace_Phi_i = trace(cov_m_inv*dCov_b_d_Phi_i)
      dCov_b_d_Phi_i_val  = -(y-X*beta)'*cov_m_inv*dCov_b_d_Phi_i*cov_m_inv*(y-X*beta) + trace_Phi_i
      stor[i] = dCov_b_d_Phi_i_val[1]
    end

  else

    # set gradient to something
    stor[:] = 1000
    println("Non-symmetric covariance matrix at logΦ:")
    show(theta[end-size(x,1)-1:end])
    negloglik = Inf

  end



end

doc"""
    ml_est(gp::GPModel, data_training::Array{Float64}, kernel::String="SE",lasso::Bool=true)

Computes the maximum likelihood estimation of the parameters of the GP model,
by minimizing the negative log-likelihood function.
"""
function ml_est(gp::GPModel, data_training::Array{Float64}, kernel::String="SE",lasso::Bool=true, percentage::Float64=0.05, sided::String="both")

  @printf "Starting ml estimation"

  # v0.4 problem
  y = vec(data_training[end,:]) # need this to fix issue with types..
  x = data_training[1:end-1,:]
  (y,x) = removeoutlier(y,x,Int64(length(y)*percentage),sided)

  @printf "Length of training data: %d" size(x,2)
  dim = size(x,1)
  n_col = dim + sum(dim:-1:1)

  X = meanfunction(x',true,1:n_col+1)
  buffer = zeros(3*size(X,1), size(X,1))
  indeices_keep = collect(1:n_col+1)
  beta_0 = zeros(n_col+1)
  logΦ_0 = log([std(y); std(y)/sqrt(2); std(x,2)])
  theta_0 = zeros(n_col+1+length(logΦ_0))

  if !lasso
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

  # set start values
  theta_0 = [beta_0; logΦ_0]

  println("Start values for optimization:")	
  show(theta_0)	

  # set closures for objectivfunction and g!
  function objectivfunctionclosure(theta)
    return objectivfunction(x,X,y,kernel, theta, buffer)
  end

  function gclosure!(theta,stor)
    return g!(y,x,X, stor,buffer, theta)
  end

  # settings for the optimization algorithm
  options = Optim.Options(show_trace=true,iterations=1000,time_limit=60*10)

  # optimize parameters using conjugate gradient descent
  opt_res = Optim.optimize(objectivfunctionclosure,
                          gclosure!,
                          theta_0,
                          ConjugateGradient(),
                          options)

  # parameters found
  theta_hat = Optim.minimizer(opt_res)

  # set GPModel
  gp.est_method = "ml"
  gp.Β = theta_hat[1:length(beta_0)]
  gp.logΦ = theta_hat[end-length(logΦ_0)+1:end]
  gp.cov_inv = inv(covariancefunction(x,exp(theta_hat[end-length(logΦ_0)+1:end]),kernel))
  gp.diff = y-X*theta_hat[1:length(beta_0)]
  gp.x = x
  gp.indecies_to_include = indeices_keep

  println("Estimated parameters of the GP model:")
  println("Estimated Β:s:")
  println(theta_hat[1:length(beta_0)])
  println("Estimated log Φ:s:")
  println(theta_hat[end-length(logΦ_0)+1:end])

end


doc"""
    removeoutlier(y::Vector{Float64}, x::Array{Float64}, nbr_outliers::Int64=10, sided::String="both")

removeoutlier removes nbr_outliers outliers either from both tails or from left/rigth tail.
"""
function removeoutlier(y::Vector{Float64}, x::Array{Float64}, nbr_outliers::Int64=10, sided::String="both")

  idx_rm = Vector{Int64}
  idx = sortperm(y, rev=true)

  if sided == "both"
    # remove nbr_outliers on both tails
    idx_rm = [idx[1:nbr_outliers]; idx[end-(nbr_outliers-1):end]]
  elseif sided == "left"
    # remove nbr_outliers on left tail
    idx_rm = idx[end-(nbr_outliers-1):end]
  else
    # remove nbr_outliers on rigth tail
    idx_rm = idx[1:nbr_outliers]
  end

  return y[setdiff(1:end,idx_rm)], x[:,setdiff(1:end,idx_rm)]

end

doc"""
    RMSE(targets::Vector{Float64}, predictions::Vector{Float64})

Computes the root mean square error.

"""
function RMSE(targets::Vector{Float64}, predictions::Vector{Float64})
  return sqrt(mean((targets-predictions).^2))
end
