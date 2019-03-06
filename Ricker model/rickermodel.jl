# This file contains the type and functions related to the Ricker model, also sets
# up the enviorment

# load packages
using Distributions
using DataFrames
using StatsBase
using Optim
#using Lasso
using GLM
using JLD
using HDF5

# load functions
#include("/home/samwiq/project 1 accalerated delated accaptance mcmc/gpmodel/gp_model.jl")
#include("/home/samwiq/project 1 accalerated delated accaptance mcmc/adaptive updating algorithms/adaptiveupdate.jl")
#include("/home/samwiq/project 1 accalerated delated accaptance mcmc/select case/selectcase.jl")

include(pwd()[1:end-13]*"/gpmodel/gp_model.jl")
include(pwd()[1:end-13]*"/adaptive updating algorithms/adaptiveupdate.jl")
include(pwd()[1:end-13]*"/select case/selectcase.jl")

# types:

"Type for prior distribution"
type PriorDistribution
  dist::String
  prior_parameters::Array{Float64}
end

"Type for the data"
type Data
  y::Array{Float64}
end

"Pararameters for the model"
type ModelParameters
  theta_true::Array{Float64} # [log(r) log(phi) log(sigma)]
  theta_known::Float64 # NaN
  theta_0::Array{Float64} # [log(r_0) log(phi_0) log(sigma_0)]
end

"Type for algorithm parameters for PMCMC"
type AlgorithmParameters
  R::Int64
  N::Int64
  burn_in::Int64
  alg::String
  pf_alg::String
  print_interval::Int64
end


"Type for algorithm parameters for gpPMCMC"
type AlgorithmParametersgpPMCMC
  R::Int64
  N::Int64
  burn_in::Int64
  length_training_data::Int64
  alg::String
  pf_alg::String
  print_interval::Int64
  est_method::String
  lasso::Bool
  pred_method::String
  independet_sampling::Bool
  noisy_est::Bool
  compare_GP_and_PF::Bool
  beta_MH::Float64
end


"Type for the problem (including algorithm parameters) for the PMCMC algorithm"
type Problem
  data::Data
  alg_param::AlgorithmParameters
  model_param::ModelParameters
  adaptive_update::AdaptationAlgorithm
  prior_dist::PriorDistribution
end

# this should be removed...
"Type for the problem (including algorithm parameters) for the gpPCMC algorithm"
type gpProblem
  data::Data
  alg_param::AlgorithmParametersgpPMCMC
  model_param::ModelParameters
  adaptive_update::AdaptationAlgorithm
  prior_dist::PriorDistribution
end

"Type for the results"
type Result
  Theta_est::Array{Float64}
  loglik_est::Array{Float64}
  accept_vec::Array{Float64}
  prior_vec::Array{Float64}
end

"Type for the results for the gpPCMC algorithm"
type gpResult
  Theta_est::Array{Float64}
  loglik_est::Array{Float64}
  accept_vec::Array{Float64}
  prior_vec::Array{Float64}
  compare_GP_PF::Array{Int64}
  data_gp_pf::Array{Float64}
  nbr_early_rejections::Int64
end


"type for the algorithm parameters for the ABC-MCMC algorithm"
type AlgorithmParametersABCMCMC
  R::Int64
  burn_in::Int64
  alg::String
  print_interval::Int64
  nbr_summary_stats::Int64
  eps::Vector
  start::String
end

"Type for the problem (including algorithm parameters) for the PMCMC algorithm"
type ProblemABCMCMC
  data::Data
  alg_param::AlgorithmParametersABCMCMC
  model_param::ModelParameters
  adaptive_update::AdaptationAlgorithm
  prior_dist::PriorDistribution
end

"Type for the results for the ABC-MCMC algorithm"
type ResultABCMCMC
  Theta_est::Array{Float64}
  accept_vec::Array{Float64}
  prior_vec::Array{Float64}
end

"Type for the parameters of the DA algorithm"
type DAParameters
  R::Int64 # nbr of DA iterations
  N::Int64 # length of training data
end


# load functions
include("mcmc.jl")

# set-up functions:

doc"""
    set_up_problem(use_sim_data::Bool=true;nbr_of_unknown_parameters::Int64=2,
  prior_dist::String = "Uniform", adaptiveon::Bool = true, ploton::Bool = false,
  nbr_of_cores::Int64 = 8, pf_algorithm::String = "bootstrap", T::Int64 = 50,x0::Float64 = 7.,
  print_interval::Int64 = 500, alg::String="PMCMC")

set_up_problem sets the parameters for the problem defined by the inputs.

# Output
* `problem`: An instance of the type Problem cointaining all information for the problem
"""
function set_up_problem(use_sim_data::Bool=true;nbr_of_unknown_parameters::Int64=2,
  prior_dist::String = "Uniform", adaptiveon::Bool = true, ploton::Bool = false,
  nbr_of_cores::Int64 = 8, pf_algorithm::String = "bootstrap", T::Int64 = 50,x0::Float64 = 7.,
  print_interval::Int64 = 500, alg::String="PMCMC")

  # set algorithm parameters
  theta_true = log([44.7; 10; 0.3])
  theta_0 =  log([3;3;10])
  Theta_parameters = [0 10; 0 4;-10 1]
  theta_known = NaN

  # create instance of AlgorithmParameters (set parameters to default values)
  alg_param = AlgorithmParameters(10000,500,5000,alg,pf_algorithm,print_interval)

  # create instance of ModelParameters, all theta paramters are on log-scale
  model_param = ModelParameters(theta_true,theta_known,theta_0)

  # set data
  if use_sim_data
    y = generate_data(T, exp(theta_true[1]),exp(theta_true[2]),exp(theta_true[3]),x0,ploton)
  end

  # create instance of Data
  data = Data(y)

  # create instance of PriorDistribution
  prior = PriorDistribution(prior_dist, Theta_parameters)

  # create instance of AdaptiveUpdate
  adaptive_update = AMUpdate(eye(length(theta_0)), 2.4/sqrt(length(theta_0)), 1., 0.7, 50)

  # return the an instance of Problem
  return Problem(data, alg_param, model_param, adaptive_update, prior)

end


doc"""
    set_up_gp_problem(use_sim_data::Bool=true;nbr_of_unknown_parameters::Int64=2,
  prior_dist::String = "Uniform", adaptiveon::Bool = true, ploton::Bool = false,
  nbr_of_cores::Int64 = 8, pf_algorithm::String = "bootstrap", est_method::String = "ml",lasso::Bool=true,
  independet_sampling::Bool = false, compare_GP_and_PF::Bool = true,noisy_est::Bool = true,
  pred_method::String = "sample",T::Int64 = 50,  x0::Float64 = 7., print_interval::Int64 = 500, nbr_predictions::Int64 = 25,
  selection_method::String="max_loglik", alg::String="PMCMC")

set_up_problem sets the parameters for the problem defined by the inputs.

# Output
* `problem`: An instance of the type gpProblem cointaining all information for the problem
"""
function set_up_gp_problem(use_sim_data::Bool=true;nbr_of_unknown_parameters::Int64=2,
  prior_dist::String = "Uniform", adaptiveon::Bool = true, ploton::Bool = false,
  nbr_of_cores::Int64 = 8, pf_algorithm::String = "bootstrap", est_method::String = "ml",lasso::Bool=true,
  independet_sampling::Bool = false, compare_GP_and_PF::Bool = true,noisy_est::Bool = true,
  pred_method::String = "sample",T::Int64 = 50,  x0::Float64 = 7., print_interval::Int64 = 500, alg::String="PMCMC", beta_MH::Float64=0.1)

  # set algorithm parameters
  theta_true = log([44.7; 10; 0.3])
  theta_0 =  log([3;3;10])
  Theta_parameters = [0 10; 0 4;-10 1]
  theta_known = NaN
  # create instance of AlgorithmParametersgpPMCMC (set parameters to default values)
  alg_param = AlgorithmParametersgpPMCMC(10000,500,2000,2000,alg,pf_algorithm,print_interval,est_method,lasso,pred_method,independet_sampling,noisy_est,compare_GP_and_PF,beta_MH)

  # create instance of ModelParameters, all theta paramters are on log-scale
  model_param = ModelParameters(theta_true,theta_known,theta_0)

  # set data
  if use_sim_data
    # load data from "data.csvexport_generated_data
    y = generate_data(T, exp(theta_true[1]),exp(theta_true[2]),exp(theta_true[3]),x0,ploton)
  end

 # create instance of Data
 data = Data(y)

  # create instance of PriorDistribution
  prior = PriorDistribution(prior_dist, Theta_parameters)

  # create instance of AdaptiveUpdate
  adaptive_update = AMUpdate(eye(length(theta_0)), 2.4/sqrt(length(theta_0)), 1., 0.7, 50)

  # return the an instance of gpProblem
  return gpProblem(data, alg_param, model_param, adaptive_update, prior)

end



#=
doc"""
    set_up_problem(use_sim_data::Bool=true;nbr_of_unknown_parameters::Int64=2,
  prior_dist::String = "Uniform", adaptiveon::Bool = true, ploton::Bool = false,
  nbr_of_cores::Int64 = 8, pf_algorithm::String = "bootstrap", T::Int64 = 50,x0::Float64 = 7.,
  print_interval::Int64 = 500, alg::String="PMCMC")

set_up_problem sets the parameters for the problem defined by the inputs.

# Output
* `problem`: An instance of the type Problem cointaining all information for the problem
"""
function set_up_abcmcmc_problem(use_sim_data::Bool=true;nbr_of_unknown_parameters::Int64=2,
  prior_dist::String = "Uniform", adaptiveon::Bool = true, ploton::Bool = false,
  nbr_of_cores::Int64 = 8, T::Int64 = 50,x0::Float64 = 7.,
  print_interval::Int64 = 500, alg::String="abcmcmc", nbr_summary_stats::Int64=5,
  eps::Float64=0.1)

  # set algorithm parameters
  theta_true = log([44.7; 10; 0.3])
  theta_0 =  log([3;3;10])
  Theta_parameters = [0 10; 0 4;-10 1]
  theta_known = NaN

  # create instance of AlgorithmParametersABCMCMC (set parameters to default values)
  alg_param = AlgorithmParametersABCMCMC(10000,2000,alg,print_interval,nbr_summary_stats,eps)

  # create instance of ModelParameters, all theta paramters are on log-scale
  model_param = ModelParameters(theta_true,theta_known,theta_0)

  # set data
  if use_sim_data
    y = generate_data(T, exp(theta_true[1]),exp(theta_true[2]),exp(theta_true[3]),x0,ploton)
  end

  # create instance of Data
  data = Data(y)

  # create instance of PriorDistribution
  prior = PriorDistribution(prior_dist, Theta_parameters)

  # create instance of AdaptiveUpdate
  adaptive_update = AMUpdate(eye(length(theta_0)), 2.4/sqrt(length(theta_0)), 1., 0.7, 50)

  # return the an instance of Problem
  return ProblemABCMCMC(data, alg_param, model_param, adaptive_update, prior)

end
=#

doc"""
    generate_data(T::Int65=50,r::Float64,phi::Float64,sigma::Float64,x0::Float64,ploton::Bool)


Generates data from the Ricker model.
"""
function generate_data(T::Int64,r::Float64,phi::Float64,sigma::Float64,x0::Float64=7.0,ploton::Bool=false)

  e = rand(Normal(0,sigma),T)
  y = zeros(T)
  x = zeros(T)

  # first iteration

  x[1] = r*x0*exp(-x0+e[1])

  y[1] = rand(Poisson(phi*x[1]))

  @simd for t = 2:T
    @inbounds x[t] = r*x[t-1]*exp(-x[t-1]+e[t])
    y[t] = rand(Poisson(phi*x[t]))
  end

  return y
end
