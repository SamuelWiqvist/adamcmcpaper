# load packages
using Distributions
using DataFrames
using StatsBase
import StatsBase.predict # this is needed to extend the predict function
using Optim
#using Lasso
using StatsFuns

# paths for desktop

# load functions
#include("/home/samwiq/project 1 accalerated delated accaptance mcmc/gpmodel/gp_model.jl")
#include("/home/samwiq/project 1 accalerated delated accaptance mcmc/adaptive updating algorithms/adaptiveupdate.jl")
#include("/home/samwiq/project 1 accalerated delated accaptance mcmc/select case/selectcase.jl")

include(pwd()*"/gpmodel/gp_model.jl")
include(pwd()*"/adaptive updating algorithms/adaptiveupdate.jl")
include(pwd()*"/select case/selectcase.jl")



"Type for prior distribution"
type PriorDistribution
  dist::String
  prior_parameters::Array{Float64}
end

"Type for the data"
type Data
  Z::Array{Float64}
  simualted::Bool
end

"Pararameters for the model"
type ModelParameters
  theta_true::Array{Float64}
  theta_known::Array{Float64}
  theta_0::Array{Float64}
end

"Type for algorithm parameters"
type AlgorithmParameters
  R::Int64
  N::Int64
  burn_in::Int64
  alg::String
  pf_alg::String
  nbr_of_cores::Int64
  nbr_x0::Int64
  nbr_x::Int64
  subsample_int::Int64
  dt::Float64
  dt_U::Float64
end

# old structure of code
"Type for algorithm parameters for gpPMCMC"
type AlgorithmParametersgpPMCMC
  R::Int64
  N::Int64
  burn_in::Int64
  alg::String
  pf_alg::String
  nbr_of_cores::Int64
  nbr_x0::Int64
  nbr_x::Int64
  subsample_int::Int64
  dt::Float64
  length_training_data::Int64
  est_method::String
  lasso::Bool
  pred_method::String
  independet_sampling::Bool
  noisy_est::Bool
  compare_GP_and_PF::Bool
  beta_MH::Float64
  dt_U::Float64
end


"Type for the problem (including algorithm parameters) "
type Problem
  data::Data
  alg_param::AlgorithmParameters
  model_param::ModelParameters
  adaptive_update::AdaptationAlgorithm
  prior_dist::PriorDistribution
end


# old structure of code
"Type for the problem (including algorithm parameters) for the gpPCMC algorithm"
type gpProblem
  data::Data
  alg_param::AlgorithmParametersgpPMCMC
  model_param::ModelParameters
  adaptive_update::AdaptationAlgorithm
  prior_dist::PriorDistribution
end




"Type for the parameters of the DA part of the DA-GP-MCMC algorithm"
type DAParameters
  R::Int64
  N::Int64
  burn_in::Int64
  alg::String
  pf_alg::String
  nbr_of_cores::Int64
  nbr_x0::Int64
  nbr_x::Int64
  subsample_int::Int64
  dt::Float64
  length_training_data::Int64
  est_method::String
  lasso::Bool
  pred_method::String
  independet_sampling::Bool
  noisy_est::Bool
  compare_GP_and_PF::Bool
  nbr_predictions::Int64
  selection_method::String
  beta_MH::Float64
end



"Type for the results"
type Result
  Theta_est::Array
  loglik_est::Array
  accept_vec::Array
  prior_vec::Array
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


include("mcmc.jl")
include("io_functions.jl")
include("generate_data.jl")


doc"""
    set_up_problem(use_sim_data::Bool=true;nbr_of_unknown_parameters::Int64=2,
  prior_dist::String = "Normal", ploton::Bool = false,
  nbr_of_cores::Int64 = 8, alg::String= "MCWM", pf_algorithm::String = "parallel_bootstrap")

set_up_problem sets the parameters for the problem defined by the inputs.

# Output
* `problem`: An instance of the type Problem cointaining all information for the problem
"""
function set_up_problem(;use_sim_data::Bool=true,nbr_of_unknown_parameters::Int64=2,
  prior_dist::String = "Normal", ploton::Bool = false,
  nbr_of_cores::Int64 = 8, alg::String= "MCWM", pf_algorithm::String = "parallel_bootstrap", data_set::String = "old", dt_U::Float64=1.)



  # set algorithm parameters
  (theta_true,theta_known,theta_0,prior_parameters) = set_parameters(nbr_of_unknown_parameters,prior_dist, data_set)



  dt = 0.01

  if data_set == "new"
    dt = 0.005
  end

  # create instance of AlgorithmParameters (set parameters to default values)
  alg_param = AlgorithmParameters(10000,25,5000,alg, pf_algorithm,nbr_of_cores,1,1,1,dt,dt_U)

  # create instance of ModelParameters
  model_param = ModelParameters(theta_true,theta_known,theta_0)

  # set data
  if use_sim_data
    if data_set == "old"
      # load simulated data from "data.csv"
      #Z_df = readtable("data.csv")
      Z_df = readtable("data_old_new_dt.csv")
      Z = convert(Array, Z_df)
    else
      # load simulated data from "data.csv"
      Z_df = readtable("data_new_new.csv")
      Z = convert(Array, Z_df)
    end
  else
    if data_set == "old"
      Z = load_data()
    else
      # load data
      file = open("new_data_set.txt")
      data = readlines(file)
      close(file)

      Z_data = zeros(length(data)-1)

      idx = 2:length(data)

      for i in idx
        try
          Z_data[i-1] = readdlm(IOBuffer(data[i]), Float64)[2]
        catch
          Z_data[i-1] = parse(data[i][end-1-4:end-1])
        end
      end

      # linear transformation of data to obtain a scaling where it is easier to
      # construct the dwp model
      Z_data = 50*Z_data

      # thinned data
      thinning = 100
      idx_thinned = 1:thinning:length(Z_data)
      Z = Z_data[idx_thinned] # zeros(Float64, idx_thinned)
      Z = Z[1:24800]

    end
  end

 # create instance of Data
 data = Data(Z, use_sim_data)

 # plot data
 if ploton
   # no plotting
   #PyPlot.figure()
   #PyPlot.plot(Z)
 end

  # create instance of PriorDistribution
  prior = PriorDistribution(prior_dist, prior_parameters)

  # create instance of AdaptiveUpdate
  adaptive_update = AMUpdate(eye(length(theta_0)), 2.38/sqrt(length(theta_0)), 1, 0.7, 25)

  # return the an instance of Problem
  return Problem(data, alg_param, model_param, adaptive_update, prior)

end




doc"""
    set_up_gp_problem(use_sim_data::Bool=true;nbr_of_unknown_parameters::Int64=2,
  prior_dist::String = "Normal", ploton::Bool = false, nbr_of_cores::Int64 = 8, alg::String= "MCWM",
  pf_algorithm::String = "parallel_bootstrap",length_training_data::Int64=2000, est_method::String="ml",
  lasso::Bool=true, pred_method::String="sample",independet_sampling::Bool=false,noisy_est::Bool=false,
  compare_GP_and_PF::Bool=false, nbr_predictions::Int64=25, selection_method::String="max_loglik")

set_up_gp_problem sets the parameters for the problem defined by the inputs.

# Output
* `problem`: An instance of the type gpProblem cointaining all information for the problem
"""
function set_up_gp_problem(;use_sim_data::Bool=true,nbr_of_unknown_parameters::Int64=2,
  prior_dist::String = "Normal", ploton::Bool = false, nbr_of_cores::Int64 = 8, alg::String= "MCWM",
  pf_algorithm::String = "parallel_bootstrap",length_training_data::Int64=2000, est_method::String="ml",
  lasso::Bool=true, pred_method::String="sample",independet_sampling::Bool=false,noisy_est::Bool=false,
  compare_GP_and_PF::Bool=false, beta_MH::Float64=0.1,data_set::String = "old", dt_U::Float64=1.,)


  # set algorithm parameters
  (theta_true,theta_known,theta_0,prior_parameters) = set_parameters(nbr_of_unknown_parameters,prior_dist,data_set)

  # create instance of AlgorithmParametersgpPMCMC (set parameters to default values)
  alg_param = AlgorithmParametersgpPMCMC(10000,25,5000,alg, pf_algorithm,nbr_of_cores,
  1,1,1,0.01,length_training_data,est_method,lasso,pred_method,independet_sampling,noisy_est,compare_GP_and_PF,beta_MH,  dt_U)

  # create instance of ModelParameters
  model_param = ModelParameters(theta_true,theta_known,theta_0)

  # set data
  if use_sim_data
    # use simulated data
    if data_set == "old"
      # load simulated data from "data.csv"
      #Z_df = readtable("data.csv")
      Z_df = readtable("data_old_new_dt.csv")
      Z = convert(Array, Z_df)
    else
      # load simulated data from "data.csv"
      Z_df = readtable("data_new.csv")
      Z = convert(Array, Z_df)
    end
  else
    # use real data
    if data_set == "old"
      Z = load_data() # load the old data set
    else
      # load data
      println("test")
      file = open("new_data_set.txt") # load the new data set
      data = readlines(file)
      close(file)

      Z_data = zeros(length(data)-1)

      idx = 2:length(data)

      for i in idx
        try
          Z_data[i-1] = readdlm(IOBuffer(data[i]), Float64)[2]
        catch
          Z_data[i-1] = parse(data[i][end-1-4:end-1])
        end
      end

      # linear transformation of data to obtain a scaling where it is easier to
      # construct the dwp model
      Z_data = 50*Z_data

      # thinned data
      thinning = 100
      idx_thinned = 1:thinning:length(Z_data)
      Z = Z_data[idx_thinned] # zeros(Float64, idx_thinned)
      Z = Z[1:24800]

    end
  end

 # create instance of Data
 data = Data(Z, use_sim_data)

 # plot data
 if ploton
   # no plotting
   #PyPlot.figure()
   #PyPlot.plot(Z)
 end

  # create instance of PriorDistribution
  prior = PriorDistribution(prior_dist, prior_parameters)

  # create instance of AdaptiveUpdate
  adaptive_update = AMUpdate(eye(length(theta_0)), 2.38/sqrt(length(theta_0)), 1, 0.7, 25)

  # return the an instance of Problem
  return gpProblem(data, alg_param, model_param, adaptive_update, prior)

end

doc"""
    set_parameters(nbr_of_unknown_parameters,prior_dist)

Sets the parameters for the prior dist. and the model parameters theta.
"""
 function set_parameters(nbr_of_unknown_parameters,prior_dist,data_set = "old")

  # true parameter valuse
  if data_set == "old"

    Κ = 0.3
    Γ = 0.9
    B = 1.
    c = 28.5
    d =  4.
    A = 0.01
    A_sign = 1
    f = 0.
    g = 0.03
    power1 = 1.5
    power2 = 1.8
    sigma =  1.9

    if nbr_of_unknown_parameters == 2 # set parameters for estimating 2 parameters

      # estimating c and d

      theta_true = log.([c d]) # true values for the unknown parameters
      theta_known = [Κ  Γ A A_sign B f g power1 power2 sigma] # set vector with known parameters
      if prior_dist == "Uniform"
        # uniform prior dist
        prior_parameters =  [log(20) log(40); 0 log(10)]
      elseif prior_dist == "Normal"
        # normal prior dist
        prior_parameters = [3.34 0.173; 1.15 0.2]
      elseif prior_dist == "nonlog"
        # prior distribution on non-log scale
        prior_parameters = [28 2; 4 1]
      end


      theta_0 = [log(100) log(40)] #theta_true #[log(100) log(40)] #theta_true #[log(100) log(40)] # theta_truestart values

    elseif nbr_of_unknown_parameters == 7

      # estimate Κ,Γ,c,d,power1,power2,sigma

      theta_true = log.([Κ,Γ,c,d,power1,power2,sigma]) # true values for the unknown parameters
      theta_known = [A, A_sign,B,f,g] # set vector with known parameters

      if prior_dist == "Uniform"
        # uniform prior dist
        prior_parameters =  [-10 2; -10 2;log(20) log(40);  0 log(10); -10 0.7; -10 0.7; log(1) log(4)]
      elseif prior_dist == "Normal"
        # normal prior dist
        prior_parameters = [-0.7 0.5;-0.7 0.5;3.34 0.173; 1.15 0.2;0 0.5;0 0.5;log(2) 0.5]
      elseif prior_dist == "nonlog"
        # prior distribution on non-log scale
        prior_parameters = [3 1.5; 3 1.5; 28 2; 4 1; 2 2; 2 2; 2 2]
      end

      theta_0 = [log(2) log(2) log(30) log(10) log(2) log(2) log(2)] #[log(1) log(1) log(30) log(5) log(2) log(2) log(1)] #theta_true #[log(1) log(30) log(4) log(2) log(2) log(2)]  #log([0.0001, 1, 1, 0.0001, 0.0001, 0.01]) # start values
    else

      error("selected wrong number of unknown parameters")

    end

    # return parameters
    return theta_true,theta_known,theta_0,prior_parameters

  else

    Κ = 0.7
    Γ = 2.1
    B = 1.
    c = 22.5
    d =  13.
    A = 0.0025
    A_sign = -1
    f = 0.
    g = 0.
    power1 = 1.3
    power2 = 1.3
    sigma =  2.6

    # not here we have only updated the case with Normal priors

    if nbr_of_unknown_parameters == 2 # set parameters for estimating 2 parameters

      # estimating c and d


      theta_true = log.([c d]) # true values for the unknown parameters
      theta_known = [Κ  Γ A A_sign B f g power1 power2 sigma] # set vector with known parameters

      if prior_dist == "Uniform"
        # uniform prior dist
        prior_parameters =  [log(10) log(40); 0 log(20)]
      elseif prior_dist == "Normal"
        # normal prior dist
        prior_parameters = [3.34 0.173; 2.3 0.4]
      elseif prior_dist == "nonlog"
        # prior distribution on non-log scale
        prior_parameters = [28 2; 4 1]
      end


      theta_0 = [log(100) log(40)] #theta_true #[log(100) log(40)] #theta_true #[log(100) log(40)] # theta_truestart values


    elseif nbr_of_unknown_parameters == 7

      # estimate Κ,Γ,c,d,power1,power2,sigma

      theta_true = log.([Κ,Γ,c,d,power1,power2,sigma]) # true values for the unknown parameters
      theta_known = [A, A_sign,B,f,g] # set vector with known parameters

      if prior_dist == "Uniform"
        # uniform prior dist
        prior_parameters =  [-10 2; -10 2;log(20) log(40);  0 log(10); -10 0.7; -10 0.7; log(1) log(4)]
      elseif prior_dist == "Normal"
        # normal prior dist
        prior_parameters = [-0.7 0.8;-0.7 0.8;3.34 0.173; 2.3 0.4;0 0.5;0 0.5;log(2) 0.5]
      elseif prior_dist == "nonlog"
        # prior distribution on non-log scale
        prior_parameters = [3 1.5; 3 1.5; 28 2; 4 1; 2 2; 2 2; 2 2]
      end

      #theta_0 = [log(10) log(10) log(100) log(40) log(2) log(2) log(2)]# [log(2) log(2) log(30) log(10) log(2) log(2) log(2)] #[log(1) log(1) log(30) log(5) log(2) log(2) log(1)] #theta_true #[log(1) log(30) log(4) log(2) log(2) log(2)]  #log([0.0001, 1, 1, 0.0001, 0.0001, 0.01]) # start values

      theta_0 = [log(0.5) log(2) log(20) log(15) log(1.5) log(1.5) log(2.5)] #[log(1) log(1) log(30) log(5) log(2) log(2) log(1)] #theta_true #[log(1) log(30) log(4) log(2) log(2) log(2)]  #log([0.0001, 1, 1, 0.0001, 0.0001, 0.01]) # start values

    else

      error("selected wrong number of unknown parameters")

    end
    # return parameters
    return theta_true,theta_known,theta_0,prior_parameters


  end

end
