# This file contains the functions and types related to the

abstract type  AdaptationAlgorithm end 

type noAdaptation <: AdaptationAlgorithm
  Cov::Array{Float64}
end

type AMUpdate <: AdaptationAlgorithm
  C_0::Array{Float64}
  r_cov_m::Float64
  gamma_0::Float64
  k::Float64
  t_0::Int64
end

type AMUpdate_gen <: AdaptationAlgorithm
  C_0::Array{Float64}
  r_cov_m::Float64
  P_star::Float64
  gamma_0::Float64
  k::Float64
  t_0::Int64
end


# Function for different updating algorithms

# set up functions

"Set parameters for noAdaptation"
function set_adaptive_alg_params(algorithm::noAdaptation, nbr_of_unknown_parameters::Int64, theta::Vector,R::Int64)

  return (algorithm.Cov, NaN)

end


"Set parameters for AMUpdate"
function set_adaptive_alg_params(algorithm::AMUpdate, nbr_of_unknown_parameters::Int64, theta::Vector,R::Int64)

  # define m_g m_g_1
  m_g = zeros(nbr_of_unknown_parameters,1)
  m_g_1 = zeros(nbr_of_unknown_parameters,1)

  return (algorithm.C_0, algorithm.gamma_0, algorithm.k, algorithm.t_0, [algorithm.r_cov_m], m_g, m_g_1)

end

"Set parameters for AMUpdate_gen"
function set_adaptive_alg_params(algorithm::AMUpdate_gen, nbr_of_unknown_parameters::Int64, theta::Vector,R::Int64)

  Cov_m = algorithm.C_0
  log_r_cov_m = log(algorithm.r_cov_m)
  log_P_star = log(algorithm.P_star)
  gamma_0 = algorithm.gamma_0
  k = algorithm.k
  t_0 = algorithm.t_0
  vec_log_r_cov_m = zeros(1,R)
  vec_log_r_cov_m[1:t_0] = log_r_cov_m
  # define m_g m_g_1
  m_g = zeros(nbr_of_unknown_parameters,1)
  m_g_1 = zeros(nbr_of_unknown_parameters,1)
  #diff_a_log_P_start = 1.
  #factor = 1.
  #gain = 1.

  return (Cov_m, log_P_star, gamma_0, k, t_0, vec_log_r_cov_m, [log_r_cov_m], m_g, m_g_1)

end



# return covariance functions

doc"""
    return_covariance_matrix(algorithm::noAdaptation, adaptive_update_params::Tuple,r::Int64)
"""
function return_covariance_matrix(algorithm::noAdaptation, adaptive_update_params::Tuple,r::Int64)

  return adaptive_update_params[1]

end

doc"""
    return_covariance_matrix(algorithm::AMUpdate, adaptive_update_params::Tuple,r::Int64)

Print function for AMUpdate.
"""
function return_covariance_matrix(algorithm::AMUpdate, adaptive_update_params::Tuple,r::Int64)
    r_cov_m = adaptive_update_params[end-2][1]
    Cov_m = adaptive_update_params[1]
    return r_cov_m^2*Cov_m
end

doc"""
    return_covariance_matrix(algorithm::AMUpdate_gen, adaptive_update_params::Tuple,r::Int64)

Print function for AMUpdate_gen.
"""
function return_covariance_matrix(algorithm::AMUpdate_gen, adaptive_update_params::Tuple,r::Int64)
    log_r_cov_m = adaptive_update_params[end-2][1]
    Cov_m = adaptive_update_params[1]
    return exp(log_r_cov_m)^2*Cov_m
end



# print_covariance functions

doc"""
    print_covariance(algorithm::noAdaptation, adaptive_update_params::Tuple,r::Int64)
"""
function print_covariance(algorithm::noAdaptation, adaptive_update_params::Tuple,r::Int64)

  println(adaptive_update_params[1])

end


doc"""
    print_covariance(algorithm::AMUpdate, adaptive_update_params::Tuple,r::Int64)
"""
function print_covariance(algorithm::AMUpdate, adaptive_update_params::Tuple,r::Int64)
    r_cov_m = adaptive_update_params[end-2][1]
    Cov_m = adaptive_update_params[1]
    println(r_cov_m^2*Cov_m)
end

doc"""
    print_covariance(algorithm::AMUpdate_gen, adaptive_update_params::Tuple,r::Int64)
"""
function print_covariance(algorithm::AMUpdate_gen, adaptive_update_params::Tuple,r::Int64)
    log_r_cov_m = adaptive_update_params[end-2][1]
    Cov_m = adaptive_update_params[1]
    println(exp(log_r_cov_m)^2*Cov_m)
end

# get_covariance functions


doc"""
    get_covariance(algorithm::noAdaptation, adaptive_update_params::Tuple,r::Int64)
"""
function get_covariance(algorithm::noAdaptation, adaptive_update_params::Tuple,r::Int64)

  return adaptive_update_params[1]

end


doc"""
    get_covariance(algorithm::AMUpdate, adaptive_update_params::Tuple,r::Int64)
"""
function get_covariance(algorithm::AMUpdate, adaptive_update_params::Tuple,r::Int64)
    r_cov_m = adaptive_update_params[end-2][1]
    Cov_m = adaptive_update_params[1]
    return r_cov_m^2*Cov_m
end

doc"""
    get_covariance(algorithm::AMUpdate_gen, adaptive_update_params::Tuple,r::Int64)
"""
function get_covariance(algorithm::AMUpdate_gen, adaptive_update_params::Tuple,r::Int64)
    log_r_cov_m = adaptive_update_params[end-2][1]
    Cov_m = adaptive_update_params[1]
    return exp(log_r_cov_m)^2*Cov_m
end

# gaussian random walk functions

doc"""
    gaussian_random_walk(algorithm::noAdaptation, adaptive_update_params::Tuple, Theta::Vector, r::Int64)
"""
function gaussian_random_walk(algorithm::noAdaptation, adaptive_update_params::Tuple, Theta::Vector, r::Int64)

  return rand(MvNormal(Theta, adaptive_update_params[1])), zeros(size(Theta))

end


doc"""
    gaussian_random_walk(algorithm::AMUpdate, adaptive_update_params::Tuple, Theta::Vector, r::Int64)
"""
function gaussian_random_walk(algorithm::AMUpdate, adaptive_update_params::Tuple, Theta::Vector, r::Int64)
  r_cov_m = adaptive_update_params[end-2][1]
  Cov_m = adaptive_update_params[1]
  return rand(MvNormal(Theta, r_cov_m^2*Cov_m)), zeros(size(Theta))
end

doc"""
    gaussian_random_walk(algorithm::AMUpdate_gen, adaptive_update_params::Tuple, Theta::Vector, r::Int64)

RW for AMUpdate_gen.
"""
function gaussian_random_walk(algorithm::AMUpdate_gen, adaptive_update_params::Tuple, Theta::Vector, r::Int64)
  log_r_cov_m = adaptive_update_params[end-2][1]
  Cov_m = adaptive_update_params[1]
  return rand(MvNormal(Theta, exp(log_r_cov_m)^2*Cov_m)), zeros(size(Theta))
end


# functions for adaptation of parameters
doc"""
    adaptation(algorithm::noAdaptation, adaptive_update_params::Tuple, Theta::Array, r::Int64,a_log::Float64)
"""
function adaptation(algorithm::noAdaptation, adaptive_update_params::Tuple, Theta::Array, r::Int64,a_log::Float64)

  # do nothing

end



doc"""
    adaptation(algorithm::AMUpdate, adaptive_update_params::Tuple, Theta::Array, r::Int64,a_log::Float64)
"""
function adaptation(algorithm::AMUpdate, adaptive_update_params::Tuple, Theta::Array, r::Int64,a_log::Float64)

  Cov_m = adaptive_update_params[1]
  m_g = adaptive_update_params[end-1]
  m_g_1 = adaptive_update_params[end]
  k = adaptive_update_params[3]
  gamma_0 = adaptive_update_params[2]
  t_0 = adaptive_update_params[4]

  g = r-1
  if r-1 == t_0
      m_g = mean(Theta[:,1:r-1],2)
      m_g_1 = m_g + gamma_0/( (g+1)^k ) * (Theta[:,g+1] - m_g)
      adaptive_update_params[1][:] = Cov_m + ( gamma_0/( (g+1)^k ) ) * ( (Theta[:, g+1] - m_g)*(Theta[:,g+1] - m_g)'     -  Cov_m)
      adaptive_update_params[end-1][:] = m_g_1
  elseif r-1 > t_0
      m_g_1 = m_g + gamma_0/( (g+1)^k ) * (Theta[:,g+1] - m_g)
      adaptive_update_params[1][:] = Cov_m + ( gamma_0/( (g+1)^k ) ) * ( (Theta[:, g+1] - m_g)*(Theta[:,g+1] - m_g)'     -  Cov_m)
      adaptive_update_params[end-1][:] = m_g_1
  end

end

doc"""
    adaptation(algorithm::AMUpdate_gen, adaptive_update_params::Tuple, Theta::Array, r::Int64,a_log::Float64)
"""
function adaptation(algorithm::AMUpdate_gen, adaptive_update_params::Tuple, Theta::Array, r::Int64, a_log::Float64)

  Cov_m = adaptive_update_params[1]
  m_g = adaptive_update_params[end-1]
  m_g_1 = adaptive_update_params[end]
  log_P_star = adaptive_update_params[2]
  log_r_cov_m = adaptive_update_params[end-2][1]
  k = adaptive_update_params[4]
  gamma_0 = adaptive_update_params[3]
  t_0 = adaptive_update_params[5]

  g = r-1;
  if r-1 >= t_0
    diff_a_log_P_start = abs(min(1, exp(a_log)) - exp(log_P_star)) #abs( min(log(1), a_log) - log_P_star)
    factor = 1 #diff_a_log_P_start/abs(log_r_cov_m)
    gain = diff_a_log_P_start/factor
    if min(1, exp(a_log)) < exp(log_P_star) #min(log(1), a_log) < log_P_star
      sign_val = -1
    else
      sign_val = 1
    end
    adaptive_update_params[end-2][1] = log_r_cov_m + sign_val * ( gamma_0/( (g+1)^k ) ) * gain
    adaptive_update_params[6][g+1] = log_r_cov_m
    if r-1 == t_0
      m_g = mean(Theta[:,1:r-1],2)
      m_g_1 = m_g + ( gamma_0/( (g+1)^k ) ) * (Theta[:,g+1] - m_g)
      adaptive_update_params[1][:] = Cov_m + ( gamma_0/( (g+1)^k ) ) * ( (Theta[:,g+1] - m_g)*(Theta[:,g+1] - m_g)'     -  Cov_m)
      adaptive_update_params[end-1][:] = m_g_1
    elseif r-1 > t_0
      m_g_1 = m_g + gamma_0/( (g+1)^k ) * (Theta[:,g+1] - m_g)
      adaptive_update_params[1][:] = Cov_m + ( gamma_0/( (g+1)^k ) ) * ( (Theta[:,g+1] - m_g)*(Theta[:,g+1] - m_g)'     -  Cov_m)
      adaptive_update_params[end-1][:] = m_g_1
    end
  end

end
