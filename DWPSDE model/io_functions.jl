# functions for I/O operations of data and results

doc"""
    export_data(problem::Problem, results::Result, jobname::String)

Exportes the resultes into three cvs files

# Files
* `output_res.csv`: contines the estimations of Theta, loglik and the accept vec
* `output_prior_dist.csv`: contines parameters for the prior distributions.
* `output_prior_dist_type.csv`: contines information reg. the prior distribution.
* `data_used.csv`: contins the data set.
"""
function export_data(problem::Problem, results::Result, jobname::String)

  # set parameters
  theta_true = problem.model_param.theta_true
  burn_in = problem.alg_param.burn_in
  prior_parameters = problem.prior_dist.prior_parameters
  dist_type = problem.prior_dist.dist
  Theta_est = results.Theta_est'
  loglik_est = results.loglik_est
  accept_vec_est = results.accept_vec

  # export results
  data_param = zeros(1, length(theta_true)+1)
  data_param[1:length(theta_true)] = theta_true
  data_param[length(theta_true)+1:length(theta_true)+1] = burn_in
  writetable("./Results/output_param"*jobname*".csv", convert(DataFrame, data_param))

  data_output = zeros(length(loglik_est), length(theta_true) + 2)
  data_output[:,1:length(theta_true)] = Theta_est
  data_output[:,length(theta_true)+1:length(theta_true)+1] = loglik_est
  data_output[:,end] = accept_vec_est

  #
  data_used = zeros(length(problem.data.Z),2)
  data_used[:,1] = problem.data.Z
  writetable("./Results/output_res"*jobname*".csv", convert(DataFrame, data_output))

  writetable("./Results/output_prior_dist"*jobname*".csv", convert(DataFrame, prior_parameters))

  writetable("./Results/output_prior_dist_type"*jobname*".csv", convert(DataFrame, [1 dist_type]))

  writetable("./Results/data_used"*jobname*".csv", convert(DataFrame,data_used))

end

doc"""
    export_data(problem::gpProblem, results::Result, jobname::String)

Exportes the resultes into three cvs files

# Files
* `output_res.csv`: contines the estimations of Theta, loglik and the accept vec
* `output_prior_dist.csv`: contines parameters for the prior distributions.
* `output_prior_dist_type.csv`: contines information reg. the prior distribution.
* `data_used.csv`: contins the data set.
"""
function export_data(problem::gpProblem, results::Result, jobname::String)

  # set parameters
  theta_true = problem.model_param.theta_true
  burn_in = problem.alg_param.burn_in
  length_training_data = problem.alg_param.length_training_data
  prior_parameters = problem.prior_dist.prior_parameters
  dist_type = problem.prior_dist.dist
  Theta_est = results.Theta_est'
  loglik_est = results.loglik_est
  accept_vec_est = results.accept_vec

  # export results
  data_param = zeros(1, length(theta_true)+1)
  data_param[1:length(theta_true)] = theta_true
  data_param[length(theta_true)+1:length(theta_true)+1] = burn_in + length_training_data
  writetable("./Results/output_param_dagp"*jobname*".csv", convert(DataFrame, data_param))

  data_output = zeros(length(loglik_est), length(theta_true) + 2)
  data_output[:,1:length(theta_true)] = Theta_est
  data_output[:,length(theta_true)+1:length(theta_true)+1] = loglik_est
  data_output[:,end] = accept_vec_est

  #
  data_used = zeros(length(problem.data.Z),2)
  data_used[:,1] = problem.data.Z
  writetable("./Results/output_res_dagp"*jobname*".csv", convert(DataFrame, data_output))

  writetable("./Results/output_prior_dist_dagp"*jobname*".csv", convert(DataFrame, prior_parameters))

  writetable("./Results/output_prior_dist_type_dagp"*jobname*".csv", convert(DataFrame, [1 dist_type]))

  writetable("./Results/data_used_dagp"*jobname*".csv", convert(DataFrame,data_used))

end



doc"""
    export_parameters(lamba_t::Array)

Exportes the parmeters for the AM gen/AM gen cw algorithm into a cvs files

# Files
* `output_time_dep_scaling.csv`: contines the parameters values .

"""
function export_parameters(lamba_t::Array)

  writetable("output_time_dep_scaling.csv", convert(DataFrame, lamba_t))

end


doc"""
    load_data()

Loads the protien folding data into an array


# Outputs
* `Z_data`: array containing the data.
"""
function load_data()
  file = open("1LE1_L.dat")
  data = readlines(file)
  close(file)

  Z_data = zeros(1,length(data))

  for i = 1:length(data)
    Z_data[i] = readdlm(IOBuffer(data[i]), Float64)[2]
  end
  return Z_data
end


#=

doc"""
    export_pf_diagnostics(loglik::Array, weigths::Array, particels::Array, Z::Array, X::Array)

Exports data realated to the diagnostics res. for the pf to the files; `output_pf_diagnistics_loglik.csv`,
`output_pf_diagnistics_particles.csv`,`output_pf_diagnistics_particles.csv`, and `output_pf_diagnistics_Z_X.csv`.
"""
function export_pf_diagnostics(loglik::Array, weigths::Array, particels::Array, Z::Array, X::Array)

  writetable("output_pf_diagnistics_loglik.csv", convert(DataFrame, loglik))
  writetable("output_pf_diagnistics_weigths.csv", convert(DataFrame, weigths))
  writetable("output_pf_diagnistics_particles.csv", convert(DataFrame, particels))

  data_Z_X = zeros(length(Z), 2)
  data_Z_X[:,1] = Z'
  data_Z_X[:,2] = X'

  writetable("output_pf_diagnistics_Z_X.csv", convert(DataFrame, data_Z_X))

end

doc"""
    export_pf_diagnostics(loglik::Array, weigths::Array, particels::Array, Z::Array, X::Array)

Exports data realated to the diagnostics res. for the pf to the files; `output_pf_diagnistics_loglik.csv`,
`output_pf_diagnistics_particles.csv`,`output_pf_diagnistics_particles.csv`, and `output_pf_diagnistics_Z_X.csv`.
"""
function export_pf_diagnostics(loglik::RowVector, weigths::Array, particels::Array, Z::Array, X::Array)

  writetable("output_pf_diagnistics_loglik.csv", convert(DataFrame, loglik))
  writetable("output_pf_diagnistics_weigths.csv", convert(DataFrame, weigths))
  writetable("output_pf_diagnistics_particles.csv", convert(DataFrame, particels))

  data_Z_X = zeros(length(Z), 2)
  data_Z_X[:,1] = Z'
  data_Z_X[:,2] = X'

  writetable("output_pf_diagnistics_Z_X.csv", convert(DataFrame, data_Z_X))

end

=#
