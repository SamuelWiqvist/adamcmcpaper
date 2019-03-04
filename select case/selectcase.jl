# This file contains the functions and types related to the models used to select which case to run in the A-DA algorithm

abstract CaseModel

using DecisionTree
using GLM

# types for the models
type BiasedCoin <: CaseModel
  p::Vector # vector with the probabilites for selecting case 1,2,3, and 4
end

type LogisticRegression <: CaseModel
  β_for_model1or3::Vector # parameter vector for the logistic regression model for case 1 or 3
  β_for_model2or4::Vector # parameter vector for the logistic regression model  for case 1 or 3
  mean_posterior::Vector
end

type DT <: CaseModel
  decisiontree1or3::DecisionTree.Node # DT model  for case 1 or 3
  decisiontree2or4::DecisionTree.Node # DT model  for case 1 or 3
end

# functions for selecting case 1 or 3
"""
    selectcase1or3(model::BiasedCoin)

Returns 1 if case 1 is selected.
"""
function selectcase1or3(model::BiasedCoin, theta::Vector, prediction_sample_ml_star::Real, prediction_sample_ml_old::Real)

  if rand(Bernoulli(model.p[1])) == 1
    return 1
  else
    return 0
  end

end

"""
    selectcase1or3(model::LogisticRegression)

Returns 1 if case 1 is selected.
"""
function selectcase1or3(model::LogisticRegression, theta::Vector, prediction_sample_ml_star::Real, prediction_sample_ml_old::Real)

  # create new obs using theta

  theta_log_reg_mod = zeros(length(theta)+2)
  model.mean_posterior
  for i = 1:length(theta)
    theta_log_reg_mod[i] = sqrt((model.mean_posterior[i] - theta[i]).^2)
  end

  theta_log_reg_mod[length(theta)+1] = sqrt(sum((mean_posterior - theta).^2))
  theta_log_reg_mod[length(theta)+2] = prediction_sample_ml_star/prediction_sample_ml_old


  prob_case_1 = _predict(model.β_for_model1or3, [1;theta_log_reg_mod])
  if rand(Bernoulli(prob_case_1)) == 1
    return 1
  else
    return 0
  end
end

"""
    selectcase1or3(model::DT)

Returns 1 if case 1 is selected.
"""
function selectcase1or3(model::DT, theta::Vector, prediction_sample_ml_star::Real, prediction_sample_ml_old::Real)


  theta_tree_model = zeros(length(theta)+1)
  theta_tree_model = [theta; prediction_sample_ml_star/prediction_sample_ml_old]

  if apply_tree(model.decisiontree1or3, theta_tree_model) == "case 1"
    return 1
  else
    return 0
  end

end

# functions for selecting case 2 or 4
"""
    selectcase2or4(model::BiasedCoin)

Returns 1 if case 2 is selected.
"""
function selectcase2or4(model::BiasedCoin, theta::Vector, prediction_sample_ml_star::Real, prediction_sample_ml_old::Real)

  if rand(Bernoulli(model.p[2])) == 1
    return 1
  else
    return 0
  end

end

"""
    selectcase2or4(model::LogisticRegression)

Returns 1 if case 2 is selected.
"""
function selectcase2or4(model::LogisticRegression, theta::Vector, prediction_sample_ml_star::Real, prediction_sample_ml_old::Real)

  # create new obs using theta

  theta_log_reg_mod = zeros(length(theta)+2)
  model.mean_posterior
  for i = 1:length(theta)
    theta_log_reg_mod[i] = sqrt((model.mean_posterior[i] - theta[i]).^2)
  end

  theta_log_reg_mod[length(theta)+1] = sqrt(sum((mean_posterior - theta).^2))
  theta_log_reg_mod[length(theta)+2] = prediction_sample_ml_star/prediction_sample_ml_old

  prob_case_2 = _predict(model.β_for_model2or4, [1;theta_log_reg_mod])
  if rand(Bernoulli(prob_case_2)) == 1
    return 1
  else
    return 0
  end
end

"""
    selectcase2or4(model::DT)

Returns 1 if case 2 is selected.
"""
function selectcase2or4(model::DT, theta::Vector,prediction_sample_ml_star::Real, prediction_sample_ml_old::Real)

  theta_tree_model = zeros(length(theta)+1)
  theta_tree_model = [theta; prediction_sample_ml_star/prediction_sample_ml_old]

  if apply_tree(model.decisiontree2or4, theta_tree_model) == "case 2"
    return 1
  else
    return 0
  end

end

# help functions

doc"""
Prediction for the logistic regression model at theta.
"""
function _predict(beta::Vector, theta::Vector)

  p_hat =  exp(theta'*beta)./(1+exp(theta'*beta))
  return p_hat[1]

end
