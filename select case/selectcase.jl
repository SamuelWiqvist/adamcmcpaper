# This file contains the functions and types related to the models used to select which case to run in the A-DA algorithm

abstract type  CaseModel end

using DecisionTree
using GLM

# types for the models
type BiaseCoin <: CaseModel
    p::Vector # vector with the probabilites for selecting case 1,2,3, and 4
end

type LogisticRegression <: CaseModel
    β_for_model1or3::Vector # parameter vector for the logistic regression model for case 1 or 3
    β_for_model2or4::Vector # parameter vector for the logistic regression model  for case 1 or 3
end

type DT <: CaseModel
    decisiontree1or3::DecisionTree.Node # DT model  for case 1 or 3
    decisiontree2or4::DecisionTree.Node # DT model  for case 1 or 3
end

# functions for selecting case 1 or 3
"""
    selectcase1or3(model::BiaseCoin)

Returns 1 if case 1 is selected.
"""
function selectcase1or3(model::BiaseCoin, theta::Vector)

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
function selectcase1or3(model::LogisticRegression, theta::Vector)
    prob_case_1 = _predict(model.β_for_model1or2, [1;theta])
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
function selectcase1or3(model::DT, theta::Vector)

    if apply_tree(model.decisiontree1or3, theta) == "case 1"
        return 1
    else
        return 0
    end

end

# functions for selecting case 2 or 4
"""
    selectcase2or4(model::BiaseCoin)

Returns 1 if case 2 is selected.
"""
function selectcase2or4(model::BiaseCoin, theta::Vector)

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
function selectcase2or4(model::LogisticRegression, theta::Vector)
    prob_case_2 = _predict(model.β_for_model2or4, [1;theta])
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
function selectcase2or4(model::DT, theta::Vector)
    error("selectcase2or4(model::DT) is not implemented yet.")

    if apply_tree(model.decisiontree2or4, theta) == "case 2"
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
