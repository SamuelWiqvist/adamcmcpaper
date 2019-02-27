# File used to print the posterior inference results for MCWM DA and ADA for
# both problems (real data and and sim data)

using DataFrames

include(pwd()*"/utilities/posteriorinference.jl")

remove_missing_values(x) = reshape(collect(skipmissing(x)),7,:)

problem = "real data"
problem = "sim data small problem"

algorithms = ["MCWM";"DA";"ADA"]

for algorithm in algorithms

    if problem == "real data" && algorithm == "MCWM"
        jobname = "gp_training_7_par_lunarc_new_data_4_coresnew_data"
    elseif problem == "real data" && algorithm == "DA"
        jobname = "_dagpest7new_datada_gp_mcmc"
    elseif problem == "real data" && algorithm == "ADA"
        jobname = "_dagpest7new_dataada_gp_mcmc_dt"
    elseif problem == "sim data small problem" && algorithm == "MCWM"
        jobname = "gp_training_7_par_lunarc_simdata_4_coressimdata"
    elseif problem == "sim data small problem" && algorithm == "DA"
        jobname = "_dagpest7simdatada_gp_mcmc"
    else #problem == "sim data scaled up problem" && algorithm = "ADA"
        jobname = "_dagpest7simdataada_gp_mcmc_dt"
    end

    if problem == "real data"
        plot_theta_true = false
    else
        plot_theta_true = true
    end

    data_res = convert(Array,readtable("DWPSDE model/Results/"*problem*"/output_res"*jobname*".csv"))

    M, N = size(data_res)

    data_param = convert(Array,readtable("DWPSDE model/Results/"*problem*"/output_param"*jobname*".csv"))

    theta_true = data_param[1:N-2]
    burn_in = Int64(data_param[N-2+1])

    data_prior_dist = convert(Array,readtable("DWPSDE model/Results/"*problem*"/output_prior_dist"*jobname*".csv"))

    data_prior_dist_type = convert(Array,readtable("DWPSDE model/Results/"*problem*"/output_prior_dist_type"*jobname*".csv"))
    data_prior_dist_type = data_prior_dist_type[2]

    Z = convert(Array,readtable("DWPSDE model/Results/"*problem*"/data_used"*jobname*".csv"))
    Z = Z[:,1]

    if algorithm == "DA" || algorithm == "ADA"
      burn_in = 1
    end

    Theta = remove_missing_values(data_res[:,1:N-2]') # stor data in column-major order
    loglik = data_res[:,N-1]
    accept_vec = data_res[:,N]

    accept_rate = sum(accept_vec)/M
    nbr_acf_lags =  50

    acf = ones(N-2,nbr_acf_lags+1)

    lower_q_int_limit = 2.5
    upper_q_int_limit = 97.5

    # use L"$\log r$"

    if N == 6
        title_vec_log = [ L"$\log \kappa$"; L"$\log \gamma$"; L"$\log c$"; L"$\log d$"]
        title_vec = [ L"$\kappa$"; L"Gamma"; "c"; "d"]
    elseif N == 4
        title_vec_log = [ L"$\log c$"; L"$\log d$" ]
        title_vec = [ "c"; "d" ];
    elseif N == 5
        title_vec_log = [ L"$\log A$";L"$\log c$"; L"$\log d$" ]
        title_vec = [ "A";"c"; "d" ]
    elseif N == 8
        title_vec_log = [ L"$\log A$"; L"$\log c$"; L"$\log d$"; L"$\log p_1$"; L"$\log p_2$"; L"$\log sigma$"]
        title_vec = [  "A    "; "c    "; "d    "; "p_1  "; "p_1  "; "sigma"]
    elseif N == 7
        title_vec_log = [ L"$\log \kappa$"; L"$\log Gamma$"; L"$\log c$"; L"$\log d$"; L"$\log sigma$"];
        title_vec = [ L"$\kappa$"; "Gamma"; "c"; "d"; "sigma"]
    elseif N == 9
        title_vec_log = [ L"$\log \kappa$"; L"$\log \gamma$"; L"$\log c$"; L"$\log d$"; L"$\log p_1$"; L"$\log p_2$"; L"$\log \sigma$"]
        title_vec = [  L"$\kappa$"; L"$\gamma$"; L"$c$"; L"$d$"; L"$p_1$"; L"$p_1$"; L"$\sigma$"]
    else
        title_vec_log = [ L"$\log \kappa$"; L"$\log Gamma$"; L"$\log A$"; L"$\log c$"; L"$\log d$"; L"$\log g$"; L"$\log p_1$"; L"$\log p_2$"; L"$\log sigma$"]
        title_vec = [ L"$\kappa$"; "Gamma"; "A"; "c"; "d"; "g"; "p_1"; "p_1"; "sigma"]
    end

    acceptance_rate = sum(accept_vec[burn_in:end])/length(accept_vec[burn_in:end])

    # print info
    println("---------------------------------------------------")
    print("Problem:")
    print(problem)
    println("")
    print("Algorithm:")
    print(algorithm)
    println("")
    println("")


    @printf "Accept rate: %.4f %% \n" round(acceptance_rate*100,2)

    println("")

    @printf "True parameter values:\n"
    Base.showarray(STDOUT,round.(theta_true,2),false)
    @printf "\n"

    println("")

    @printf "Posterior mean:\n"
    Base.showarray(STDOUT,round.(mean(Theta[:,burn_in+1:end],2),2),false)
    @printf "\n"

    println("")

    @printf "Posterior standard deviation:\n"
    Base.showarray(STDOUT,round.(std(Theta[:,burn_in+1:end],2),2),false)
    @printf "\n"

    println("")

    @printf "Posterior quantile intervals (2.5th and 97.5th quantiles as default):\n"
    Base.showarray(STDOUT,round.(calcquantileint(Theta[:,burn_in+1:end],lower_q_int_limit,upper_q_int_limit),2),false)
    @printf "\n"
    println("---------------------------------------------------")

end
