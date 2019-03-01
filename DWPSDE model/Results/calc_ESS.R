
library(mcmcse)
library(readr)

# Real data

# MCWM
output <- read_csv("real data/output_resgp_training_7_par_lunarc_new_data_4_coresnew_data.csv")
#View(output)
X = (output)[10000:20000,1:7]
t =  74795.716796139/2/60

ess = ess(X)
ess_min = min(ess)
ess_min_per_sec_mcwm = ess_min/t

# DA
output <- read_csv("real data/output_res_dagpest7new_datada_gp_mcmc.csv")
#View(output)
X = (output)[1:10000,1:7]
t =  9474.373923259/60

ess = ess(X)
ess_min = min(ess)
ess_min_per_sec_da = ess_min/t

# ADA
output <- read_csv("real data/output_res_dagpest7new_dataada_gp_mcmc_dt.csv")
#View(output)
X = (output)[1:10000,1:7]
t =  6053.148903795/60

ess = ess(X)
ess_min = min(ess)
ess_min_per_sec_ada = ess_min/t

# print
print(cat("real data, MCWM:", ess_min_per_sec_mcwm))
print(cat("real data, DA:", ess_min_per_sec_da))
print(cat("real data, ADA:", ess_min_per_sec_ada))


# sim data small problem 

# MCWM
output <- read_csv("sim data small problem/output_resgp_training_7_par_lunarc_simdata_4_coressimdata.csv")
#View(output)
X = (output)[10000:20000,1:7]
t =  74795.716796139/2/60

ess = ess(X)
ess_min = min(ess)
ess_min_per_sec_mcwm = ess_min/t

# DA
output <- read_csv("sim data small problem/output_res_dagpest7simdatada_gp_mcmc.csv")
#View(output)
X = (output)[1:10000,1:7]
t =  9474.373923259/60

ess = ess(X)
ess_min = min(ess)
ess_min_per_sec_da = ess_min/t

# ADA
output <- read_csv("sim data small problem/output_res_dagpest7simdataada_gp_mcmc_dt.csv")
#View(output)
X = (output)[1:10000,1:7]
t =  6053.148903795/60

ess = ess(X)
ess_min = min(ess)
ess_min_per_sec_ada = ess_min/t

# print
print(cat("real data, MCWM:", ess_min_per_sec_mcwm))
print(cat("real data, DA:", ess_min_per_sec_da))
print(cat("real data, ADA:", ess_min_per_sec_ada))

