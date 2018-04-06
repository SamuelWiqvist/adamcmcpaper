

library(mcmcse) # ESS package 
library(readr)
library(data.table) # to read large csv files fast 

# PMCMC/MCWM

output <- fread("Thetapmcmc.csv")

output <- fread("Thetamcwm.csv")


output <- t(output)
View(output)

X = output 
X = output[2001:nrow(output),]
View(X)

# time for pmcmc 
t = 859*(50/52)

# time for mcwm 
t = 1589*(50/52)


ess = ess(X)
ess_min = min(ess)
ess_min_per_sec = ess_min/t
print(ess_min_per_sec)


# DA/ADA

output <- fread("Theta_dagpmcmcMCWM.csv")

output <- fread("Theta_adagpmcmc_dt.csv")


output <- t(output)
View(output)

X = output 


# time DA 
t = 507


# time ADA 
t = 507



ess = ess(X)
ess_min = min(ess)
ess_min_per_sec = ess_min/t
print(ess_min_per_sec)
