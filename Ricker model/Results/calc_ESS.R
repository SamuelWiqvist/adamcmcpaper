

library(mcmcse) # ESS package 
library(readr)
library(data.table) # to read large csv files fast 

# PMCMC/MCWM

output <- fread("./Results/Thetascaling_mcwm_50.csv")



output <- t(output)
View(output)

X = output 
X = output[2001:nrow(output),]


t = 14.1118*50 


ess = ess(X)
ess_min = min(ess)
ess_min_per_sec = ess_min/t
print(ess_min_per_sec)


# DA/ADA

output <- fread("./Results/Theta_ergpaccelerated.csv")



output <- t(output)
View(output)

X = output 


t = 21.5205*50 


ess = ess(X)
ess_min = min(ess)
ess_min_per_sec = ess_min/t
print(ess_min_per_sec)
