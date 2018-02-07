
library(mcmcse)
library(readr)


# MCWM/PMCMC 
output <- read_csv("C:/Users/samuel/Dropbox/Phd Education/LUNARC/DWPSDE model/Results/output_resgp_training_7_par.csv")
View(output)

X = (output)[10000:20000,1:7]
t =  203.5073*10


ess = ess(X)
ess_min = min(ess)
ess_min_per_sec = ess_min/t
print(ess_min_per_sec)




# DA/ADA
output <- read_csv("C:/Users/samuel/Dropbox/Phd Education/LUNARC/DWPSDE model/Results/output_res_ergpest7_betamh_01ada_gp_mcmc.csv")
View(output)

X = (output)[1:10000,1:7]
t =  39.7483*10


ess = ess(X)
ess_min = min(ess)
ess_min_per_sec = ess_min/t
print(ess_min_per_sec)


# dagp  
X = output_res_ergpest5_R5000_N25_25_direct_MH_desktop[11001:16000,1:5]
t =  3154

# load data for Ricker model 
Theta_trans100000iter_2nd_run <- read_csv("C:/Users/samuel/Dropbox/Phd Education/LUNARC/Ricker model est 3 parameters v2/Results/Theta_trans100000iter_2nd_run.csv")
View(Theta_trans100000iter_2nd_run)

Theta_ergp_trans100000iter_2nd_run <- read_csv("C:/Users/samuel/Dropbox/Phd Education/LUNARC/Ricker model est 3 parameters v2/Results/Theta_ergp_trans100000iter_2nd_run.csv")
View(Theta_ergp_trans100000iter_2nd_run)

# mcwm 
X = Theta_trans100000iter_2nd_run[2001:102000,1:3]
t =  21.1*100


# dagp  
X = Theta_ergp_trans100000iter_2nd_run[4001:104000,1:3]
t =  28.6*100


ess = ess(X)
ess_min = min(ess)
ess_min_per_sec = ess_min/t

# store value
ess_min_per_sec_mcwm = ess_min_per_sec
ess_min_per_sec_dagp = ess_min_per_sec

# compute relative ESS_min/sec  
ess_min_per_sec_mcwm/ess_min_per_sec_mcwm
ess_min_per_sec_dagp/ess_min_per_sec_mcwm
