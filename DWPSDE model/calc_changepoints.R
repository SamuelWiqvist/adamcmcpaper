# In this script we calcualte the change points in the file "cp_load_to_R.txt" and the file "load_data_to_matlab" is write. 
# The file "load_data_to_matlab" containes the location for the change points the mean value for each segment and the length of each segment

library("changepoint", lib.loc="~/R/win-library/3.3")
#library("R.matlab", lib.loc="~/R/win-library/3.2")
library(readr)

real_data = output <- read_csv("Results/data_usedmcwm_7_para_realdata.csv")
real_data = real_data$x1
  
# clac nbr cps for real data 
cp_data_real_data <- cpt.mean(real_data, method = "PELT", penalty = "Manual", pen.value = "0.1 * n")
#cp_data_export_real_data <- cbind(cpts(cp_data_real_data), coef(cp_data_real_data)$mean, seg.len(cp_data_real_data))
cp_data_export_real_data <- cbind(cpts(cp_data_real_data))
write.table(cp_data_export_real_data, file = "Results/cp_data_real_data.txt", row.names = FALSE, col.names = TRUE)


post_pred_data <- read_csv("Results/post_pred_data.csv")
post_pred_data = data.matrix(post_pred_data)

cp_data_export_post_pred <- rep(0., ncol(post_pred_data))

for (i in 1:ncol(post_pred_data)){
  cp_post_pred_data <- cpt.mean(post_pred_data[,i], method = "PELT", penalty = "Manual", pen.value = "0.1 * n")
  cp_data_export_post_pred[i] <- length(cpts(cp_post_pred_data))
}

write.table(cp_data_export_post_pred, file = "Results/cp_data_post_pred.txt", row.names = FALSE, col.names = TRUE)
