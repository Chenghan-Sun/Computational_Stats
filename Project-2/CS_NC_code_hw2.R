# This .R code file consists of:
# Problem 2: Data Analysis of housingprice.csv Dataset
# Arthurs: Chenghan Sun, Nanhao Chen
# NOTE: please run this code in the directory (folder) of HW2

############ Problem 2 ############
# set working directory
setwd("/Users/furinkazan/Box/STA_243/STA243_HW2/")  # set path to the current directory

# define paths and load the resource datasets
path_src_file_train = "train.data.csv"
path_src_file_test = "test.data.csv"
src_train = read.csv(path_src_file_train)
src_test = read.csv(path_src_file_test)

############ Part (a) ############
# Build a linear model on the training data using lm() by regessing the housing price on
# these variables: bedrooms, bathrooms, sqft living, and sqft lot.
linear_model_a = lm(price ~ bedrooms + bathrooms + sqft_living + sqft_lot, data=src_train)
r_2_train = summary(linear_model_a)$r.squared
print(paste("The R^2 of the model on training data = ", r_2_train))
# output:
# [1] "The R^2 of the model on training data =  0.510113853079458"

# Perfrom the above trained linear model on testing data
pred_test_a = predict.lm(linear_model_a, src_test)
ori_test_a = src_test$price
test_SSR_a = sum((ori_test_a - pred_test_a)^2)  # SSR of test dataset
mean_test_a = mean(ori_test_a)
test_SSTO_a = sum((ori_test_a - mean_test_a)^2)  # SSTO of test dataset
r_2_test = 1 - (test_SSR_a / test_SSTO_a)
print(paste("The R^2 of the model on test data = ", r_2_test))  # R^2 of test set
# output:
# [1] "The R^2 of the model on test data =  0.50499446140371"


############ Part (b) ############
# define paths and load the resource datasets
path_src_file_fancy = "fancyhouse.csv"
path_src_file_price = "housingprice.csv"
src_fancy = read.csv(path_src_file_fancy)
src_price = read.csv(path_src_file_price)

# apply the linear model from Part (a) to Bill Gates’ house
BG_house_price = predict.lm(linear_model_a, src_fancy)
print(paste("The estimated price of Bill Gates’ house on the Linear model = ", BG_house_price))
# output:
# [1] "The estimated price of Bill Gates’ house on the Linear model =  15436769.5382226"


############ Part (c) ############
# feature engineering
# Add another variable by multiplying the number of bedrooms by the number of bathrooms
linear_model_c = lm(price ~ bedrooms*bathrooms + sqft_living + sqft_lot, data=src_train)  # model with interaction term
r_2_train_improve = summary(linear_model_c)$r.squared
print(paste("The R^2 of the improved model on training data = ", r_2_train_improve))
# output:
# [1] "The R^2 of the improved model on training data =  0.517353292773831"

# Similar as Part (a), perfrom the above trained linear model on testing data
pred_test_c = predict.lm(linear_model_c, src_test)
test_SSR_c = sum((ori_test_a - pred_test_c)^2)  # improved SSR of test dataset
r_2_test_improve = 1 - (test_SSR_c / test_SSTO_a)
print(paste("The R^2 of the improved model on test data = ", r_2_test_improve))  # improved R^2 of test set
# output:
# [1] "The R^2 of the improved model on test data =  0.510535545859055"


############ Part (d) ############
# Using gradient descent algorithm on the sample-based least-squares objective function, to
# estimate the OLS regression parameter vector

# In order to apply gradient descent algorithm, make a function as step size finder by concept of bounded eigenvalues
select_eta = function(data_X) {
  # Params
    # data_X: input data as a matrix
  # Return:
    # eta: selected step size 
  eigen_lambs = eigen(t(data_X) %*% data_X)
  eta = 2 / (min(eigen_lambs$values) + max(eigen_lambs$values))
  return(eta)
}

# find step size eta candidate
# First extract and standardize the design matrix X
data_X_global = model.matrix(linear_model_a)
data_X_sd_global = scale(data_X_global)
data_X_sd_global = cbind(1, data_X_sd_global[, -1])
selected_eta = select_eta(data_X_sd_global)
print(paste("The step size candidate = ", selected_eta))
# output
# [1] "The step size candidate =  5.31096932523487e-05"

# Here implement the gradient descent algorithm 
gradient_descent = function(data_X, res_y, eta=5.31e-05, grad_tol=10^(-5), maxiters=10^4, standardize=TRUE) {
  # Implement based on page.3 Algorithm 1 of OPT.pdf
  # Params:
    # data_X: input data as a matrix
    # res_y: observation variables 
    # eta: step size --> a constant 
    # grad_tol: tolerance of the error difference for norm of gradient 
    # maxiters: maximum number of iterations 
    # standardize: standardize the dataset for better gradient descent algorithm performance 
  # Return:
    # a list of: 1. estimated theta vector; 2: number of iterations 
  if (standardize == TRUE) { # check standardization
    data_X_sd = scale(data_X)
    data_X_sd = cbind(1, data_X_sd[, -1])  # make a design matrix by adding a column of 1's
  }
  else {
    print(paste("Warning: Input data matrix X was not standardized."))
    break
  }
  
  p = ncol(data_X_sd)
  para_theta = as.matrix(rep(0, p))  # initialize theta vector with 0's
  
  iter = 0
  for (i in 1:maxiters) {
    para_theta_old = para_theta  # store old theta into another variable 
    res_direct = as.numeric(data_X_sd %*% para_theta_old) - res_y  # compute response direction 
    gradient = as.numeric(res_direct %*% data_X_sd)  # compute gradient
    # print(typeof(grad))
    # print(typeof(eta))
    para_theta = para_theta_old - eta*gradient  # update estimator
    # print(para_theta)
    if (iter > maxiters) {
      print(paste("Algorithm unfinished by reaching the maximum iterations."))
      break
    }
    print(sqrt(sum((as.vector(para_theta)-as.vector(para_theta_old))^2)))
    if (sqrt(sum((as.vector(para_theta)-as.vector(para_theta_old))^2)) <= grad_tol) {
      print(paste("Algorithm finished by reaching the tolerance."))
      break
    }
    iter = iter + 1
  }
  return(c(para_theta, iter))
}

# Here defined a function for computing the R^2 
calc_r_square = function(fit_y, res_y) {
  modelSSR = sum((res_y - fit_y)^2)  # # SSR of dataset
  mean_res_y = mean(res_y) 
  modelSSTO = sum((res_y - mean_res_y)^2)  # SSTO of dataset
  model_r_2 = 1 - (modelSSR / modelSSTO)  # compute r^2
  return(model_r_2)
}

# Apply gradient descent algorithm for training set based on model from part (a)
res_y_train = src_train$price
grad_results = gradient_descent(data_X_global, res_y_train)  # input data X w/o standardization

# Calculate R^2 of the original model for training set
gd_train_a = as.numeric(data_X_sd_global %*% as.matrix(grad_results[1:5]))  # X*theta
gd_train_r_2_a = calc_r_square(gd_train_a, res_y_train)
print(paste("The R^2 of the original model for training set using gradient descent algorithm = ", gd_train_r_2_a))





############ Part (e) ############
# Perform all the things above now using stochastic gradient descent (with one sample in each iteration).

# Here implement the stochastic gradient descent algorithm 
stochastic_gd = function(data_X, res_y, const_C, sgd_tol=10^(-4), maxiters=10^4, standardize=TRUE) {
  # a version of mini-batch stochastic gradient descent
  # Params:
    # data_X: input data as a matrix
    # res_y: observation variables 
    # eta: step size --> a constant 
    # grad_tol: tolerance of the error difference for norm of gradient 
    # maxiters: maximum number of iterations 
    # standardize: standardize the dataset for better gradient descent algorithm performance 
  # Return:
    # a list of: 1. estimated theta vector; 2: number of iterations 
  if (standardize == TRUE) { # check standardization as GD algorithm
    data_X_sd = scale(data_X)
    data_X_sd = cbind(1, data_X_sd[, -1])  # make a design matrix by adding a column of 1's
  }
  else {
    print(paste("Warning: Input data matrix X was not standardized."))
  }
  
  n = nrow(data_X_sd)  # number of rows 
  p = ncol(data_X_sd)  # number of columns
  para_theta = as.matrix(rep(0, p))  # initialize theta vector with 0's
  
  iter = 0  # initialize iter number 
  sample_pool = sample(n, replace=FALSE)  # make a list of sample integer from n w/o replacement 
  
  # generate decreasing step size eta_t based on conclusion of page.17 of OPT.pdf
  for (i in 1:maxiters) {
    iter = iter + 1  # update iter number at beginning
    eta_t = const_C / (iter + 1)  # decreasing step size based on C and iter number
    sample = sample_pool[iter]  # iterate each randomized sample from sample pool
    print(paste("The selected sample = ", sample))
    samp_data_X_sd = data_X_sd[sample, ]
    samp_res_y = res_y[sample]
    para_theta_old = para_theta  # store old theta into another variable 
    
    res_direct = as.numeric(samp_data_X_sd %*% para_theta_old) - samp_res_y  # compute response direction 
    # print(res_direct)
    gradient = as.numeric(res_direct %*% samp_data_X_sd)  # compute gradient
    para_theta = para_theta_old - eta_t*gradient  # update estimator
    # print(para_theta)
    
    if (iter > maxiters) {
      print(paste("Algorithm unfinished by reaching the maximum iterations."))
      break
    }
    # if ()
  }
  return(c(para_theta, iter))
}

# Apply gradient descent algorithm for training set based on model from part (a)
sgd_results = stochastic_gd(data_X_global, res_y_train, 5)

# Calculate R^2 of the original model for training set
sgd_train_a = as.numeric(data_X_sd_global %*% as.matrix(sgd_results[1:5]))  # X*theta
sgd_train_r_2_a = calc_r_square(sgd_train_a, res_y_train)
print(paste("The R^2 of the original model for training set using stochastic gradient descent algorithm = ", sgd_train_r_2_a))








