# This .R code file consists of:
# Problem 2: Data Analysis of housingprice.csv Dataset
# Arthurs: Chenghan Sun, Nanhao Chen
# NOTE: please run this code in the directory (folder) of HW2

############ Problem 2 ############
# set working directory
setwd("./")  # set path to the current directory

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
data_X_train_a_global = model.matrix(linear_model_a)
data_X_train_sd_a_global = scale(data_X_train_a_global)
data_X_train_sd_a_global = cbind(1, data_X_train_sd_a_global[, -1])
selected_eta = select_eta(data_X_train_sd_a_global)
print(paste("The step size candidate = ", selected_eta))
# output
# [1] "The step size candidate =  5.31096932523487e-05"

# Here implement the gradient descent algorithm 
gradient_descent = function(data_X, res_y, eta=5.31e-05, grad_tol=10^(-8), maxiters=10^4, standardize=TRUE) {
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
    # print(sqrt(sum((as.vector(para_theta)-as.vector(para_theta_old))^2)))
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

# Part-d-1: Calculate R^2 of the linear model A for training set
# Apply gradient descent algorithm for training set based on model from part (a)
res_y_train_a = src_train$price
grad_results_train_a_gd = gradient_descent(data_X_train_a_global, res_y_train_a)  # input data X w/o standardization
gd_train_a = as.numeric(data_X_train_sd_a_global %*% as.matrix(grad_results_train_a_gd[1:5]))  # X*theta
gd_train_r_2_a = calc_r_square(gd_train_a, res_y_train_a)
print(paste("The R^2 of the linear model A for training set using GD = ", gd_train_r_2_a))
print(paste("The number of steps to finish the algorithm = ", grad_results_train_a_gd[6]))
# output:
# [1] "The R^2 of the linear model A for training set using GD =  0.510113853079458"
# [1] "The number of steps to finish the algorithm =  150"

# Part-d-2: Calculate R^2 of the linear model A for testing set
features = c('bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot')  # from Part (a)
data_X_test_a_global = as.matrix(src_test[, features])  # extract target features 
data_X_test_sd_a_global = scale(data_X_test_a_global)  # standardization
data_X_test_sd_a_global = cbind(1, data_X_test_sd_a_global)  # add column of 1's

res_y_test = src_test$price
gd_test_a = as.numeric(data_X_test_sd_a_global %*% as.matrix(grad_results_train_a_gd[1:5]))  # X*theta
gd_test_r_2_a = calc_r_square(gd_test_a, res_y_test)
print(paste("The R^2 of the linear model A for testing set using GD = ", gd_test_r_2_a))
# output:
# [1] "The R^2 of the linear model A for testing set using GD =  0.504933222347898"

# Part-d-3: estimate BC's house price based on GD
fancy_a_global = as.matrix(src_fancy[, features])  # extract target features 
# proceed with manually standardization of fancy house dataset
fancy_a_global = cbind(1, fancy_a_global) 
mean_global = colMeans(data_X_train_a_global)
sd_global = apply(data_X_train_a_global, 2, sd)
fancy_sd_a_global = sweep(fancy_a_global, 2, mean_global, '-')
fancy_sd_a_global = sweep(fancy_sd_a_global, 2, sd_global, '/')
fancy_sd_a_global[1] = 1
# get the BC's house price 
bchp_a = as.numeric(fancy_sd_a_global %*% as.matrix(grad_results_train_a_gd[1:5])) # Fit for Gates house
print(paste("The estimated BC's house price from linear model A for training set using GD = ", bchp_a))
# output
# [1] "The estimated BC's house price from linear model A for training set using GD =  15436769.5382226"

# Repeat the strategy above:
# Part-d-4: Calculate R^2 of the improved model C for training set
# find step size eta candidate
# First extract and standardize the design matrix X
data_X_train_c_global = model.matrix(linear_model_c)
data_X_train_sd_c_global = scale(data_X_train_c_global)
data_X_train_sd_c_global = cbind(1, data_X_train_sd_c_global[, -1])
selected_eta_c = select_eta(data_X_train_sd_c_global)
print(paste("The step size candidate = ", selected_eta_c))
# output
# [1] "The step size candidate =  4.11658644888999e-05"

# Apply gradient descent algorithm for training set based on model from part (c)
res_y_train_c = src_train$price
grad_results_train_c_gd = gradient_descent(data_X_train_c_global, res_y_train_c, eta=4.1e-05)  # input data X w/o standardization
gd_train_c = as.numeric(data_X_train_sd_c_global %*% as.matrix(grad_results_train_c_gd[1:6]))  # X*theta
gd_train_r_2_c = calc_r_square(gd_train_c, res_y_train_c)
print(paste("The R^2 of the improved model C for training set using GD = ", gd_train_r_2_c))
print(paste("The number of steps to finish the algorithm = ", grad_results_train_c_gd[7]))
# output:
# [1] "The R^2 of the linear model C for training set using GD =  0.51735329277383"
# [1] "The number of steps to finish the algorithm =  1448"

# Part-d-5: Calculate R^2 of the improved model C for testing set
data_X_test_c_global = cbind(as.matrix(src_test[, features]), src_test$bedrooms*src_test$bathrooms)  # extract target features 
data_X_test_sd_c_global = scale(data_X_test_c_global)  # standardization
data_X_test_sd_c_global = cbind(1, data_X_test_sd_c_global)  # add column of 1's

gd_test_c = as.numeric(data_X_test_sd_c_global %*% as.matrix(grad_results_train_c_gd[1:6]))  # X*theta
gd_test_r_2_c = calc_r_square(gd_test_c, res_y_test)
print(paste("The R^2 of the improved model C for testing set using GD = ", gd_test_r_2_c))
# output:
# [1] "The R^2 of the linear model C for testing set using GD =  0.51052055091153"

# Part-d-6: estimate BC's house price based on GD
fancy_c_global = cbind(as.matrix(src_fancy[, features]), src_fancy$bedrooms*src_fancy$bathrooms)  # extract target features 
# proceed with manually standardization of fancy house dataset
fancy_c_global = cbind(1, fancy_c_global) 
mean_global = colMeans(data_X_train_c_global)
sd_global = apply(data_X_train_c_global, 2, sd)
fancy_sd_c_global = sweep(fancy_c_global, 2, mean_global, '-')
fancy_sd_c_global = sweep(fancy_sd_c_global, 2, sd_global, '/')
fancy_sd_c_global[1] = 1
# get the BC's house price 
bchp_c = as.numeric(fancy_sd_c_global %*% as.matrix(grad_results_train_c_gd[1:6])) # Fit for Gates house
print(paste("The estimated BC's house price from Improved model C for training set using GD = ", bchp_c))
# output
# [1] "The estimated BC's house price from Improved model C for training set using GD =  18607312.8904335"

# Summary:
table_model_A = data.frame("Step size A" = selected_eta,
                           "Actual Iteration Numbers A" = grad_results_train_a_gd[6], 
                           "Tolerance A" = 10^(-8),
                           "Maximum iterations A" = 10^4,
                           "Training R^2 A" = gd_train_r_2_a,
                           "Testing R^2 A" = gd_test_r_2_a,
                           "BC's house price A" = bchp_a)

table_model_C = data.frame("Step size C" = selected_eta_c,
                           "Actual Iteration Numbers C" = grad_results_train_c_gd[7], 
                           "Tolerance C" = 10^(-8),
                           "Maximum iterations C" = 10^4,
                           "Training R^2 C" = gd_train_r_2_c,
                           "Testing R^2 C" = gd_test_r_2_c,
                           "BC's house price C" = bchp_c)


############ Part (e) ############
# Perform all the things above now using stochastic gradient descent (with one sample in each iteration).

# Here implement the stochastic gradient descent algorithm 
stochastic_gd = function(data_X, res_y, const_C, true_theta, sgd_tol=10^(-4), sgd_maxiters=10^5, standardize=TRUE) {
  # a version of mini-batch stochastic gradient descent
  # Params:
    # data_X: input data as a matrix
    # res_y: observation variables 
    # const_C: a tuning constant, used to get step size 
    # true_theta: theta from lm() build-in function
    # sgd_tol: tolerance of the error difference
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
  for (i in 1:sgd_maxiters) {
    iter = iter + 1  # update iter number at beginning
    print(paste("Iteration number: ", iter))
    eta_t = const_C / (iter + 1)  # decreasing step size based on C and iter number
    
    # select sample (based on given batch size)
    if (iter <= n) {
      sample = sample_pool[iter]  # iterate each randomized sample from sample pool
      # print(paste("The selected sample = ", sample))
    }
    else {
      sample = sample(sample_pool, 1)  # in case of run out of samples 
    }
    
    samp_data_X_sd = data_X_sd[sample, ]
    samp_res_y = res_y[sample]
    res_direct = as.numeric(samp_data_X_sd %*% para_theta) - samp_res_y  # compute response direction 
    gradient = as.numeric(res_direct %*% samp_data_X_sd)  # compute gradient
    para_theta = para_theta - eta_t*gradient  # update estimation of theta vector
    # print(para_theta)
    
    if (iter > sgd_maxiters) {
      print(paste("Algorithm unfinished by reaching the maximum iterations."))
      break
    }
    
    rel_err = norm((para_theta - true_theta), "2") / norm(as.matrix(true_theta), "2")
    print(rel_err)
    if (rel_err <= sgd_tol) {
      print(paste("Algorithm finished by reaching the tolerance."))
      break
    }
    # 
  }
  return(c(para_theta, iter))
}

# get the True theta from lm() build-in function
true_theta_a = summary(linear_model_a)$coefficients[,1]
true_theta_c = summary(linear_model_c)$coefficients[,1]

# Part-d-1: Calculate R^2 of the linear model A for training set
# Apply gradient descent algorithm for training set based on model from part (a)
res_y_train_a = src_train$price
sgd_results_train_a = stochastic_gd(data_X_train_a_global, res_y_train_a, 1, true_theta_a)  # input data X w/o standardization
sgd_train_a = as.numeric(data_X_train_sd_a_global %*% as.matrix(sgd_results_train_a[1:5]))  # X*theta
sgd_train_r_2_a = calc_r_square(sgd_train_a, res_y_train_a)
print(paste("The R^2 of the linear model A for training set using SGD = ", sgd_train_r_2_a))
print(paste("The number of steps to finish the algorithm = ", sgd_results_train_a[6]))
# output:
# [1] "The R^2 of the linear model A for training set using GD =  0.510113853079458"
# [1] "The number of steps to finish the algorithm =  150"







