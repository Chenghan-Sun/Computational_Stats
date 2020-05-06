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

# Part (a)
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


# Part (b)
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


# Part (c)
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


# Part (d)
# Using gradient descent algorithm on the sample-based least-squares objective function, to
# estimate the OLS regression parameter vector

# In order to apply gradient descent algorithm, make a function as step size finder
select_eta = function(data_X) {
  # Params
    # data_X: input data as a matrix
  # Return:
    # eta: selected step size 
  eigen_vals = eigen(t(data_X) %*% data_X)
  eta = 2 / (min(eigen_vals$values) + max(eigen_vals$values))
  return(eta)
}

# find step size eta candidate
features = c("bedrooms", "bathrooms", "sqft_living", "sqft_lot", "price")
data_X = as.matrix(src_train[features])
selected_eta = select_eta(data_X)
print(paste("The step size candidate = ", selected_eta))

# Here implement gradient descent algorithm 
gradient_descent = function(data_X, res_y, eta, grad_tol=10^(-5), maxiters=10^4, standardize=TRUE) {
  # Params:
    # data_X: input data as a matrix
    # res_y: 
    # eta: step size --> a constant 
    # grad_tol: tolerance of the error difference for norm of gradient 
    # maxiters: maximum number of iterations 
    # standardize: 
  # Return:
    # 
  p = ncol(data_X)
  para_theta = as.vector(rep(0, p))  # initialize theta vector with 0's
  # check standardization
  if (standardize = TRUE) {
    data_X_sd = scale(data_X)
    data_X_sd[, 1] = 1  # make a design matrix by adding a column of 1's
  }
  else {
    print("Warning: Input data matrix X was not standardized.")
  }
  
  iter = 0
  for (i in 1:maxiters) {
    para_theta_old = para_theta  # store old theta into another variable 
    res_direct = (data_X_sd %*% t(para_theta_old)) - res_y  # compute response direction 
    grad = (res_direct %*% data_X_sd)  # compute gradient
    para_theta = para_theta_old - gradient * eta  # update estimator
    
    if (iter > maxiters) {
      print("Algorithm unfinished by reaching the maximum iterations.")
    }
    else if (norm(grad) < grad_tol) {
      print("Algorithm finished by reaching the tolerance.")
      break
    }
    else {
      iter = iter + 1
    }
  }
  return(c(para_theta, iter))
}

# Apply gradient descent algorithm





