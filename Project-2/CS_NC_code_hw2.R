# This .R code file consists of:
# Problem 2: Data Analysis of housingprice.csv Dataset
# Arthurs: Chenghan Sun, Nanhao Chen
# NOTE: please run this code in the directory (folder) of HW2

############ Problem 2 ############
# set working directory
setwd("/Users/furinkazan/Box/STA_243/STA243_HW2/")  # set path to the current directory

# define paths and load four resource datasets
path_src_file_train = "train.data.csv"
path_src_file_test = "test.data.csv"
path_src_file_fancy = "fancyhouse.csv"
path_src_file_price = "housingprice.csv"

src_train = read.csv(path_src_file_train)
src_test = read.csv(path_src_file_test)
src_fancy = read.csv(path_src_file_fancy)
src_price = read.csv(path_src_file_price)

# Part (a)
# Build a linear model on the training data using lm() by regessing the housing price on
# these variables: bedrooms, bathrooms, sqft living, and sqft lot.
linear_model_a = lm(price ~ bedrooms + bathrooms + sqft_living + sqft_lot, data=src_train)
r_2_train = summary(linear_model_a)$r.squared
print(paste("The R^2 of the model on training data = ", r_2_train))

# Perfrom the above trained linear model on testing data
pred_test_a = predict.lm(linear_model_a, src_test)
ori_test_a = src_test$price
test_SSR_a = sum((ori_test_a - pred_test_a)^2)  # SSR of test dataset
mean_test_a = mean(ori_test_a)
test_SSTO_a = sum((ori_test_a - mean_test_a)^2)  # SSTO of test dataset
r_2_test = 1 - (test_SSR_a / test_SSTO_a)
print(paste("The R^2 of the model on test data = ", r_2_test))  # R^2 of test set

# Part (b)





