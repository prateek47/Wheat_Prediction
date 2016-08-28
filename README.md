# Wheat_Prediction

### Problem Statement:
Given two years of winter wheat data, try and predict the wheat yield for several counties in the United States.

The data can be obtained from

--> 2013: https://aerialintel.blob.core.windows.net/recruiting/datasets/wheat-2013-supervised.csv

--> 2014: https://aerialintel.blob.core.windows.net/recruiting/datasets/wheat-2014-supervised.csv

#### The code for this analysis is written in python using ipython notebook.

## Approach:
1. Firstly, all the location based fields like County, state, latitude, longitude is removed and the data for both the years are joined to make a single training dataset.
2. All row with 'NA' values are removed ( The number of rows are significantly less, 615 rows in 360,042 rows)
3. I split the data in training and test set ( test split = 0.3)
4. I implemented different regression models for prediction:-

     a.) RandomForest Regression
     b.) Gradient Boosting Regression
     c.) Feed-Forward Neural Network regression
     d.) K Nearest Neighbour regression
5. Metric used for evaluation is "Mean squared error"

## Results
1. The best approach among all of these was RandomForest Regression with a MSE value of 32

#### The MSE values for all the models are listed in the table below:

|S.No|                   Models               |     MSE       |
|----| -------------------------------------: |:-------------:| 
| 1. | RandomForest Regression                |     32.       |
| 2. | Gradient Boosting Regression           |     62.33     |
| 3. | Feed-Forward Neural Network Regression |               |
| 4. | K Nearest Neighbour Regression         |     41.22     |

## Technical Choices
More detail in the ipython notebook itself...

## Key Findings and Insights
1. 

## Improvements


