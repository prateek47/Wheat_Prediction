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
The best approach among all of these was RandomForest Regression with a MSE value of 32.54

#### The MSE values for all the models are listed in the table below:

|S.No|                   Models               |     MSE       |
|----| -------------------------------------: |:-------------:| 
| 1. | RandomForest Regression                |     32.54     |
| 2. | Gradient Boosting Regression           |     46.57     |
| 3. | Feed-Forward Neural Network Regression |     129       |
| 4. | K Nearest Neighbour Regression         |     41.16     |

## Technical Choices
Are annotated in the ipython notebook itself.

## Key Findings and Insights
1. In each model we can observe that the importance of features like precipTypeIsOther, precipTypeIsSnow, precipTypeIsRain, precipAccumulation, precipProbability is pretty low in predicting the wheat yield, Therefore, these features can be ignored further analysis.
2. On the contrary features like windSpeed, windBearing, pressure, NDVI, DayInSeason play a vital role in predicting wheat yield.
3. Out of 150 counties only 19 have more than 1 value of Yield for a particular winter cycle. Therefore, it further supports the fact that location based fields are not a very good indicator and it is wise to ignore them.
4. The yield for a particular combination of longitude and latidue always remains the same in a particular winter cycle.


## Improvements
Using the US census data on agriculture, we can use various features on county level

County wise(Census Data) on agriculture (2007, adjusted) --> https://www.census.gov/support/USACdataDownloads.html

1. Total number of farms in each county( Assumption: The more the number of farms the more the yield){ The above data gives us 2007 census (adjusted)}
2. Average age of farm operators( Hypothesis: If average age should be in between 30-40, the yield should increase, as this age group have a unique combination of youthful energy and experience and would work harder and smarter, while higher average age, shows that, the youth is losing interest and that would have detrimental effect on farming.)
3. Average size of farms( Hypothesis: As the average size increases, the yield should increase)
4. The Feed-Forward Neural Network shows the least predictive power, but if more time is devoted it could be further optimized to perform better( XGBoost also did not perform quite well in the first iteration but after optimizing the hyper parameters associated with it, the performance improved significantly).

