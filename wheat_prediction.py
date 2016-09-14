
# coding: utf-8

# ## Wheat Prediction Model

# Packages used

# In[1]:

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as err
from sklearn.ensemble import GradientBoostingRegressor
from sknn.mlp import Regressor, Layer
from sklearn.neighbors import KNeighborsRegressor

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# Reading and concatenating the 2013 and 2014 data to form 1 dataset

# In[8]:

data_2013 = pd.read_csv("wheat-2013-supervised.csv")
data_2014 = pd.read_csv("wheat-2014-supervised.csv")
total_data = pd.concat([data_2013, data_2014],axis=0)


# Checking the datashape

# In[9]:

print(data_2013.shape)
print(total_data.shape)


# #### Some Extrapolatory Analysis:

# ##### Summary Statistics

# In[10]:

data_2013.describe()


# In[153]:

data_2013.dtypes


# In[103]:

data_2014.describe()


# ### We are removing the location based data because the yield for a particular location is always the same

# #### ** Hypothesis 1 **: 
# A particular location (Longitude, Latitude) always have the same yield throughout a particular winter wheat yield, the Yield is different for different winters.
# 

# In[12]:

# First rounding the latitude and longitude
data_2013[['Latitude','Longitude']] = data_2013[['Latitude','Longitude']].round(6)
data_2014[['Latitude','Longitude']] = data_2014[['Latitude','Longitude']].round(6)


# In[95]:

# creating a dataframe for unique combinations of latitude and longitude
lon_lat13 = data_2013.drop_duplicates(['Latitude','Longitude']).ix[:,2:4]
lon_lat13 = lon_lat13.reset_index(inplace=False)
lon_lat13.head()


# In[102]:

yield_list = []
num_diff_yield = []
for i in range(0, len(lon_lat13)):
        yield_list.append(data_2013[(data_2013.Latitude == lon_lat13.Latitude[i]) &
                                    (data_2013.Longitude == lon_lat13.Longitude[i])].Yield.unique())
        num_diff_yield.append(len(data_2013[(data_2013.Latitude == lon_lat13.Latitude[i]) &
                                            (data_2013.Longitude == lon_lat13.Longitude[i])].Yield.unique()))

df1 = pd.DataFrame({'Location' : list(zip(lon_lat13.Latitude, lon_lat13.Longitude)), 
                   'Unique_Yields' : yield_list, 'Number_yield': num_diff_yield}, 
                  columns= ['Location','Unique_Yields', 'Number_yield'])

print("Number of coordinates with more than 1 yield value: %.4f " % len(df1[df1.Number_yield > 1]))
print("Total number of coordinates: %.4f" %len(df1))


# #### Hypothesis 2:
# Mostly particular county always produce the same yield throughout a particular winter year but the value of Yield is different for different years
# 

# In[43]:

# creating unique county list
countynames= data_2013.CountyName.unique()
# creating unique yields for each county
Yield_list = []
Num_diff_yields = []
for county in countynames:
    Yield_list.append(data_2013[data_2013.CountyName == county].Yield.unique())
    Num_diff_yields.append(len(data_2013[data_2013.CountyName == county].Yield.unique()))
df = pd.DataFrame({'County' : countynames, 'Unique_Yields' : Yield_list, 'Number_yield': Num_diff_yields}, 
                  columns= ['County','Unique_Yields', 'Number_yield'])

print("Number of counties with more than 1 yield value: %.4f " % len(df[df.Number_yield > 1]))
print("Total number of counties: %.4f" %len(df))
print(df.head(20))


# #### Removing location based fields in the data i.e. CountyName, State, Latitude and Longitude

# In[107]:

total_data1 = total_data.drop(['CountyName','State','Latitude','Longitude','Date'],axis=1)


# #### Removing the rows with 'NA' values

# In[108]:

total_data2 = total_data1.dropna(axis = 0)


# In[109]:

# number of rows before and after removing 'na' values
print(total_data1.shape[0])
print(total_data2.shape[0])


# Separating the Feature space and target value

# In[110]:

cols = list(total_data2.columns)
cols.remove('Yield')


# #### Creating a Train and Test split from the data

# In[170]:

train_split, test_split = train_test_split(total_data2, test_size = 0.3)


# ------------------------------------------------------------------------------

# ### Model Creation:

# #### Technical Choice #1:
# We start with a Tree based model rather than a linear model as when the relationship between a feature and the output is conditional upon the values of other features. A tree-based model would be able to capture such a conditionality, but linear models simply cannot.
# 
# Also, Tree-based models in principle can approximate functions with any "shape", whereas linear models can only produce functions with a linear "shape" with respect to a chosen set of features.

# #### Model 1: RandomForest Regressor

# In[112]:

# instantiating the model
rf_model = RandomForestRegressor(n_estimators=100)


# #### Technical Choice #2:
# 'n_estimators': The number of trees in the forest
#                 As the number of trees increases the complexity, time taken to run the model increases but the errors                 decreases. But with higher number of trees the model can overfit. Therefore the number of trees should                 be optimal i.e to maintain balance between reduced error and overfitting.

# In[113]:

# fitting the model
rf_model.fit(X=train_split.ix[:,cols],y=train_split.ix[:,'Yield'])

# important features
imp = list(zip(cols,rf_model.feature_importances_))
imp=sorted(imp,key=lambda x:x[1])


# In[114]:

# Printing the feature importance
print(imp)


# #### Make Predictions on the Test set

# In[115]:

pred1 = rf_model.predict(test_split.ix[:,cols])


# ### Model Evaluation(Metrics Used)

# #### Mean Square Error

# In[116]:

error1 = err(test_split.ix[:,"Yield"], pred1)
print( "MSE: %.4f" % error1)


# #### Technical Choice #3:
# ** Metric used: Mean Square error: **
# 
# Among various metrics like 'mean absolute error', median absolute error', R2 score and 'Mean squared error'. We choose
# 'mean Square Error'. It refers to the mean of the squared deviation of predicted value from the true valued. It is always positive and a value closer to zero is better.
# 
# Since the errors are square before they are averaged therefore, the MSE gives a relatively high weight to large errors. This means the RMSE is most useful when large errors are particularly undesirable.

# ------------------------------------------------------------------------

# #### Model 2: Gradient Boosting Regressor

# #### Technical Choice # 4:
# The next model, we opted for is Gradient Boosting, as it concentrates on reducing the error rather than fitting trees on random samples of the data and it more robust to overfitting than radom forest.

# In[117]:

# Parameters
params = {'n_estimators' :600, 'learning_rate' : 0.4, 'loss' : 'ls',
         'max_depth' : 8}
# instantiating the model
XGboost_model = GradientBoostingRegressor(**params)


# #### Technical Choice # 5:
# The parameters values used in the above model:
# 1. 'n_estimators': The number of boosting stages to perform on the data
# 2. 'learning_rate': learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
# 3. 'loss': 'ls' refers to least squares regression. It is a natural choice for regression due to its superior computational properties.
# 4. 'max_depth': The maximum depth limits the number of nodes in the tree. 

# In[122]:

# fitting the model
XGboost_model.fit(train_split.ix[:,cols], train_split.ix[:,"Yield"])


# #### Predictions using the Test set

# In[123]:

pred2 = XGboost_model.predict(test_split.ix[:,cols])


# #### Mean Squared error

# In[124]:

error2 = err(test_split.ix[:,'Yield'], pred2)
print( "MSE: %.4f" % error2)


# In[125]:

# important features 
imp2 = list(zip(cols, XGboost_model.feature_importances_))
imp2 = sorted(imp2,key=lambda x:x[1])
print(imp2)


# #### Technical Choice #6:
# Do get a better idea of the number of stages to boost the tree, we plot the deviance plot on the train and the testing data to get a better sense of the error at each boosting stage.
# Hypothesis: The plot(Deviance vs Boosting iterations) should follow a exponential curve, with deviance curving to a constant value with increasing number of iterations, so the point where the curve tends to be a straight line we can choose the iteration value as 'n_estimators'. 

# ### Plotting the test and train get deviance vs the boosting iterations

# In[126]:

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(XGboost_model.staged_predict(test_split.ix[:,cols])):
    test_score[i] = XGboost_model.loss_(test_split.ix[:,"Yield"], y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, XGboost_model.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')


# -----------------------------------------------------------------------

# #### Model 3: Feed Forward Neural Network Model

# In[269]:

# instantiating the model
nn_model = Regressor(
    layers=[
        Layer("Rectifier", units=100),
        Layer("Linear")],
    learning_rate=0.005,
    n_iter=20)


# #### Technical Choice #7:
# 1. The activation function for the hidden layer is 'Rectifier' with 100 neurons and 'linear' for the output layer
# 2. The learning rate: the learning rate of backpropogation method.
# 3. n_iter: The number of epoch's

# #### Technical Choice #8:
# 1. It is best to first normalize or standardize the data before inputting it in the neural network, as standardizing the inputs can make training faster and reduce the chances of getting stuck in local optima.
# 2. We need to first to remove the categorical variables (before normalizing) and then add them back after normalizing the rest of the numeric variables, as normalizing the categorical fields will make them lose their basic purpose.
# 3. Also, it is always better to convert the data and response variable into a numpy array before feeding into a neural network

# In[251]:

# Response variable
train_response = train_split.ix[:,"Yield"]
test_response = test_split.ix[:,"Yield"]


# In[252]:

# removing the categorical variable
train_split_cat = train_split.ix[:, 9:12]
test_split_cat = test_split.ix[:,9:12]

# dropping the categorical variable from rest of the data
train_split2 = train_split.drop(train_split.columns[[9,10,11,20]],axis=1)
test_split2 = test_split.drop(train_split.columns[[9,10,11,20]],axis=1)


# In[254]:

# Normalizing the input data
train_split_norm = (train_split2 - train_split2.mean())/(train_split2.max() - train_split2.min())
test_split_norm = (test_split2 - test_split2.mean())/(test_split2.max() - test_split2.min())


# In[255]:

# Normalizing the response variable
train_response_norm = (train_response - train_response.mean())/(train_response.max() - train_response.min())
test_response_norm = (test_response - test_response.mean())/(test_response.max() - test_response.min())


# In[256]:

# adding the categorical variable back to the normalized data for training
train_split_norm1 = pd.concat([train_split_norm, train_split_cat], axis=1)
test_split_norm1 = pd.concat([test_split_norm, test_split_cat], axis=1)


# In[257]:

# converting the dataframe into a numpy array before feeding in the neural net
Xtrain_array = train_split_norm1.ix[:,cols].as_matrix()
ytrain_array = train_response_norm.as_matrix()
Xtest_array = test_split_norm1.ix[:,cols].as_matrix()
ytest_array = test_response.as_matrix()


# In[270]:

# training the model
nn_model.fit(Xtrain_array, ytrain_array)


# #### Predicting on the test split

# In[271]:

pred3= nn_model.predict(Xtest_array)
# denormalizing the data
pred3_ = (pred3*(test_response.max() - test_response.min())) + test_response.mean()


# #### Technical Choice #9:
#  I have denormalized the response variable before calculating the MSE value.

# #### Mean Squared Error

# In[273]:

error3 = err(ytest_array, pred3_)
print( "MSE: %.4f" % error3)


# ----------------------------------------------------------------------------------------------------

# #### Model 4: K Nearest Neighbors

# In[118]:

# instantiating the model
knn_model = KNeighborsRegressor(n_neighbors= 3)


# #### Technical Choice #10:
#  I have used the nearest neighbors as 3(default = 5), This is chosen after performing iterations with various different values of the n_neighbors.

# In[119]:

# fitting the model
knn_model.fit(train_split.ix[:,cols], train_split.ix[:,"Yield"])


# #### Predicting on the Test split

# In[120]:

pred4 = knn_model.predict(test_split.ix[:,cols])


# #### Mean Squared Error

# In[121]:

error4 = err(test_split.ix[:,"Yield"], pred4)
print("MSE : %.4f" % error4)


# In[ ]:




# --------------------------------------------------------------

# ## Rough Work

# In[ ]:




# In[145]:

# Plotting Feature Importance
#feature_importance = XGboost_model.feature_importances_
# make importances relative to max importance
#feature_importance = 100.0 * (feature_importance / feature_importance.max())
#sorted_idx = np.argsort(feature_importance)
#pos = np.arange(sorted_idx.shape[0]) + .5
#plt.subplot(1, 2, 2)
#plt.barh(pos, feature_importance[sorted_idx], align='center')
#plt.yticks(pos, cols[sorted_idx])  # <-- use something else than 'cols'
#plt.xlabel('Relative Importance')
#plt.title('Variable Importance')
#plt.show()

# incomplete

