# linear regression with GDP data
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

# loading datasources
data_happiness = pd.read_csv("data.csv", thousands=',') # didn't know about this
gdp_per_capita = pd.read_csv("gdp_data/csv", thousands=',')

# We left out merging the data
merged = merge_data(data_happiness, gdp_per_capita) # this function don't exists

X = np.c_[merged["GDP per capita"]]  # The predictor Variable
Y = np.c_[merged["Life satisfaction"]] # The labels or categories, or target variable

model = sklearn.linear_model.LinearRegression()


model.fit(X, Y)

X_example = [[12345]] # GDP per capita

model.predict(X_example) # predicts the hapinnes or life satisfaction
