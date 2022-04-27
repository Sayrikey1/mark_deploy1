import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def marks_prediction(marks):
    df = pd.read_csv("california_housing_train.csv")

    x = df["median_income"]
    y = df['median_house_value']
    
    x = x.values
    y = y.values
    
    model = LinearRegression()
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    model.fit(x,y)
    
    X_test = np.array(marks)
    X_test = X_test.reshape((1,-1))
    
    ans = model.predict(X_test)
    ans1 = ans[0][0].round(4)
    #final = ' '.join([str(item) for item in ans])
    return ans1
