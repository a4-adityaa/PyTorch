import sklearn
from sklearn.datasets import make_circles
import pandas as pd

# create sample

n_sample=1000

x,y = make_circles(n_sample,
                   noise=0.03,
                   random_state=42)

# print(f"First five data of x:\n {x[:5]}")
# print(f"First five data of y:\n {y[:5]}")

# make dataframe of circle data
circles= pd.DataFrame({"x1": x[:,0], # Dataframe is used to visulaise the data in tabular form (a pandas fctn)
                      "x2": x[:,1],
                      "label": y})

print(circles.head(10)) # cicle.head() is used to define the no of data we want

print(circles.label.value_counts()) # counts diffeerent lables 