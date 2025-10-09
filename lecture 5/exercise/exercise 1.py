# -----------------------------
# Linear Regression on Boston Housing
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 1) Load data
url = "https://github.com/YBIFoundation/Dataset/raw/main/Boston.csv"
data = pd.read_csv(url)

# Display the first few rows and basic information about the dataset
print(data.head())
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Display basic statistics of the dataset
print(data.describe())

# 2) Exploratory Data Analysis (EDA)

# Visualize the distribution of the target variable 'medv'
plt.figure(figsize=(8, 6))
sns.histplot(data['medv'], bins=30, kde=True)
plt.title('Distribution of Median Value of Homes (medv)')
plt.xlabel('Median Value of Homes (medv)')
plt.ylabel('Frequency')
plt.show()
