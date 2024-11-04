# CO2 Emissions Prediction Model

This project includes a Linear Regression model to predict CO2 emissions based on various input features. The model is implemented in the `Linear Regression.ipynb` notebook file, with a detailed report in `Report.pdf`.

# Getting Started

To get started with this project, follow the steps below to set up your environment and run the application.

## Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- Jupyter Notebook

Although the Linear Regression algorithm is implemented manually using Batch Gradient Descent, the following libraries are necessary to parse the data, display graphs and test the assumptions of Linear Regression.

The following packages are required to run the jupyter file in its entirety.

- `matplotlib` - for plotting graphs

- `pandas` - for parsing, filtering and cleaning the dataset

- `sklearn` - for splitting the dataset into training, validation and testing subsets

- `seaborn` - for plotting categorical features

- `statsmodels` - for testing the assumptions of Linear Regression


The following segment is the list of requirements found at the first code cell in the notebook file.

```py
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
```

## Usage

The notebook includes instructions and code cells to guide through the process of training the model and making predictions. Follow the comments and markdown cells within the notebook for a step-by-step guide.
