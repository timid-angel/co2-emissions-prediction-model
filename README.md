# CO2 Emissions Prediction Model

This project includes a Linear Regression model to predict CO2 emissions based on various input features. The model is implemented in the [notebook file](https://github.com/timid-angel/co2-emissions-prediction-model/blob/master/Linear%20Regression.ipynb), with a detailed [report](https://github.com/timid-angel/co2-emissions-prediction-model/blob/master/Report.pdf).

The dataset is imported from Kaggle, a complete overview of the dataset can be found on their [website](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles/data).

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

- `statsmodels` `scipy` - for testing the assumptions of Linear Regression


These dependencies can be found in `requirements.txt`. To install all of them, simply run the command:
```bash
make install
```

# Usage

The notebook includes instructions and code cells to guide through the process of training the model and making predictions. Follow the comments and markdown cells within the notebook for a step-by-step guide. A brief description of the contents of the notebook are provided below.


This project implements a linear regression model to predict carbon dioxide emissions based on various vehicle features. It contains sections for data filtering/cleaning, model training using batch gradient descent, performance evaluation, and validation of linear regression assumptions.

## Linear Regression Implementation using Batch Gradient Descent
This class defines a linear regression model that accepts the learning rate and the epochs (number of iterations for the batch gradient descent) as hyperparameters. There are essentially two functions:
- `fit(X, Y)`: takes the features and the corresponding value to train the model
- `predict(X)`: takes a list of features and returns the model's predicted values

## Performance Metrics
This section defines functions to evaluate the performance of the linear regression model:
- `mean_absolute_error(real_values, predicted_values)`: calculates the mean absolute error between the real and predicted values.
- `root_mean_squared_error(real_values, predicted_values)`: calculates the root mean squared error between the real and predicted values.
- `coefficient_of_determination(real_values, predicted_values)`: calculates the R-squared value to determine the proportion of variance in the dependent variable that is predictable from the independent variables.

### Importing the data and dividing it into a training, testing and validation sets using a 60-20-20 split 
This section imports the dataset, filters out unnecessary features, and splits the data into training, validation, and testing subsets.

### Visualizing the individual relationships between the independent variables and the target
This section visualizes the relationships between the independent variables and the target variable (CO2 emissions) using scatter plots for numerical features and box plots for categorical features.

## Training the model and tuning the hyperparameters
This section trains the linear regression model using different learning rates and selects the best performing model based on the validation data subset. The best performing model is chosen based on the Root Mean Squared Error parameter.

## Testing the Model
This section tests the final version of the model using the testing data subset and evaluates its performance using various metrics.

## Showing that the conditions for Linear Regression are met
This section verifies that the conditions for linear regression are met:
- **Linearity**: Demonstrates the linear relationship between the dependent and independent variables.
- **Homoscedasticity**: Tests the consistency of the variance of residuals across all levels of the independent variables using the Breusch-Pagan test.
- **No Multicollinearity**: Calculates the Variance Inflation Factor (VIF) for each predictor to ensure that independent variables are not too highly correlated.
- **Normality of Errors**: Verifies that the residuals are normally distributed using a histogram and Q-Q plot.
- **No Autocorrelation of Errors**: Ensures that the residuals are independent of one another using the Durbin-Watson test.
