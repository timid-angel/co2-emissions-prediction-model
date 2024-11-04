# CO2 Emissions Prediction Model

This project includes a Linear Regression model to predict CO2 emissions based on various input features. The model is implemented in the [notebook file](https://github.com/timid-angel/co2-emissions-prediction-model/blob/master/Linear%20Regression.ipynb), with a detailed [report](https://github.com/timid-angel/co2-emissions-prediction-model/blob/master/Report.pdf).

The dataset is imported from Kaggle, a complete overview of the dataset can be found on their [website](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles/data).

# Getting Started

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

## Usage

The notebook includes instructions and code cells to guide through the process of training the model and making predictions. Follow the comments and markdown cells within the notebook for a step-by-step guide.
