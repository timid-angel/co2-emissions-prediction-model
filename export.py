# %% [markdown]
# # Carbon Dioxide Emissions Prediction Model

# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm


# %% [markdown]
# ## Linear Regression Implementation using Batch Gradient Descent
# This class defines a linear regression model that acceps the learning rate and the epochs (number of iterations for the batch gradient descent) as hyperparameters. There are essentailly two functions:
# - `fit(X, Y)`: takes the features and the corresponding value to train the model
# - `predict(X)`: takes a list of features and returns the model's predicted values
# 

# %%
# Implementation of Linear Regression using Batch Gradient Descent
class LinearRegression:
    def __init__(self, lr=0.001, epochs=10000):# Adjust the epochs here
        self.learning_rate = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None


    def fit(self, X, Y):
        sample_count, feature_count = len(X), len(X[0])
        self.weights = [60.0] * feature_count
        self.bias = 0.0

        for _ in range(self.epochs):
            y_preds = [self.bias] * sample_count
            for i in range(sample_count):
                for fi in range(feature_count):
                    y_preds[i] += X[i][fi] * self.weights[fi]
            
            y_diff = [y_preds[i] - Y[i] for i in range(len(Y))]
            bias_diff = (2/sample_count) * sum(y_diff)

            weight_diff = [0] * feature_count
            for i in range(sample_count):
                for fi in range(feature_count):
                    weight_diff[fi] += (X[i][fi] * y_diff[i])
            
            for fi in range(feature_count):
                weight_diff[fi] *= 1/sample_count
            
            self.bias -= self.learning_rate * bias_diff
            for fi in range(len(self.weights)):
                self.weights[fi] -= self.learning_rate * weight_diff[fi]
            

    def predict(self, X):
        res = [self.bias] * len(X)
        for i in range(len(X)):
            for fi in range(len(X[0])):
                res[i] += self.weights[fi] * X[i][fi]
        
        return res

# %% [markdown]
# ## Performance Metrics

# %%
def mean_absolute_error(real_values, predicted_values):
    ms = 0
    for i in range(len(real_values)):
        ms += abs(real_values[i] - predicted_values[i])
    
    ms /= len(real_values)
    return ms

def root_mean_squared_error(real_values, predicted_values):
    ms = 0
    for i in range(len(real_values)):
        ms += pow(real_values[i] - predicted_values[i], 2)
    
    ms /= len(real_values)
    return pow(ms, 0.5)


def coefficient_of_determination(real_values, predicted_values):
    mean_real = sum(real_values) / len(real_values)
    
    total_variance = sum((val - mean_real) ** 2 for val in real_values)
    explained_variance = sum((real_values[i] - predicted_values[i]) ** 2 for i in range(len(real_values)))
    
    r_squared = 1 - (explained_variance / total_variance)
    return r_squared


# %% [markdown]
# #### Importing the data and dividing it into a training, testing and validation sets using a 60-20-20 split 

# %%
# Import the dataset and filter out unnecessary features
data = pd.read_csv("./data/CO2_Emissions.csv")

# %%
data.shape

# %%
data.info()

# %%
columns = [
    'Make',  
    'Vehicle Class', 
    'Engine Size(L)', 
    'Transmission', 
    'CO2 Emissions(g/km)'
]

data = data[columns]
# Renaming the columns that Have space in their names
data.columns = [col.replace(' ', '_') for col in data.columns]


data.head()

# %%
# Lets see how many unique categories we have for the categorical columns
categorical_columns = ['Make', 'Vehicle_Class', 'Transmission']
unique_counts = data[categorical_columns].nunique()

print(unique_counts)

# %% [markdown]
# - Assigning values starting from one for columns with categorical values

# %%
original_data = data.copy() # preserve names of categorical features
categorical_columns = ['Make', 'Vehicle_Class', 'Transmission']
for col in categorical_columns:
    data[col] = pd.factorize(data[col])[0] + 1

data.head()

# %%
# separate the last column and split the data
data_x = data.drop(columns=['CO2_Emissions(g/km)'])
data_y = data['CO2_Emissions(g/km)']

train_x, temp_x, train_y, temp_y = train_test_split(data_x, data_y, test_size=0.40, random_state=42)
validation_x, test_x, validation_y, test_y = train_test_split(temp_x, temp_y, test_size=0.50, random_state=42)


# %% [markdown]
# ### Visualizing the individual relationships between the independent variables and the target

# %%
# plot numerical features
numerical_features = ['Engine_Size(L)']

plt.figure(figsize=(15, 10))  
for i, feature in enumerate(numerical_features):
    plt.subplot(2, 2, i + 1)  
    plt.scatter(data_x[feature], data_y, s=10, c=data_y, cmap="viridis", alpha=0.9)
    plt.title(f'{feature} vs CO2 Emissions')
    plt.xlabel(feature)
    plt.ylabel("CO2 Emissions")

plt.tight_layout() 
plt.show()


# %%
# plot categorical features
categorical_features = ['Vehicle_Class', 'Make', 'Transmission']

for i, feature in enumerate(categorical_features):
    plt.figure(figsize=(17, 6))
    sns.boxplot(x=feature, y='CO2_Emissions(g/km)', data=original_data, hue=feature, palette="pastel", legend=False)
    plt.title('CO2 Emissions by '+feature)
    plt.xlabel(feature)
    plt.ylabel('CO2 Emissions (g/km)')
    plt.xticks(rotation=45)
    plt.show()

# %% [markdown]
# ## Training the model and tuning the hyperparameters
# 
# The learning rate will be tweaked to find a good middleground among all the hyperparameters. The validation data subset will be used to establish the best pair of hyperparameters for this particular model.

# %%

learning_rates = [0.00001] # change the learning rate here
models = [LinearRegression() for lr in learning_rates]

conv_train_x = train_x.values.tolist()
conv_train_y = list(train_y)

# Visualize features of the training data subset
train_x

# %%
# Visualize results of the training data subset
train_y

# %% [markdown]
# #### Train each model, run the test with the validation data subset and choose the best performing model

# %%
# Train each model
for model in models:
    model.fit(conv_train_x, conv_train_y)

# %%
conv_validation_x = validation_x.values.tolist()
conv_validation_y = list(validation_y)

rms_scores = [0] * len(models)
for i, model in enumerate(models):
    print(f" | Running validation tests for model #{i+1}")
    rms_scores[i] = root_mean_squared_error(conv_validation_y, model.predict(conv_validation_x))
    print(f" | Model #{i+1} scored an RMS value of: {round(rms_scores[i], 2)}")

best_model_index = 0
for i in range(len(models)):
    if rms_scores[i] > rms_scores[best_model_index]:
        best_model_index = i

best_model = models[best_model_index]
print(f"\n | Model {best_model_index + 1} with a learning rate of {learning_rates[best_model_index]} has the best RMS score.")

# %% [markdown]
# # Testing the Model
# 
# The testing data subset will be used along with the following performance metrics to evaluate the final version of the model:
# 

# %%
predictions = best_model.predict(test_x.values.tolist())

# Creating a DataFrame to compare actual vs predicted values
comparison_df = pd.DataFrame({
    'Actual CO2 Emissions (g/km)': test_y,
    'Predicted CO2 Emissions (g/km)': predictions
})

# Displaying the comparison table
print(comparison_df.head(10).to_string(index=False))

# Plotting the predicted values against the actual values
value_range = [test_y.min(), test_y.max()]

plt.scatter(list(test_y), predictions, c=predictions, cmap="cool", alpha=0.9)
plt.plot(value_range, value_range, lw="3", color="black")
plt.xlabel("CO2 Emissions (g/km)")
plt.ylabel("Predicted CO2 Emissions (g/km)")
plt.title("Actual vs Predicted CO2 Emissions")
plt.show()

# %%
residuals = [list(test_y)[i] - predictions[i] for i in range(len(test_y))]

# plot the residuals
plt.figure(figsize=(20, 10))
plt.scatter(predictions, residuals, c=residuals, cmap="hot", s=150)
plt.hlines(y=0, xmin=value_range[0], xmax=value_range[1], color="black")
plt.xlabel("Predicted CO2 Emissions (g/km)")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# %% [markdown]
# - Evaluating the model using performance metrics

# %%

conv_test_x = test_x.values.tolist()
conv_test_y = list(test_y)

test_predictions = best_model.predict(conv_test_x)

mae = mean_absolute_error(conv_test_y, test_predictions)
rmse = root_mean_squared_error(conv_test_y, test_predictions)
r_squared = coefficient_of_determination(conv_test_y, test_predictions)

print(f"Performance metrics for the best model (learning rate = {learning_rates[best_model_index]}):")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Coefficient of Determination (R^2): {r_squared}")

# %% [markdown]
# ## Showing that the conditions for Linear Regression are met
# The following are the conditions that determine whether linear regression is an appropriate model for a given problem:
# 
# - ***Linear relationship***: states that there should be a linear relationship between the dependent and independent variables. This shall be demonstrated for each of the features in our dataset.
# 
# - ***Homoscedasticity***: states that the variance of residuals should be consistent across all levels of the independent variables. This can also be demonstrated with the aforementioned graphs.
# 
# - ***No Multicollinearity***: states that independent variables must not be too highly correlated with each other.
# 
# - ***Normality of Errors***: states that the residuals (differences between observed and predicted values) should be normally distributed. This can be verified using a histogram or Q-Q plot of residuals.
# 
# - ***No Autocorrelation of Errors***: states that the residuals should not show patterns over time or across observations, ensuring they are independent of one another. This can be tested using the Durbin-Watson test.
# 

# %% [markdown]
# 1. **Linearity**

# %%
predictions = best_model.predict(test_x.values.tolist())
value_range = [test_y.min(), test_y.max()]

# compare the predicted values and the actual values
plt.scatter(list(test_y), predictions, c=predictions, cmap="cool", alpha=0.9)
plt.plot(value_range, value_range, lw="3", color="black")
plt.xlabel("CO2 Emissions (g/km)")
plt.ylabel("Predicted CO2 Emissions (g/km)")
plt.show()

# %% [markdown]
#  2. **Homoscedasticity (Constant Variance of Errors)**
# 
#  Here we use the "Breusch-Pagan Test". A high p-value (typically > 0.05) indicates that homoscedasticity holds. A low p-value suggests heteroscedasticity.

# %%


model = sm.OLS(train_y, sm.add_constant(train_x)).fit()
bp_test = het_breuschpagan(model.resid, model.model.exog)
p_value = bp_test[1]
print("Breusch-Pagan p-value:", p_value)



# %% [markdown]
#  3. **No Multicollinearity**
#  
#  Calculate the Variance Inflation Factor (VIF) for each predictor. VIF quantifies multicollinearity. 
# A VIF below 5 (some use 10) typically indicates acceptable multicollinearity. VIF values above 5 suggest high multicollinearity among predictors.
# 

# %%


vif_data = pd.DataFrame()
vif_data["feature"] = train_x.columns
vif_data["VIF"] = [variance_inflation_factor(train_x.values, i) for i in range(train_x.shape[1])]

print(vif_data)




# %% [markdown]
# - 4. **Normality of Errors**

# %%


# Histogram
plt.hist(residuals, bins=30, color="skyblue", edgecolor="black")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

# Q-Q Plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.show()


# %% [markdown]
# 5. **No Autocorrelation of Errors**
# 
# Here we use the Durbin Watson Test. The Durbin-Watson statistic ranges from 0 to 4. A value around 2 indicates no autocorrelation. Values closer to 0 suggest positive autocorrelation, and values closer to 4 suggest negative autocorrelation.

# %%


dw_stat = durbin_watson(residuals)
print("Durbin-Watson statistic:", dw_stat)



