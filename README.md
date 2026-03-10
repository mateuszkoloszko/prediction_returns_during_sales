# Insurance Pricing Pipeline (French MTPL Data)

## Overview
The goal of this project is to build a model that predicts the number of product returns per week during a sales campaign, based on historical data including discounts, sales, inventory levels, and product attributes.

## Technologies
The project was developed using the following technologies and libraries:
* **Python** 
* **Pandas** 
* **NumPy**
* **Scikit-learn**
* **Seaborn & Matplotlib**
* **SQLAlchemy**

## Steps

1. **Data Loading**
   - Importing necessary analytical modules and libraries.
   - Loading the raw dataset for analysis.

2. **Exploratory Data Analysis**
   - Basic descriptive statistics like data structure or variable types.
   - Find missing data and data description.
   
3. **Data Cleaning and Preprocessing**
   - Handling missing values and potential data entry errors.
   - Encoding categorical variables.
   - Feature scaling and normalization.

4. **Correlation Analysis**
   - Calculating correlations between numerical variables.
   - Visualizing relationships between features using heatmaps.

5. **Data Splitting**
   - Defining the target variable and input features.
   - Splitting the dataset into training, validation, and testing.
   
6. **Model Training & Evaluation**
   - Initializing and training regression models (Linear Regression, Ridge, Lasso).
   - Making predictions on the testing sets.
   - Evaluating model performance using Mean Squared Error (MSE) and R-squared ($R^2$) score.