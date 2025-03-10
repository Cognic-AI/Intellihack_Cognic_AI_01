# Intellihack_Cognic_AI_1

# **Weather Forecasting with Machine Learning**  

## Introduction
Accurate weather predictions are crucial for farmers to optimize irrigation, planting, and harvesting. Traditional weather forecasts often lack precision at a hyper-local level, making it challenging for farmers to make informed decisions.

This project aims to develop a machine learning model that predicts whether it will rain (`rain_or_not`) based on historical weather data. The initial dataset consists of 310 days of daily weather observations, including temperature, humidity, wind speed, and rainfall status. However, the raw data contains missing values, errors, and inconsistencies, requiring thorough preprocessing before training the model.

The final goal is to build and optimize a predictive model that can forecast rainfall probabilities for the next 21 days, helping farmers make data-driven agricultural decisions.

## Project Workflow
### 1. Data Preprocessing
- Handle missing values and incorrect entries
- Convert date formats and normalize numerical features
- Encode categorical variables if necessary

### 2. Feature Engineering
- Create new features based on domain knowledge (e.g., rolling averages, seasonal effects)
- Transform skewed features for better model performance
- Scale numerical features to standardize data

### 3. Exploratory Data Analysis (EDA)
- Analyze class distribution of `rain_or_not`
- Identify feature correlations
- Visualize weather trends and relationships

### 4. Model Training & Evaluation


### 5. Optimization
- Feature selection based on correlation
- Hyperparameter tuning for better performance
- Address class imbalance if necessary

### 6. Prediction Output
- The final model should provide the probability of rain for the next 21 days

## **Project Structure**  

```
📂 INTELLIHACK_COGNIC_AI_01  
│-- 📂 Data-Set  
│   ├── 📂 processed_data        # Processed training & testing datasets  
│   ├── weather_train.csv       # Training dataset  
│   ├── weather_test.csv        # Testing dataset  
│   ├── weather_data.csv        # Initial weather data set
  
│  
│-- 📂 Plots                    # Visualizations & analysis outputs  
│  
│-- 📂 Scripts  
│   ├── EDA.ipynb               # Exploratory Data Analysis (EDA)  
│   ├── Pre_Process.ipynb       # Data preprocessing and feature engineering   
│   ├── 
│  
│-- Q-1 Weather Forecasting     # (Challenge Question)  
│-- README.md                   # Project documentation  
```



## **Requirements**  
- Python 3.8  
- ### Libraries Used:
- `pandas`: Data manipulation and analysis  
- `numpy`: Numerical operations  
- `matplotlib & seaborn`: Data visualization  
- `os`: File operations  
- `scipy.stats`: Statistical functions and hypothesis testing  
- `statsmodels.api`: Statistical modeling and time series analysis  
- `scikit-learn`:
  - `SimpleImputer, KNNImputer`: Handling missing values  
  - `StandardScaler`: Feature scaling  
  - `train_test_split`: Splitting dataset into train-test sets  
  - `mutual_info_classif`: Feature selection  
- `missingno`: Visualizing missing values  
- `datetime`: Handling date operations  
- `warnings`: Suppress unnecessary warnings  
 

