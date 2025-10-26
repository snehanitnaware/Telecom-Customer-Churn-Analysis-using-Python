#importing all libararies

import numpy as np   #for numerical operations
import pandas as pd  #for data handling and manipulation
import matplotlib.pyplot as plt  #for plotting charts or visuals
import seaborn as sns  #for advanced stats visualization
from sklearn.model_selection import train_test_split  #to split data for training/testing
from sklearn.linear_model import LogisticRegression  #ML algorithms for classification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  #for model evaluation

#Loading Dataset-Read dataset into a Pandas DataFrame
df = pd.read_csv('data/Churn.csv')
# Display first 5 rows to understand data structure
df.head()
#Read dataset into a Pandas DataFrame
df = pd.read_csv('data/Churn.csv')
# Display first 5 rows to understand data structure
df.head()
# Inspecting Data - Part 1 - Check column names, data types, and non-null counts
df.info()
# Inspecting Data - Part 2 - Check basic stats for numerical columns(such as mean, median, min, max stc))
# df.describe(include = 'all')  #both categorial and numrical variable
df.describe()   # Shows only numerical variable
#Inspecting Data - Part 3 - Check any missing values present in each column
df.isnull().sum()
# Inspecting Data- Part 4 - To check the type of each dataset
df.dtypes
###print(df.columns)

# Data Cleaning
# Convert 'Total Charges' column to numeric type (just to ensure consistency)
# Some rows might contain blank strings - errors='coerce' turns them into NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors ='coerce')
# Reconfirm that conversion didn’t introduce new NaNs
###print("Missing Values after conversion:\n", df['TotalCharges'].isnull().sum())
# Fill missing 'TotalCharges' with median value of the column
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# drop the 'CustomerID' column - it's unique for each person and not useful for prediction
# df.drop('customerID',axis=1, inplace=True)  #already droppend in first run
df.drop(columns=['customerID', 'CustomerID'], inplace=True, errors='ignore')   #to confirm customerID is dropped

# Data Review
# Check how many customers churned vs stayed
df['Churn'].value_counts()
# Plot churn distribution (to see imbalance in target audience)
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title("Churn Distribution")
plt.show()

# Exploratory Data Analysis - Part 1 - Compare customer tenure(how long they've been with company) with churn
sns.boxplot(x='Churn', y='tenure', data=df, palette='pastel')
plt.title("Tenure vs Churn")
plt.show()
#Exploratory Data Analysis - Part 2 - Check churn rate by contract type
sns.countplot(x='Contract',hue='Churn', data=df, palette='cool')
plt.title('Contract type vs Churn')
plt.show()
# Exploratory Data Analysis - Part 3 - Compare monthly charges for churned vs retained customers
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, hue='Churn', palette = 'muted')
plt.title('Monthly Charges vs Churn')
plt.show()
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()
# 8.Convert Categorical Columns

# Use one-hot encoding to convert categorial columns into numerical
# TODO: fix comment: One-hot encoding indicates presenc eor absence oof that category
# drop_first= True avoids dummy variable trao
df_encoded=pd.get_dummies(df,drop_first=True)

# 9. Split Data for Training & Testing
# Seperate independent variables(features) and target variable(label)
x= df_encoded.drop('Churn_Yes',axis=1)  #All columns except target
y = df_encoded['Churn_Yes'] #Target column(1=churned, 0=stayed)

# Split data into 80% training and 20% testing subsets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

# Print dataset sizes for verification
print("Training set size:",x_train.shape)
print("Test set size:",x_test.shape)

#10. Train Logistics regression Model
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
# Initialize Logistics regression model
model = LogisticRegression(max_iter=2000)
# Train the model using training data
model.fit(x_train_scaled,y_train)

#Use the model to predict churn on test data
# y_pred = model.predict(x_test)

# 11. Evaluate Model Performance
# Calculate accuracy score - proportion of correct predictions
print("Model Accuracy:",accuracy_score(y_test,y_pred))
# Display confusion matrix - shows correct & incorrect classifications
print("Confusion matrix:\n", confusion_matrix(y_test,y_pred))
# Generate a detailed report: precision, recall, f1-score
print("Classification Report:\n", classification_report(y_test,y_pred))
# 12. Feature Importance(Optional Insight)
# Check which feature influence churn most (positive = increase churn possibility)
importance = pd.DataFrame({'Feature': x.columns,'coefficient': model.coef_[0]}).sort_values(by='coefficient', ascending=False)
# Display top 10 features
importance.head(10)

print("****Done***")
