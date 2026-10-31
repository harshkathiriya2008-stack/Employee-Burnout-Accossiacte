import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("\n" + "-" * 50)
print("       Employee Burnout Risk Assessment")
print("-" * 50)

print("\n" + "-" * 50)
print("       Data set")
print("-" * 50)

# Use maximum 10,000 records from train.csv
df = pd.read_csv("Dataset\\train.csv").head(10000)
#Swetviz Report 
# report = sv.analyze(df)
# report.show_html()

print("\nRecords used :", len(df))

print("\n" + "-" * 50)
print("       Understanding Dataset")
print("-" * 50)

print("\nShape :", df.shape)
print("\nColumns :", df.columns.tolist())

print("\n" + "-" * 50)
print("       Quantitative Data")
print("-" * 50)

print("\nquantitative data:int64 and float64")
quantitative = df.select_dtypes(include=['int64', 'float64'])
print("quantitative data:", quantitative)
print("quantitative column labels:\n", quantitative.columns)

print("\n" + "-" * 50)
print("       Qualitative Data")
print("-" * 50)

print("\n qualitative data:object and category")
qualitative = df.select_dtypes(include=['object', 'category'])
print("Qualitative data:", qualitative)
print("qualitative column labels:\n", qualitative.columns)

print("\n" + "-" * 50)
print("       EDA")
print("-" * 50)

print(df.head(5))
print(df.tail(5))
print(df.isnull().sum())
print(df.describe())
print(df.nunique())

print("\n" + "-" * 50)
print("       Correlation and covariation")
print("-" * 50)

print(df.cov(numeric_only=True))
print(df.corr(numeric_only=True))

print("\n" + "-" * 50)
print("       Value Counts")
print("-" * 50)

print(df['Gender'].value_counts())
print(df['Company Type'].value_counts())
print(df['WFH Setup Available'].value_counts())
print(df['Designation'].value_counts())
print(df['Resource Allocation'].value_counts())
print(df['Mental Fatigue Score'].value_counts())
print(df['Burn_Rate'].value_counts())

print("\n" + "-" * 50)
print("       Data preprocessing")
print("-" * 50)

# Employee ID and Date of Joining are not used as prediction features.
# Remove rows where Burn_Rate is missing
df = df.dropna(subset=['Burn_Rate'])

x = df.drop(columns=['Employee ID', 'Date of Joining', 'Burn_Rate'])
y = df['Burn_Rate']

print("\n" + "-" * 50)
print("       categorical and numerical columns")
print("-" * 50)

categorical_columns=[
    'Gender',
    'Company Type',
    'WFH Setup Available'
]

numeric_columns=[
    'Designation',
    'Resource Allocation',
    'Mental Fatigue Score'
]

print("Categorical columns:",categorical_columns)
print("Numerical columns:",numeric_columns)

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_columns),
    ('num', numeric_transformer, numeric_columns)
])

print("\n" + "-" * 50)
print("       Regression")
print("-" * 50)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("\n" + "-" * 50)
print("      R2 and RSS calculation")
print("-" * 50)

r2 = r2_score(y_test, y_pred)
print("r2 value:", r2)

rss = ((y_test - y_pred) ** 2).sum()
print("RSS:", rss)

print("\n" + "-" * 50)
print("       MAE and MSE calculation")
print("-" * 50)

mae = mean_absolute_error(y_test, y_pred)
print("mean mae:", mae)

mse = mean_squared_error(y_test, y_pred)
print("mean mse:", mse)

print("\n" + "-" * 50)
print("       RMSE calculation")
print("-" * 50)

rmse = np.sqrt(mse)
print("RMSE:", rmse)

print("\n" + "-" * 50)
print("       Employee Information")
print("-" * 50)

gender = input("Enter Gender: ")
company_type = input("Enter Company Type: ")
wfh = input("Enter WFH Setup Available (Yes/No): ")
designation = float(input("Enter Designation (1/2/3): "))
resource_allocation = float(input("Enter Resource Allocation: "))
mental_fatigue = float(input("Enter Mental Fatigue Score: "))

employee = pd.DataFrame(
    [[gender, company_type, wfh, designation,
      resource_allocation, mental_fatigue]],
    columns=x.columns
)

prediction = model.predict(employee)[0]

# Keep the prediction within the Burn_Rate scale observed in the dataset.
prediction = max(float(y.min()), min(float(y.max()), float(prediction)))

print("\n" + "-" * 50)
print("             PREDICTION RESULT")
print("-" * 50)

print("Gender                  :", gender)
print("Company Type            :", company_type)
print("WFH Setup Available     :", wfh)
print("Designation             :", designation)
print("Resource Allocation     :", resource_allocation)
print("Mental Fatigue Score    :", mental_fatigue)

print("\nPredicted Burn_Rate     :", round(prediction, 4))

# Risk category for easy interpretation.
if prediction >= 0.70:
    risk = "HIGH"
elif prediction >= 0.40:
    risk = "MEDIUM"
else:
    risk = "LOW"

print("Burnout Risk Level      :", risk)

print("\n" + "=" * 50)
print("          ASSESSMENT COMPLETED")
print("=" * 50)
