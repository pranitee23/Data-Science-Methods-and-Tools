# -*- coding: utf-8 -*-
"""AsteroidPrediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mBepETc3WffdP-zw_2EHCnHk_ehq_Gyv
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('/content/drive/MyDrive/DSMT/nasa.csv')
df.head()

df.info()

df.describe()

df.isnull().sum()

duplicate_rows = df[df.duplicated()]

if duplicate_rows.shape[0] == 0:
    print("No duplicate rows found.")
else:
    print("Duplicate rows found:")
    print(duplicate_rows)

categorical_columns = df.select_dtypes(include=['object']).columns
numeric_columns = df.select_dtypes(exclude=['object']).columns

print("Categorical: ",categorical_columns)
print("Categorical Column Count:", len(categorical_columns))

print("\nNumeric",numeric_columns)
print("Numeric Column Count:", len(numeric_columns))

df.nunique()

# Set the style of seaborn
sns.set_style("whitegrid")

# Define pastel color palette
pastel_palette = sns.color_palette("pastel")[:2]

# Define columns and titles
columns = ['Hazardous', 'Equinox', 'Orbiting Body']
titles = ['Distribution of Hazardous Asteroids',
          'Distribution of Asteroids by Equinox',
          'Distribution of Asteroids by Orbiting Body']

# Plotting loop
fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # Creating a figure with 1 row and 3 columns
for i, column in enumerate(columns):
    sns.countplot(data=df, x=column, hue=column, palette=pastel_palette, ax=axes[i], legend=False)
    axes[i].set_title(titles[i])
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

palette = sns.color_palette("pastel", len(df['Orbit Uncertainity'].unique()))

# Create the count plot
plt.figure(figsize=(7, 6))
sns.countplot(data=df, x='Orbit Uncertainity', hue="Orbit Uncertainity", palette=palette)
plt.title('Count of Orbit Uncertainty')
plt.xlabel('Orbit Uncertainty')
plt.ylabel('Count')
plt.show()

# Delete duplicate columns.
# Use Name instead of Neo Reference ID.
# Use Mile unit instead of other units.
# Use Miles per hour only in velocity.
# Dropping Equinox and Orbiting Body since it has just one value and it will not help to predict the target variable

df.drop(["Neo Reference ID", "Est Dia in KM(min)", "Est Dia in KM(max)", "Est Dia in M(min)", "Est Dia in M(max)",
         "Est Dia in Feet(min)", "Est Dia in Feet(max)", "Relative Velocity km per sec",
         "Miss Dist.(Astronomical)", "Miss Dist.(lunar)", "Miss Dist.(kilometers)", "Equinox","Orbiting Body" ], axis = 1, inplace = True)

# df['Hazardous'] = df['Hazardous'].astype('object')

df.head()

df.columns

# Convert 'Close Approach Date' to datetime
df['Close Approach Date'] = pd.to_datetime(df['Close Approach Date'])

# Extracting year, month, and day from the date
df['Close Approach Year'] = df['Close Approach Date'].dt.year
df['Month'] = df['Close Approach Date'].dt.month
df['Day'] = df['Close Approach Date'].dt.day

# Drop the original 'Close Approach Date' column
df.drop(columns=['Close Approach Date', 'Month', 'Day'], inplace=True)

df[["Orbit Determination Year", "Orbit Determination Month", "Orbit Determination Day"]] = df["Orbit Determination Date"].str.split("-", expand = True)
df.drop(["Orbit Determination Date", "Orbit Determination Month", "Orbit Determination Day"], axis = 1, inplace = True)
df.head()

df.columns

# plt.figure(figsize = (10, 30))
# for i, col in enumerate(df.columns[1:-1], 1):
#     plt.subplot(8, 3, i)
#     sns.histplot(x = df[col], hue = df["Hazardous"], multiple = "dodge")
#     plt.title(f"Distribution of {col} Data")
#     plt.tight_layout()
#     plt.xticks(rotation = 90)
#     plt.plot()

X2 = df[['Name', 'Absolute Magnitude', 'Est Dia in Miles(min)',
       'Est Dia in Miles(max)', 'Epoch Date Close Approach', 'Miles per hour',
       'Miss Dist.(miles)', 'Orbit ID', 'Orbit Uncertainity', "Relative Velocity km per hr",
       'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant',
       'Epoch Osculation', 'Eccentricity', 'Semi Major Axis', 'Inclination',
       'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance',
       'Perihelion Arg', 'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly',
       'Mean Motion', 'Close Approach Year', 'Orbit Determination Year']]

df['Hazardous'] = df['Hazardous'].astype(int)

y2 = df['Hazardous']

logreg_model = LogisticRegression(max_iter=1000)
from sklearn.feature_selection import RFE
# Initialize RFE with the Logistic Regression model
rfe = RFE(logreg_model, n_features_to_select=1)

# Fit RFE
rfe.fit(X2, y2)

# Get ranking of features
feature_ranking = pd.Series(rfe.ranking_, index=X2.columns).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
colors = sns.color_palette("cubehelix_r", len(feature_ranking))
feature_ranking.plot(kind='barh', color=colors)
plt.title('Feature Ranking using Logistic Regression')
plt.ylabel('Ranking')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.show()

df.nunique()

df.shape

categorical_columns = df.select_dtypes(include=['object']).columns
#numeric_columns = df.select_dtypes(exclude=['object']).columns

print("Categorical: ",categorical_columns)
#print("Categorical Column Count:", len(categorical_columns))

# Dropping the columns which are of least importance based on the feature importance graph *******************
df.drop(["Orbit Determination Year", "Epoch Date Close Approach", "Name", "Close Approach Year"], axis = 1, inplace = True)

df.shape

df.columns

plt.figure(figsize=(15,13))
sns.heatmap(df.corr(),annot=True,linewidths=.1,fmt='.2f')
plt.show()

# Droppping semi-major axis since its collinear with multiple columns
df.drop(["Semi Major Axis"], axis = 1, inplace = True)

df.drop(["Relative Velocity km per hr"], axis = 1, inplace = True)

df.drop(["Est Dia in Miles(min)"], axis = 1, inplace = True)

df.columns

plt.figure(figsize=(15,13))
sns.heatmap(df.corr(),annot=True,linewidths=.1,fmt='.2f')
plt.show()

df['Hazardous'].value_counts()

numeric_columns = df.select_dtypes(exclude=['object']).columns
print("\nNumeric",numeric_columns)
print("Numeric Column Count:", len(numeric_columns))

numerical_columns = ['Absolute Magnitude', 'Est Dia in Miles(max)', 'Miles per hour',
                     'Miss Dist.(miles)', 'Orbit ID', 'Orbit Uncertainity',
                     'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant',
                     'Epoch Osculation', 'Eccentricity', 'Inclination', 'Asc Node Longitude',
                     'Orbital Period', 'Perihelion Distance', 'Perihelion Arg',
                     'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly', 'Mean Motion']

n_cols = 2
import numpy as np

# set the number of rows with the predefined number of columns
n_rows = int(np.ceil(len(numerical_columns)/n_cols))

# Create figure
fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 3*n_rows))
# Flatten the axes array for easier indexing
ax = ax.flatten()

for i, feature in enumerate(numerical_columns):
    sns.boxplot(data=df, x=feature, ax=ax[i])

# If the number of features is odd, remove the empty subplot
if len(numerical_columns) % 2 != 0:
    fig.delaxes(ax[-1])

plt.tight_layout()
plt.show(block=False)

df.drop(["Epoch Osculation", "Orbit ID", "Jupiter Tisserand Invariant"], axis = 1, inplace = True)

df.columns

plt.figure(figsize=(12, 10))
for i, col in enumerate(df.columns, 1):
    plt.subplot(6, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

"""## OUTLIERS"""

numerical_columns = ['Absolute Magnitude', 'Est Dia in Miles(max)', 'Miles per hour',
                     'Miss Dist.(miles)', 'Orbit Uncertainity',
                     'Minimum Orbit Intersection',
                    'Eccentricity', 'Inclination', 'Asc Node Longitude',
                     'Orbital Period', 'Perihelion Distance', 'Perihelion Arg',
                     'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly', 'Mean Motion']

n_cols = 2
import numpy as np

# set the number of rows with the predefined number of columns
n_rows = int(np.ceil(len(numerical_columns)/n_cols))

# Create figure
fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 3*n_rows))
# Flatten the axes array for easier indexing
ax = ax.flatten()

for i, feature in enumerate(numerical_columns):
    sns.boxplot(data=df, x=feature, ax=ax[i])

# If the number of features is odd, remove the empty subplot
if len(numerical_columns) % 2 != 0:
    fig.delaxes(ax[-1])

plt.tight_layout()
plt.show(block=False)

df_outliers_rem = df[(df['Orbital Period'] <= 2500) &
                   (df['Est Dia in Miles(max)'] <= 10000) &
                   (df['Orbital Period'] <= 2500) &
                   (df['Perihelion Time'] > 2.451) &
                   (df['Inclination'] < 70) &
                   (df['Aphelion Dist'] <= 6) &
                   (df['Mean Motion'] <= 2500)]

df_outliers_rem.reset_index(inplace=True, drop=True)
df_outliers_rem.info()



plt.figure(figsize = (10, 30))
for i, col in enumerate(df.columns[1:-1], 1):
    plt.subplot(8, 3, i)
    sns.histplot(x = df[col], hue = df["Hazardous"], multiple = "dodge")
    plt.title(f"Distribution of {col} Data")
    plt.tight_layout()
    plt.xticks(rotation = 90)
    plt.plot()

df_outliers_rem['Hazardous'].value_counts()

"""## Label Encoding"""

lbl_enc = LabelEncoder()
df_outliers_rem.loc[:, 'Hazardous Encoded'] = lbl_enc.fit_transform(df_outliers_rem['Hazardous'])
df_outliers_rem['Hazardous Encoded'].value_counts()

"""# Modelling"""

X = df_outliers_rem.drop(["Hazardous Encoded", "Hazardous"],axis=1)
y =df_outliers_rem['Hazardous Encoded']

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)

data_equals = X_train.equals(X_test)
if data_equals:
  print('Training and test data have same data')
else:
  print('Training and test data have different data')

"""## Logistic Regression"""

logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)

y_pred_logreg = logreg_model.predict(X_test)

# Accuracy
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg:.4f}")

# Confusion Matrix
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix_logreg, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

class_report_logreg = classification_report(y_test, y_pred_logreg)
print("Classification Report - Logistic Regression:\n", class_report_logreg)

"""## Decision Trees"""

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy:", dt_accuracy)

# dt_model = DecisionTreeClassifier(random_state=42)
# dt_scores = cross_val_score(dt_model, X_train, y_train, cv=5)
# print("Decision Tree Cross-Validation Mean Accuracy:", dt_scores.mean())

conf_matrix_dt = confusion_matrix(y_test, dt_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Decision Trees')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

class_report_dt = classification_report(y_test, dt_pred)
print("Classification Report - Logistic Regression:\n", class_report_dt)

"""## XGBoost"""

xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print("XGBoost Accuracy:", xgb_accuracy)

eval_set = [(X_train, y_train), (X_test, y_test)]

xgb_model = XGBClassifier(random_state=42,
                          gamma=1,
                          max_depth=5,
                          min_child_weight=1,
                          learning_rate=0.1,
                          subsample=0.8,
                          colsample_bytree=0.8)
xgb_model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10, verbose=False)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print("XGBoost Accuracy:", xgb_accuracy)

"""## Random Forest"""

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)

# Random Forest
from sklearn.model_selection import cross_val_score
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print("Random Forest Cross-Validation Mean Accuracy:", rf_scores.mean())

"""## SVC"""

# from sklearn.svm import SVC

# # Train SVM model
# svm_model = SVC(kernel='linear', random_state=42)
# svm_model.fit(X_train, y_train)

# # Predictions
# y_pred_svm = svm_model.predict(X_test)

# # Accuracy
# accuracy_svm = accuracy_score(y_test, y_pred_svm)
# print(f"SVM Accuracy: {accuracy_svm:.4f}")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

"""# KNN"""

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Predictions for KNeighborsClassifier
y_pred_knn = knn_model.predict(X_test)

# Accuracy for KNeighborsClassifier
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNeighborsClassifier Accuracy: {accuracy_knn:.4f}")

"""# Gaussian"""

gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)

# Predictions for GaussianNB
y_pred_gnb = gnb_model.predict(X_test)

# Accuracy for GaussianNB
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print(f"GaussianNB Accuracy: {accuracy_gnb:.4f}")

from joblib import dump

# Assuming your logistic regression model is named 'logreg_model'
# Save the model to a file
dump(logreg_model, 'logistic_regression_model.pkl')