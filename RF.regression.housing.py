# %% Section 1: classes loading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# %% Section 2: loading of sklearn's subcomponents and dataset
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

# %% Section 3: loading the data and inspect it
california_housing = fetch_california_housing()

df = pd.DataFrame(
    california_housing.data, columns=california_housing.feature_names
    )
df_head = df.head(5)
print("The first 5 lines:")
print(df_head)

'''
The data is two arrays: the target array (features*samples)
and the targeta array
'''

df_feature = pd.DataFrame(
    california_housing.feature_names, columns=["feature"]
    )
df_target = pd.DataFrame(california_housing.target, columns=["target"])
print("features", df_feature)
print("targets", df_target)

# I want to see if there are some NaN values and, in case, how to impute them
null_counts = df.isnull().sum()
if null_counts.any():
    df_filled_mean = df.fillna(df.mean())
    print("\n Imputed data:", df_filled_mean)
else:
    print("\n No NaN value to impute")

# %% Section 4: splitting the data in training and test
X = pd.DataFrame(california_housing.data)
Y = pd.DataFrame(california_housing.target)

# splitting
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

y_train = y_train.values.ravel()  # shrinking the array in 1D
y_test = y_test.values.ravel()  # shrinking the array in 1D

# testing if it's all'okey
len_train = len(x_train)
len_test = len(x_test)

expected_len_train = 4 * len_test
if np.isclose(len_train, expected_len_train, atol=1):
    print("It's all'ok, proceeding...")
else:
    print("There's an error, aborting...")
    sys.exit()

# %% Section 5: fitting the RF regression
regressor = RandomForestRegressor(n_estimators=100)   # 100 trees
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("The result is (mse):", mse)
print("The result is (R^2):", r2)
# %% Section 6: PVI (permutation importance), for the features
pvi = permutation_importance(
    regressor, x_test, y_test,
    n_repeats=20, n_jobs=10
    )
pvi_mean = pvi.importances_mean
print("Result of PVI:", pvi_mean)
feature_list = df_feature.values.flatten()
FeaturePVI_df = pd.DataFrame(
    {"Features": feature_list, "Importance": pvi_mean}
    )
print(FeaturePVI_df)

# %% Section 7: creating the df with the most PVI feature

# Finding the most importante feature for the RF
important_features_df = pd.DataFrame(FeaturePVI_df.head(2).iloc[:, 0])
print(important_features_df)  # This is the df with the most important features

important_features_trans = important_features_df.T  # df with transposed matrix

# This df is x_train values with feature as first head
FeatValues_df = pd.DataFrame(
    x_train.values, columns=california_housing.feature_names
    )

# I can filter important_features with TrainFeature_df
columns_too_keep = important_features_df.values.tolist()  # This is a touplet
columns_too_keep_list = [  # This is a plain list
    col[0] for col in columns_too_keep
    ]

filtered_df = FeatValues_df[  # This is a df with the most important features
    columns_too_keep_list
    ]
print(filtered_df)

# %% Plotting the surface plot with 100 trees
#  Defining the axes
x_min, x_max = filtered_df.iloc[:, 0].min()-1, filtered_df.iloc[:, 0].max()+1
y_min, y_max = filtered_df.iloc[:, 1].min()-1, filtered_df.iloc[:, 1].max()+1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]  # Here the coordinate of axes
grid_points_df = pd.DataFrame(grid_points, columns=filtered_df.columns)

# Fitting e predicting the points in the grid & R^2
model_100 = RandomForestRegressor(n_estimators=100)
model_100_fit_plot = model_100.fit(filtered_df, y_train)  # Fit of the features
regressor_prediction_plot = model_100.predict(grid_points_df)
zz = regressor_prediction_plot.reshape(xx.shape)  # Prediction of the features
R2_plot = r2_score(y_train, model_100.predict(filtered_df))
print('R^2 del plot', R2_plot)

# Dividing the DFs (30%) in order be plotted
y_train_df = pd.DataFrame(y_train)
scatter_plot_30 = y_train_df.sample(frac=0.05, random_state=42)
filtered_df_30 = filtered_df.sample(frac=0.05, random_state=42)

# Plotting the figure (hopefully)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, zz, cmap='viridis', alpha=0.9)
plt.scatter(
            filtered_df_30.iloc[:, 0], filtered_df_30.iloc[:, 1],
            c=scatter_plot_30, edgecolor='k', s=50, cmap='viridis', alpha=0.6
            )
plt.colorbar(label='Predicted SalePrice')

# Annotations and labels
plt.xlabel(filtered_df.columns[0])
plt.ylabel(filtered_df.columns[1])
plt.title('Decision Tree Predictions')
plt.text(x_max - 0.3, y_min + 1,
         f'$R^2$={R2_plot:.2f}', fontsize=15, ha='right'
         )
plt.show()

'''
Why the R^2 of this last model is better that the first model? It should be
the opposite: the leat feature you put, the worse prediction you get (or is it
a overfitting phenomenon?)
'''
