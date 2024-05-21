import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import *
from src.driver import *
from src.matching import *
if not os.path.exists('plot'):
    os.makedirs('plot')
if not os.path.exists('statistics'):
    os.makedirs('statistics')
# Data preparation
X_train, X_test, y_train, y_test = create_data(
    query=main_query,
    query_dtype="",
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    target="status",
)
# Combine features and target for the training set
train_data = pd.concat([X_train, y_train], axis=1)
# Calculate statistics for features for each status
status_groups = train_data.groupby('status')
# Create a dictionary to store statistics for each status
status_feature_statistics = {}
# Iterate over each status group to calculate and store statistics
for status, group in status_groups:
    print(f"Calculating statistics for status: {status}")
    status_feature_statistics[status] = group.describe().transpose()
    # Optionally, save the statistics to a CSV file
    status_feature_statistics[status].to_csv(f'statistics/statistics_{status}.csv')
    # Plot histograms for numerical features for each status

# Combine features and target for the training set
train_data = pd.concat([X_train, y_train], axis=1)
# Select only numerical features for the pairplot
numerical_data = train_data.select_dtypes(include=['float64']).join(train_data['status'])
numerical_data - numerical_data + "bike_created_at_month"
# Create a pairplot with hue set to 'status'
pairplot_fig = sns.pairplot(numerical_data, hue='status', plot_kws={'alpha': 0.5})
# Save the pairplot to a file
pairplot_fig.savefig('plot/pairplot_with_hue_status.png')


# Optionally, print the statistics to the console
for status, stats in status_feature_statistics.items():
    print(f"\nStatistics for status: {status}")
    print(stats)

