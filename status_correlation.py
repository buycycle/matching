import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data import *
from src.driver import *
from src.matching import *
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Create the /plot directory if it doesn't exist
if not os.path.exists('plot'):
    os.makedirs('plot')
# Data preparation
X_train, X_test, y_train, y_test = create_data(
    query=main_query,
    query_dtype="",
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    target="status",
)
# Define the transformation pipeline
data_transform_pipeline = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)
# Fit and transform the data
X_train_transformed = data_transform_pipeline.fit_transform(X_train)
X_test_transformed = data_transform_pipeline.transform(X_test)
# Encode the target
y_train_transformed, y_test_transformed, labelencoder = encode_target(y_train, y_test)
# Initialize a dictionary to store feature importances for each category
feature_importances = {}
# Train a separate model for each category
categories = labelencoder.classes_
for category in categories:
    print(f"Training model for category: {category}")

    # Create binary target for the current category
    y_train_binary = (y_train_transformed == labelencoder.transform([category])[0]).astype(int)
    y_test_binary = (y_test_transformed == labelencoder.transform([category])[0]).astype(int)

    # Model training
    classifier = train(X_train_transformed, y_train_binary, model=XGBClassifier)

    # Store feature importances
    feature_importances[category] = classifier.feature_importances_
# Get the feature names after transformation
transformed_feature_names = data_transform_pipeline.get_feature_names_out()
# Create a mapping from transformed features to original features
feature_mapping = {}
for original_feature in numerical_features + categorical_features:
    if original_feature in numerical_features:
        feature_mapping[original_feature] = [f'num__{original_feature}']
    else:
        feature_mapping[original_feature] = [f'cat__{original_feature}_{category}' for category in X_train[original_feature].unique()]
# Aggregate feature importances for each category
aggregated_importances = {category: {} for category in categories}
for category in categories:
    importances = feature_importances[category]
    for original_feature, transformed_features in feature_mapping.items():
        aggregated_importances[category][original_feature] = sum(importances[transformed_feature_names.tolist().index(f)] for f in transformed_features if f in transformed_feature_names)
# Plot aggregated feature importances for each category
for category in categories:
    importances = aggregated_importances[category]
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top_features, top_importances = zip(*sorted_importances[:10])

    # Plot the feature importances of the top 10 features
    plt.figure(figsize=(10, 6))
    plt.title(f"Top 10 Aggregated Feature Importances for {category}")
    plt.bar(range(10), top_importances, color="lightblue", align="center")
    plt.xticks(range(10), top_features, rotation=45)
    plt.xlim([-1, 10])
    plt.tight_layout()
    plt.savefig(f'plot/top_10_aggregated_feature_importances_{category}.png')
    plt.close()

