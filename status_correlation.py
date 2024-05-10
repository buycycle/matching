from src.data import *

from src.driver import *
from src.matching import *
import seaborn as sns
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = create_data(
    query=main_query,
    query_dtype="",
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    target="status",
)

X_train_transformed, X_test_transformed, data_transform_pipeline = fit_transform(
    X_train, X_test, categorical_features, numerical_features
)

y_train_transformed, y_test_transformed, labelencoder = encode_target(y_train, y_test)


classifier = train(X_train_transformed, y_train_transformed, model=XGBClassifier)


probs = predict_probabilities(X_test_transformed, classifier)

test_probability(X_test_transformed, y_test_transformed, classifier, labelencoder)

feature_names = [f"Feature {i+1}" for i in range(X_train_transformed.shape[1])]

importances = classifier.feature_importances_
# Sort the feature importances in descending order and get the indices
indices = np.argsort(importances)[::-1]
# Select the indices of the top 10 most important features
top_indices = indices[:10]
real_feature_names = X_train_transformed.columns
# Rearrange the real feature names so they match the sorted feature importances
sorted_feature_names = [real_feature_names[i] for i in top_indices]
# Plot the feature importances of the top 10 features
plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importances")
plt.bar(range(10), importances[top_indices], color="lightblue", align="center")
plt.xticks(range(10), sorted_feature_names, rotation=45)
plt.xlim([-1, 10])
plt.tight_layout()
plt.show()

# Assuming X_train and y_train are your features and target variable respectively
# and that y_train is a categorical variable with the categories you want to analyze
# Convert the categorical target variable into a DataFrame with binary columns
y_train_binary = pd.get_dummies(y_train)
# Combine the features and binary target columns
combined_data = pd.concat([X_train, y_train_binary], axis=1)
# Calculate the correlation matrix
numeric_data = combined_data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
# Now, let's visualize the correlation between features and each binary target
for category in y_train_binary.columns:
    # Check if the category exists in the correlation matrix
    if category in correlation_matrix.columns:
        # Select the correlations between features and the binary target
        category_correlations = correlation_matrix[category].drop(y_train_binary.columns)

        # Create a bar plot for the correlations
        plt.figure(figsize=(8, 6))
        category_correlations.sort_values().plot(kind='barh')
        plt.title(f'Feature Correlations with {category}')
        plt.xlabel('Correlation')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Category '{category}' not found in correlation matrix.")
