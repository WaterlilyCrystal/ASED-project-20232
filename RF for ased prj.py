import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def plot_importance(model, X):
    feature_names = X.columns

    # Set a larger figure size
    plt.figure(figsize=(12, 8))

    # Plot MDI feature importance
    if hasattr(model, 'feature_importances_'):
        plt.barh(feature_names, model.feature_importances_)
        plt.xlabel('Feature Importance (MDI)')
        plt.ylabel('Feature')
        plt.title('Feature Importance Scores (MDI)')

        plt.tight_layout()
        plt.show()


# Load data (replace "Train_data.csv" with your actual file path)
data = pd.read_csv("Train_data_new.csv")

# Identify categorical columns
categorical_cols = ['service', 'flag']

# One-hot encode categorical variables
encoder = OneHotEncoder(drop='first')
encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]).toarray())
feature_names = encoder.get_feature_names_out(categorical_cols)
encoded_cols.columns = feature_names

# Concatenate encoded columns with original dataset
data_encoded = pd.concat([data.drop(columns=categorical_cols), encoded_cols], axis=1)

# Split data into features and target variable
#X = data_encoded[["duration", "protocol_type",  "src_bytes", "dst_bytes", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "srv_count", "count",  "logged_in", "service_ftp_data", "service_ecr_i", "service_eco_i", "service_domain_u"]]
X = data_encoded.drop(columns=['class'])
y = data_encoded['class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your model (example using RandomForestClassifier)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot feature importances
plot_importance(model, X)
print("Classification Report (Test):")
print(classification_report(y_test, y_pred))
