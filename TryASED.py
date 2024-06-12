# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# #from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.inspection import permutation_importance
# import matplotlib.pyplot as plt

# def init():    
#     df = pd.read_csv("Train_data.csv")

#     X = df
#     y = df["class"] #normal=0, anomaly=1

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

#     return X_train, X_test, y_train, y_test, X, y


# def plot(model, X, y, permutation_importance_result):
#     feature_names = X.columns

#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

#     # Plot MDI feature importance
#     axes[0].barh(feature_names, model.feature_importances_)
#     axes[0].set_xlabel('Feature Importance (MDI)')
#     axes[0].set_ylabel('Feature')
#     axes[0].set_title('Feature Importance Scores (MDI)')

#     # Plot permutation importance 
#     importances_mean = permutation_importance_result.importances_mean
#     importances = pd.DataFrame(importances_mean, index=feature_names, columns=['Importance'])
#     ax = importances.plot.barh(ax=axes[1], width = 0.8)
#     ax.set_title("Permutation Importances")
#     ax.set_xlabel("Mean importance score")

#     plt.tight_layout()
#     plt.show()

# def train_classification(model, X_train, y_train, hyperparameters):
#     clf = GridSearchCV(model, hyperparameters, cv=10, scoring='accuracy')
#     clf.fit(X_train, y_train)

#     print("Best parameters found for classification:", clf.best_params_)
#     return clf.best_estimator_

# # def evaluate_classification(model, X_train, y_train, X_test, y_test):
# #     print("\n----- Validation phase (Classification): -----")
# #     y_pred_train = model.predict(X_train)
# #     accuracy_train = accuracy_score(y_train, y_pred_train)
# #     print("Training Accuracy:", accuracy_train)
# #     print("Classification Report (Training):")
# #     print(classification_report(y_train, y_pred_train))

# #     print("\n----- Testing phase (Classification): -----")
# #     y_pred_test = model.predict(X_test)
# #     accuracy_test = accuracy_score(y_test, y_pred_test)
# #     print("Testing Accuracy:", accuracy_test)
# #     print("Classification Report (Testing):")
# #     print(classification_report(y_test, y_pred_test))

# #     return model.get_params() 

# def calculate_permutation_importance(model, X_test, y_test):
#     result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
#     return result

# import lightgbm as lgb
# # Hyperparameters for LightGBM 
# lgb_hyperparameters = {
#     'boosting_type': 'gbdt',
#     'objective': 'classification',
#     'metric': 'rmse',
#     'num_leaves': 450,
#     'learning_rate': 0.05,
#     'feature_fraction': 1,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': 0
# }
# def evaluate_classification(model, X_test, y_test):
#     print("\n----- Testing phase (Classification): -----")
#     y_pred_test = model.predict(X_test)
#     accuracy_test = accuracy_score(y_test, y_pred_test)
#     print("Testing Accuracy:", accuracy_test)
#     print("Classification Report (Testing):")
#     print(classification_report(y_test, y_pred_test))

#     return model.get_params()

# # After evaluating the model
# output_file = "ASED.txt"
# # Initialize data
# X_train, X_test, y_train, y_test, X, y = init()
# lgb_model = train_classification(lgb.LGBMClassifier(), X_train, y_train, lgb_hyperparameters)
# evaluate_classification(lgb_model, X_test, y_test)

# # After evaluating the model
# output_file = "ASED.txt"
# # Initialize data
# X_train, X_test, y_train, y_test, X, y = init()
# lgb_model=train_classification(lgb, X_train, y_train, lgb_hyperparameters)
# evaluate_classification(lgb_model, X_train, y_train, X_test, y_test, output_file,lgb_hyperparameters)

#Not ok yet: 
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.inspection import permutation_importance
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression

# def init():    
#     df = pd.read_csv("Train_data.csv")
#     X = df.drop(columns=["class",'service', 'flag'])  
#     y = df["class"]   #(normal=0, anomaly=1)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

#     return X_train, X_test, y_train, y_test, X, y


# def plot_importance(model, X, permutation_importance_result):
#     feature_names = X.columns

#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

#     # Plot MDI feature importance
#     if hasattr(model, 'feature_importances_'):
#         axes[0].barh(feature_names, model.feature_importances_)
#         axes[0].set_xlabel('Feature Importance (MDI)')
#         axes[0].set_ylabel('Feature')
#         axes[0].set_title('Feature Importance Scores (MDI)')

#     # Plot permutation importance 
#     importances_mean = permutation_importance_result.importances_mean
#     importances = pd.DataFrame(importances_mean, index=feature_names, columns=['Importance'])
#     ax = importances.plot.barh(ax=axes[1], width=0.8)
#     ax.set_title("Permutation Importances")
#     ax.set_xlabel("Mean importance score")

#     plt.tight_layout()
#     plt.show()

# def train_classification(model, X_train, y_train, hyperparameters):
#     clf = GridSearchCV(model, hyperparameters, cv=10, scoring='accuracy')
#     clf.fit(X_train, y_train)

#     print("Best parameters found for classification:", clf.best_params_)
#     return clf.best_estimator_

# def evaluate_classification(model, X_test, y_test):
#     print("\n----- Testing phase (Classification): -----")
#     y_pred_test = model.predict(X_test)
#     accuracy_test = accuracy_score(y_test, y_pred_test)
#     print("Testing Accuracy:", accuracy_test)
#     print("Classification Report (Testing):")
#     print(classification_report(y_test, y_pred_test))

#     return model

# def calculate_permutation_importance(model, X_test, y_test):
#     result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
#     return result

# # Define classifiers to try
# classifiers = {
#     #"Random Forest": RandomForestClassifier(),
#     "SVM": SVC(),
#     "Logistic Regression": LogisticRegression()
# }

# # Hyperparameters for classifiers
# hyperparameters = {
#     # "Random Forest": {
#     #     'n_estimators': [100, 300, 500],
#     #     'max_depth': [None, 10, 20],
#     #     'min_samples_split': [2, 5, 10],
#     #     'min_samples_leaf': [1, 2, 4]
#     # },
#     "SVM": {
#         'C': [0.1, 1, 10],
#         'kernel': ['linear', 'rbf', 'poly'],
#         'gamma': ['scale', 'auto']
#     },
#     "Logistic Regression": {
#         'C': [0.1, 1, 10],
#         'solver': ['liblinear', 'lbfgs', 'sag', 'saga']
#     }
# }

# # Initialize data
# X_train, X_test, y_train, y_test, X, y = init()

# # Train and evaluate each classifier
# for clf_name, clf in classifiers.items():
#     print(f"\nTraining {clf_name}:")
#     clf_model = train_classification(clf, X_train, y_train, hyperparameters[clf_name])
#     clf_model = evaluate_classification(clf_model, X_test, y_test)
#     permutation_importance_result = calculate_permutation_importance(clf_model, X_test, y_test)
#     plot_importance(clf_model, X, permutation_importance_result)




#ran
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("Train_data.csv")

# # Identify categorical columns
# categorical_cols = ['service', 'flag']

# # One-hot encode categorical variables
# encoder = OneHotEncoder(drop='first')
# encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
# encoded_cols.columns = encoder.get_feature_names_out(categorical_cols)

# # Concatenate encoded columns with original dataset
# data_encoded = pd.concat([data.drop(columns=categorical_cols), encoded_cols], axis=1)

# Split data into features and target variable
X = data.drop(columns=['class','service', 'flag'])
y = data['class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your model (example using RandomForestClassifier)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

