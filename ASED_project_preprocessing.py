import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("Train_data.csv")

# Identify categorical columns
categorical_cols = ['protocol_type','service', 'flag']  #normal=0,anomaly=1

# One-hot encode categorical variables
encoder = OneHotEncoder(drop='first')
encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]).toarray())
feature_names = encoder.get_feature_names_out(categorical_cols)
encoded_cols.columns = feature_names

# Concatenate encoded columns with original dataset
data_encoded = pd.concat([data.drop(columns=categorical_cols), encoded_cols], axis=1)

# Specify the file path for the new CSV file
output_csv_path = "encoded_data.csv"

# Write the encoded DataFrame to a new CSV file
data_encoded.to_csv(output_csv_path, index=False)

# Provide a confirmation message
print("Encoded data saved to:", output_csv_path)
