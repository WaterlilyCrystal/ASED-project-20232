import pandas as pd

train_data = pd.read_csv("Train_data_new.csv")
def detect_anomalies(data):
    # Combine all numerical features into a single series
    combined_series = pd.concat([data[column] for column in data.columns if column not in ['class','service','flag']])
    # Set anomaly threshold as mean ± 3 standard deviations
    mean = combined_series.mean()
    std = combined_series.std()
    anomaly_threshold = mean + 3 * std
    # Find data points exceeding anomaly threshold
    anomalies = combined_series[combined_series > anomaly_threshold].index
    print("Anomalies:", anomalies)
    return anomalies

predicted_anomalies_indices = detect_anomalies(train_data)

# Function to compare predicted anomalies with real test results
def compare_results(predicted_anomalies, real_anomalies):
    true_positives = len(set(predicted_anomalies) & set(real_anomalies))
    false_positives = len(set(predicted_anomalies) - set(real_anomalies))
    false_negatives = len(set(real_anomalies) - set(predicted_anomalies))
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = true_positives / len(real_anomalies)
    
    results = {
        "true_positives":true_positives,
        "false_positives":false_positives,
        "false_negatives":false_negatives,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Accuracy": accuracy
    }
    return results

real_test_anomalies_indices = train_data[train_data['class'] == 1].index
train_results = compare_results(predicted_anomalies_indices, real_test_anomalies_indices)

with open("statistic_accuracy.txt", "a") as f:
    for key, value in train_results.items():
        f.write(f"{key}: {value}\n")