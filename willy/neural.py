import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class AccidentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x):
        return self.network(x)


def load_and_preprocess_data(file_path="mvc.csv", sample_fraction=1.0):
    # Read the CSV file
    print("Loading full dataset...")
    df = pd.read_csv(file_path)

    # Take 1% random sample
    print(f"Taking {sample_fraction*100}% random sample...")
    df = df.sample(frac=sample_fraction, random_state=42)
    print(f"Sample size: {len(df)} records")

    # Extract time features
    df["Hour"] = df["Time"].apply(
        lambda x: int(x.split(":")[0]) if isinstance(x, str) else 0
    )
    df["Minute"] = df["Time"].apply(
        lambda x: int(x.split(":")[1]) if isinstance(x, str) else 0
    )

    # Define features based on your column names
    categorical_columns = [
        "Day of Week",
        "Lighting Conditions",
        "Municipality",
        "Collision Type Descriptor",
        "County Name",
        "Road Descriptor",
        "Weather Conditions",
        "Traffic Control Device",
        "Road Surface Conditions",
        "Pedestrian Bicyclist Action",
        "Event Descriptor",
    ]

    binary_columns = ["Police Report"]
    numeric_columns = ["Hour", "Minute", "Number of Vehicles Involved", "Year"]

    # Process categorical columns
    label_encoders = {}
    for column in categorical_columns:
        if column in df.columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].fillna("Unknown"))
            label_encoders[column] = le

    # Process binary columns
    for column in binary_columns:
        if column in df.columns:
            df[column] = (df[column] == "Y").astype(int)

    # Create target variable from Crash Descriptor
    def get_severity(crash_type):
        if pd.isna(crash_type):
            return 0  # Default to property damage
        crash_type = str(crash_type).lower()
        if "fatal" in crash_type:
            return 2  # Fatal
        elif "injury" in crash_type:
            return 1  # Any Injury
        else:
            return 0  # Property Damage Only

    df["Severity"] = df["Crash Descriptor"].apply(get_severity)

    # Print class distribution
    print("\nClass distribution in sample:")
    print(df["Severity"].value_counts().sort_index())

    # Combine all features
    feature_columns = categorical_columns + binary_columns + numeric_columns

    # Handle missing values
    for column in feature_columns:
        if column in df.columns:
            if df[column].dtype in ["int64", "float64"]:
                df[column] = df[column].fillna(df[column].mean())
            else:
                df[column] = df[column].fillna("Unknown")

    # Scale numeric features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_columns])
    y = df["Severity"].values

    return X, y, feature_columns, scaler, label_encoders, df


def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=30, device="cpu"
):
    model = model.to(device)
    best_val_loss = float("inf")
    training_history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_accuracy = 100 * correct / total

        # Store history
        training_history["train_loss"].append(train_loss / len(train_loader))
        training_history["val_loss"].append(val_loss / len(val_loader))
        training_history["val_accuracy"].append(val_accuracy)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, "
                f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%"
            )

    return training_history


def print_stats(y_hat, y_test, model_description=""):
    print(f"Model: {model_description}")
    acc = np.mean(y_hat == y_test)
    print(f'accuracy: {acc}')
    print()

    print('test data counts')
    print(f'no injury: {sum(y_test == 0)}, injury: {sum(y_test == 1)}')
    print('predicted data counts')
    print(f'no injury: {sum(y_hat == 0)}, injury: {sum(y_hat == 1)}')
    print()

    correct_guesses = np.sum(y_hat == y_test)
    false_negatives = np.sum((y_hat == 0) & (y_test == 1))
    false_positives = np.sum((y_hat == 1) & (y_test == 0))
    true_negatives = np.sum((y_hat == 0) & (y_test == 0))
    true_positives = np.sum((y_hat == 1) & (y_test == 1))

    # Calculate total number of samples
    total_samples = len(y_test)

    # Calculate percentages
    correct_guesses_percent = (correct_guesses / total_samples) * 100
    false_negatives_percent = (false_negatives / total_samples) * 100
    false_positives_percent = (false_positives / total_samples) * 100
    true_negatives_percent = (true_negatives / total_samples) * 100
    true_positives_percent = (true_positives / total_samples) * 100

    # out of all predicted injuries, how many are actually injuries
    precision = true_positives / (true_positives + false_positives) * 100

    # out of all actual injuries, how many did the model correcly predict
    recall = true_positives / (true_positives + false_negatives) * 100

    # print(f"Correct guesses: {correct_guesses} ({correct_guesses_percent:.2f}%)")
    print(f"False negatives: {false_negatives} ({false_negatives_percent:.2f}%)")
    print(f"False positives: {false_positives} ({false_positives_percent:.2f}%)")
    print(f"True negatives: {true_negatives} ({true_negatives_percent:.2f}%)")
    print(f"True positives: {true_positives} ({true_positives_percent:.2f}%)")

    print(f"Out of all predicted injuries, how many are actually injuries: {precision:.2f}%")
    print(f"Out of all actual injuries, how many did the model correctly predict: {recall:.2f}%")


def main():
    X, y, feature_columns, scaler, label_encoders, df = load_and_preprocess_data(
        "mvc.csv", sample_fraction=1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create data loaders
    train_dataset = AccidentDataset(X_train, y_train)
    test_dataset = AccidentDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = NeuralNetwork(input_size=len(feature_columns))
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(1.0 / class_counts)
    class_weights = class_weights / class_weights.sum()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = train_model(model, train_loader, test_loader, criterion, optimizer)

    return model, scaler, label_encoders, history


def main2():
    X, y, feature_columns, scaler, label_encoders, df = load_and_preprocess_data(
        "mvc.csv", sample_fraction=1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create data loaders
    train_dataset = AccidentDataset(X_train, y_train)
    test_dataset = AccidentDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = NeuralNetwork(input_size=len(feature_columns))
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(1.0 / class_counts)
    class_weights = class_weights / class_weights.sum()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = train_model(model, train_loader, test_loader, criterion, optimizer)

    # Make predictions on the test set
    model.eval()
    y_hat = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            y_hat.extend(predicted.cpu().numpy())

    y_hat = np.array(y_hat)

    # Print stats
    print_stats(y_hat, y_test, model_description="Neural Network for MVC Severity")

    return model, scaler, label_encoders, history


if __name__ == "__main__":
    # main()
    main2()
