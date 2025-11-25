import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def generate_california_house_prices_mapping(n_train=500, n_val=250, n_test=250, num_houses_per_instance=10):
    # Load processed_data
    california_housing = fetch_california_housing(as_frame=True)
    df = california_housing.frame

    # Split into features and targets
    features = df.drop('MedHouseVal', axis=1).values
    targets = df['MedHouseVal'].values

    # Split into training and test sets
    features_tmp, features_test, targets_tmp, targets_test = train_test_split(
        features, targets, test_size=0.25, random_state=42
    )
    features_train, features_val, targets_train, targets_val = train_test_split(
        features_tmp, targets_tmp, test_size=0.33, random_state=42
    )

    # Standardize features separately for train and test
    s_scaler = StandardScaler()
    features_train = s_scaler.fit_transform(features_train.astype(np.float))
    features_val = s_scaler.transform(features_val.astype(np.float))
    features_test = s_scaler.transform(features_test.astype(np.float))

    def create_instances(features, targets, n_instances, num_houses_per_instance):
        # Calculate how many houses we need
        total_houses_needed = n_instances * num_houses_per_instance
        
        # If we don't have enough houses, we'll need to sample with replacement
        if total_houses_needed > len(features):
            # Sample with replacement to get enough houses
            indices = np.random.choice(len(features), size=total_houses_needed, replace=True)
            features_sampled = features[indices]
            targets_sampled = targets[indices]
        else:
            # Shuffle and take the first total_houses_needed houses
            indices = np.random.permutation(len(features))[:total_houses_needed]
            features_sampled = features[indices]
            targets_sampled = targets[indices]

        # Reshape into instances
        features_reshaped = features_sampled.reshape(n_instances, num_houses_per_instance, features.shape[1])
        targets_reshaped = targets_sampled.reshape(n_instances, num_houses_per_instance)

        return features_reshaped, targets_reshaped

    # Create training and test instances
    train_features, train_targets = create_instances(
        features_train, targets_train, n_train, num_houses_per_instance
    )
    val_features, val_targets = create_instances(
        features_val, targets_val, n_train, num_houses_per_instance
    )
    test_features, test_targets = create_instances(
        features_test, targets_test, n_test, num_houses_per_instance
    )

    return train_features, train_targets, val_features, val_targets, test_features, test_targets
