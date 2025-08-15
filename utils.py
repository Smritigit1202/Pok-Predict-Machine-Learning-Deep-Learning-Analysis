import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path='data/pokemon.csv'):
    df = pd.read_csv(path)
    df = df.dropna()
    # Standardize column names (remove spaces, lowercase)
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    return df


def preprocess_features(df, target_column, scale=True):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    
    # Optionally scale numeric features
    if scale:
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
