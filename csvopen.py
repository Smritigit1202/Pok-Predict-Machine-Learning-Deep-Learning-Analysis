import pandas as pd

from utils import preprocess_features
df = pd.read_csv('data/pokemon.csv')
print(df.columns)
X_train, X_test, y_train, y_test = preprocess_features(df, target_column='Legendary')
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")