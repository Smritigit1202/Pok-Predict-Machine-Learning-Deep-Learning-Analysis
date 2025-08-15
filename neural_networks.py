from utils import load_data, preprocess_features
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = load_data()
X_train, X_test, y_train, y_test = preprocess_features(df, target_column='Legendary')

model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Neural Network Accuracy: {accuracy}")
