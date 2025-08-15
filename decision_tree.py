from utils import load_data, preprocess_features
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = load_data()
X_train, X_test, y_train, y_test = preprocess_features(df, target_column='Type_1')

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy}")
