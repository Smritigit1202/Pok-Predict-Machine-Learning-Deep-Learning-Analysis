from utils import load_data, preprocess_features
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

df = load_data()
X_train, X_test, y_train, y_test = preprocess_features(df, target_column='Legendary')

# Encode target for keras
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Deep Learning Accuracy: {accuracy}")
