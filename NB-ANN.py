import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

# Disable TensorFlow Notifications Messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


df = pd.read_csv('diabetes1.csv')


X = df.iloc[:, :8]
y = df.iloc[:, -1]

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3)

# Create a Naive Bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)


y_pred = nb.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("NB Accuracy: %", accuracy*100)


# Create ANN sequential model
model = Sequential()

# Add the first layer (input layer)
model.add(Dense(8, input_dim=8, activation='relu'))

# Add the second layer (hidden layer)
model.add(Dense(4, activation='relu'))

# Add the third layer (output layer)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the training data
model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)

# Evaluate the model on the testing data
_, accuracy2 = model.evaluate(X_test, y_test, verbose=0)
print("ANN Accuracy: %", accuracy2*100)