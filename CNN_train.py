import pandas as pd
import numpy as np
from keras import Sequential
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape, LSTM, Dense

df = pd.read_csv('train_image_paths_new.csv')

# 讀取圖片並進行資料處理
X = []
y = []
for _, row in df.iterrows():
    img = load_img(row['path'], target_size=(64, 64, 1))
    img_array = img_to_array(img) / 255.0  # 正規化像素值
    X.append(img_array)
    y.append(row['state'])

# 轉換成NumPy陣列
X = np.array(X)
y = np.array(y)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.0625, random_state=42)

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten(input_shape=(64, 64, 1)))

# Reshaping the output of the previous layer to a 3D tensor
classifier.add(Reshape((1, -1)))

# Add a LSTM Layer
classifier.add(LSTM(units=128))

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 32

classifier.fit(X_train,
               y_train,
               batch_size=batch_size,
               epochs=10,
               validation_data=(X_val, y_val))

# 評估模型
loss, accuracy = classifier.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')