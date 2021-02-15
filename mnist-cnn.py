import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

im_rows = 28
im_cols = 28
im_color = 1
in_shape = (im_rows, im_cols, im_color)
out_size = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, im_rows, im_cols, im_color)
X_train = X_train.astype('float32')/255
X_test = X_test.reshape(-1, im_rows, im_cols, im_color)
X_test = X_test.astype('float32')/255

y_train = np_utils.to_categorical(y_train.astype('int32'), 10)
y_test = np_utils.to_categorical(y_test.astype('int32'), 10)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.compile(
  loss='categorical_crossentropy',
  optimizer=RMSprop(),
  metrics=['accuracy']
)

hist = model.fit(X_train,
                y_train,
                batch_size = 128,
                epochs = 12,
                verbose = 1,
                validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose = 1)
print("正解率 = ", score[1], "loss = ", score[0])

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
