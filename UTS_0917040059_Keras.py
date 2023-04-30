import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.callbacks import Callback
import warnings

X1 = np.array([1])
X2 = np.array([2])
X3 = np.array([0.5])
Y1 = np.array([-1])
Y2 = np.array([1])
Y3 = np.array([0])

merged_input = np.stack([X1, X2, X3], axis=1)
merged_output = np.stack([Y1, Y2], axis=1)
# X = 1,2,0.5
# Y = -1,1

 
# define the keras model
model = Sequential()
model.add(Dense(4, input_dim=3, activation='sigmoid'))
model.add(Dense(2, activation='tanh'))

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='loss', value=0.0001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping by value error" % epoch)
            self.model.stop_training = True

# compile the keras model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
early_stopping_monitor = [
    EarlyStoppingByLossVal(monitor='loss', value=1e-4, verbose=1)
    # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
  
]
model.fit(merged_input, merged_output, epochs=150, batch_size=10, validation_data=(merged_input, merged_output),callbacks = [early_stopping_monitor])


# Evaluate the Model
score = model.evaluate(merged_input, merged_output, verbose=0)
print('Test accuracy:', score[1])

# Prediction
model.predict(merged_input)

# Summarize the Model
model.summary()
weights1 = model.layers[0].get_weights()[0]
bias1    = model.layers[0].get_weights()[1]
weights2 = model.layers[1].get_weights()[0]
bias2    = model.layers[1].get_weights()[1]
print("IW : ")
print(weights1)
print("bIW : ")
print(bias1)
print("LW : ")
print(weights2)
print("bLW : ")
print(bias2)