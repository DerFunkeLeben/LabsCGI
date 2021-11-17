# %%

from sys import exit
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, Flatten, Activation, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PiecewiseConstantDecay, PolynomialDecay, InverseTimeDecay
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import os
from IPython import display
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
EPOCHS = 90
LATENT_SIZE = 32


(x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()
x_trn = x_trn / 255
x_tst = x_tst / 255

encoder = Sequential([
    Flatten(input_shape = (28, 28)),
    Dense(512),
    LeakyReLU(),
    Dropout(0.5),
    Dense(256),
    LeakyReLU(),
    Dropout(0.5),
    Dense(128),
    LeakyReLU(),
    Dropout(0.5),
    Dense(64),
    LeakyReLU(),
    Dropout(0.5),
    Dense(LATENT_SIZE),
    LeakyReLU()
])

decoder = Sequential([
    Dense(64, input_shape = (LATENT_SIZE,)),
    LeakyReLU(),
    Dropout(0.5),
    Dense(128),
    LeakyReLU(),
    Dropout(0.5),
    Dense(256),
    LeakyReLU(),
    Dropout(0.5),
    Dense(512),
    LeakyReLU(),
    Dropout(0.5),
    Dense(784),
    Activation("sigmoid"),
    Reshape((28, 28))
])


img = Input(shape = (28, 28))
latent_vector = encoder(img)
output = decoder(latent_vector)

model = Model(inputs = img, outputs = output)
model.compile(optimizer=RMSprop(1e-3), loss = "binary_crossentropy")

dataGen = ImageDataGenerator(rotation_range=15,width_shift_range=0.2,height_shift_range=0.2,
                             shear_range=0.15,zoom_range=[0.5,2],validation_split=0.2, featurewise_center = True)
dataGen.fit(x_trn.reshape(60000, 28, 28, 1))

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.001)
]

model.fit(x_trn, x_trn, epochs=EPOCHS, callbacks=callbacks, validation_split=0.2)

model.summary()
encoder.summary()
decoder.summary()

#for epoch in range(EPOCHS):
fig, axs = plt.subplots(4, 4)
rand = x_tst[np.random.randint(0, 10000, 16)].reshape((4, 4, 1, 28, 28))

#display.clear_output()

print("Generated images:")
for i in range(4):
    for j in range(4):
        axs[i, j].imshow(model.predict(rand[i, j])[0], cmap = "gray")
        axs[i, j].axis("off")

plt.subplots_adjust(wspace = 0, hspace = 0)
plt.show()

print("Target images:")
for X_batch, y_batch in dataGen.flow(x_trn.reshape(60000, 28, 28, 1), x_trn.reshape(60000, 28, 28, 1), batch_size=9):
	for i in range(0, 9):
		plt.subplot(330 + 1 + i)
		plt.imshow(X_batch[i].reshape(28, 28), cmap='gray')
	plt.show()
	break


# %%