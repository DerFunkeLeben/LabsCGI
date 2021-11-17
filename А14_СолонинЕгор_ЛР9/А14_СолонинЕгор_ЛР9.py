# %%

from mnist.loader import MNIST
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
pathToData = 'D:\\mpei\\LabsCGI\\А14_СолонинЕгор_ЛР9\\'
# pathToData – ранее заданный путь к папке с данными
mndata = MNIST(pathToData)
mndata.gz = True  # Разрешаем чтение архивированных данных
# Обучающая выборка (данные и метки)
imagesTrain, labelsTrain = mndata.load_training()
# Тестовая выборка (данные и метки)
imagesTest, labelsTest = mndata.load_testing()
X_train = np.asarray(imagesTrain)
y_train = np.asarray(labelsTrain)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

pixels = 5
shift_pixels = np.zeros(pixels)

# Выводим 9 изображений обучающего набора
names = []

for i in range(10):
    names.append(chr(48 + i))  # ['0', '1', '2', ..., '9']
for i in range(9):
    plt.subplot(3, 3, i + 1)

    ind = y_train[i]
    img_top = X_train[i][0:14, 0:28, 0]
    img_btm = X_train[i][15:28, 0:28, 0]

    for row in range(len(img_top)):
        img_top[row] = np.concatenate(
            (img_top[row, pixels:28], shift_pixels), axis=0)

    for row in range(len(img_btm)):
        img_btm[row] = np.concatenate(
            (shift_pixels, img_btm[row, 0:28-pixels]), axis=0)

    img = np.concatenate((img_top, img_btm), axis=0)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.title(names[ind])
    plt.axis('off')

plt.subplots_adjust(hspace=0.5)
plt.show()


# %%
