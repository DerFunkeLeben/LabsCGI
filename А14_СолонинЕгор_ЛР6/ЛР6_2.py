# %%
from sys import exit
from keras.backend import dropout
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization  # Activation
import random
import time
#
start = time.time()


def create_model(n_classes, inp_shape, c_filters, d_units, kernel_size, c_strides,
                 pool_size, p_strides, k_init, loss, metrics):
    inp = Input(shape=inp_shape)
    conv = Conv2D(c_filters[0], kernel_size=kernel_size, strides=c_strides,
                  padding='same', kernel_initializer=k_init, bias_initializer=k_init,
                  activation='relu')(inp)
    pool = MaxPooling2D(pool_size=pool_size,
                        strides=p_strides, padding='same')(conv)
    conv2 = Conv2D(c_filters[1], kernel_size=kernel_size, strides=c_strides,
                   padding='same', kernel_initializer=k_init, bias_initializer=k_init,
                   activation='relu')(pool)
    pool2 = MaxPooling2D(pool_size=pool_size,
                         strides=p_strides, padding='same')(conv2)
    flat = Flatten()(pool2)
    dense = Dense(d_units[0], activation='relu')(flat)
    dense2 = Dense(d_units[1], activation='linear')(dense)
    output = Dense(n_classes, activation='softmax')(dense2)
    model = Model(inputs=inp, outputs=output)
    # Adam – имя метода оптимизации (минимизируются потери)
    # Компиляция нейронной сети
    model.compile('Adam', loss=loss, metrics=metrics)
    model.summary()  # Вывод сведений о нейронной сети
    return model


def show_x(x, img_rows, img_cols, N):
    n = int(np.sqrt(N))
    print(x[0].shape, len(x), n)
    for i, j in enumerate(np.random.randint(10000, size=n*n)):
        plt.subplot(n, n, i + 1)
        # Убираем 3-е измерение
        plt.imshow(x[j].reshape(img_rows, img_cols), cmap='gray')
        plt.axis('off')
    plt.show()
#
# Вывод графиков


def one_plot(n, y_lb, loss_acc, val_loss_acc):
    plt.subplot(1, 2, n)
    if n == 1:
        lb, lb2 = 'loss', 'val_loss'
        yMin = 0
        yMax = 1.05 * max(max(loss_acc), max(val_loss_acc))
    else:
        lb, lb2 = 'acc', 'val_acc'
        yMin = min(min(loss_acc), min(val_loss_acc))
        yMax = 1.0
    plt.plot(loss_acc, color='r', label=lb, linestyle='--')
    plt.plot(val_loss_acc, color='g', label=lb2)
    plt.ylabel(y_lb)
    plt.xlabel('Эпоха')
    plt.ylim([0.95 * yMin, yMax])
    plt.legend()
#


def loadBinData(pathToData, img_rows, img_cols, mode, num_classes):
    print('Загрузка данных из двоичных файлов...')
    with open(pathToData + 'imagesTrain.bin', 'rb') as read_binary:
        x_train = np.fromfile(read_binary, dtype=np.uint8)
    with open(pathToData + 'labelsTrain.bin', 'rb') as read_binary:
        y_train = np.fromfile(read_binary, dtype=np.uint8)
    with open(pathToData + 'imagesTest.bin', 'rb') as read_binary:
        x_test = np.fromfile(read_binary, dtype=np.uint8)
    with open(pathToData + 'labelsTest.bin', 'rb') as read_binary:
        y_test = np.fromfile(read_binary, dtype=np.uint8)
    # Преобразование целочисленных данных в float32 и нормализация; данные лежат в диапазоне [0.0, 1.0]
    x_train = np.array(x_train, dtype='float32') / 255
    x_test = np.array(x_test, dtype='float32') / 255

    print(x_test.shape)
    x_train_shape_0 = int(x_train.shape[0] / (img_rows * img_cols))
    x_test_shape_0 = int(x_test.shape[0] / (img_rows * img_cols))
    # 1 - оттенок серого цвета
    x_train = x_train.reshape(x_train_shape_0, img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test_shape_0, img_rows, img_cols, 1)
    # Преобразование в категориальное представление: метки - числа из диапазона [0, 9] в двоичный вектор размера num_classes
    # Так, в случае MNIST метка 5 (соответствует классу 6) будет преобразована в вектор [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
    # print(y_train[0]) # (MNIST) Напечатает: 5
    print('Преобразуем массивы меток в категориальное представление')
    if mode == 'emnist':
        y_train += 9  # -1+10
        y_test += 9
        x_train = x_train.transpose(0, 2, 1, 3)
        x_test = x_test.transpose(0, 2, 1, 3)
    # print(y_train[0])
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # print(y_train[0]) # (MNIST) Напечатает: [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    return x_train, y_train, x_test, y_test
#


def showData(n_test, predicted_classes, true_classes, x_test, showTrue, img_rows, img_cols):
    list_of_images = []
    print('Индекс | Прогноз | Правильный класс')
    for i in range(n_test):
        cls_pred = predicted_classes[i]  # Предсказанное моделью имя класса
        cls_true = true_classes[i]  # Истинное имя класса
        if not showTrue and cls_pred != cls_true:
            list_of_images.append([i, cls_pred, cls_true])
            print('  {}   |   {}    |    {}'.format(i, cls_pred, cls_true))
        if showTrue and cls_pred == cls_true:
            list_of_images.append([i, cls_pred, cls_true])
    plt.figure('Ошибки классификации')

    randindexes = np.arange(list_of_images.__len__())
    np.random.shuffle(randindexes)
    randindexes = randindexes[:15]

    for k in range(len(randindexes)):
        plt.subplot(3, 5, k + 1)
        index = randindexes[k]
        lst = list_of_images[index]
        plt.imshow(x_test[lst[0]].reshape(img_rows, img_cols), cmap='gray')

        predicted = lst[1] if lst[1] < 10 else chr(55 + lst[1])  # 65-10
        real_class = lst[2] if lst[2] < 10 else chr(55 + lst[2])

        plt.title('{}/{}'.format(predicted, real_class))
        plt.axis('off')
    plt.show()


def accuracyPerClass(predicted_classes, true_classes):
    truePerClass = np.zeros(36)
    predictedPerClass = np.zeros(36)
    acc = np.ones(36) * 100

    for i in predicted_classes:
        predictedPerClass[i] += 1

    for i in true_classes:
        truePerClass[i] += 1

    for i in range(36):
        acc[i] *= min(truePerClass[i], predictedPerClass[i]) / \
            max(truePerClass[i], predictedPerClass[i])

    print('Класс | Точность ')
    for i in range(36):
        print('  {}   |   {:3.2f}{}    '.format(i if i < 10 else chr(
            55 + i), acc[i], "%"))


def program():
    seedVal = 348
    mode = 'train'  # 'train' 'test'
    img_rows = img_cols = 28
    show_k = False  # False True
    pred = True
    show_true_predicted = True
    folder = "D:/mpei/LabsCGI/А14_СолонинЕгор_ЛР6/"
    epochs = 20
    fn_model = folder + "bothBinData/" + 'lk3.h5'
    #
    pathToHistory = folder + "bothHistory/"
    suff = '.txt'
    # Имена файлов, в которые сохраняется история обучения
    fn_loss = pathToHistory + 'loss_' + suff
    fn_acc = pathToHistory + 'acc_' + suff
    fn_val_loss = pathToHistory + 'val_loss_' + suff
    fn_val_acc = pathToHistory + 'val_acc_' + suff

    # Загрузка обучающего и проверочного множества из бинарных файлов
    # Загружаются изображения и их метки
    x_train, y_train, x_test, y_test = loadBinData(
        folder + "mnistBinData/", img_rows, img_cols, 'mnist', 36)
    x_train_emnist, y_train_emnist, x_test_emnist, y_test_emnist = loadBinData(
        folder + "emnistBinData/", img_rows, img_cols, 'emnist', 36)

    x_train = x_train[:30000]
    y_train = y_train[:30000]
    x_test = x_test[:5000]
    y_test = y_test[:5000]
    x_train_emnist = x_train_emnist[:30000]
    y_train_emnist = y_train_emnist[:30000]
    x_test_emnist = x_test_emnist[:5000]
    y_test_emnist = y_test_emnist[:5000]

    x_train = np.concatenate((x_train, x_train_emnist), axis=0)
    y_train = np.concatenate((y_train, y_train_emnist), axis=0)
    x_test = np.concatenate((x_test, x_test_emnist), axis=0)
    y_test = np.concatenate((y_test, y_test_emnist), axis=0)

    if show_k:
        show_x(x_test, img_rows, img_cols, 16, y_test)
        exit()
    if pred:
        from keras.models import load_model
        model = load_model(fn_model)

        if mode == 'test':
            score = model.evaluate(x_test, y_test, verbose=0)
        else:
            score = model.evaluate(x_train, y_train, verbose=0)
        # Вывод потерь и точности
        print('Потери при тестировании:', round(score[0], 4))
        print('Точность при тестировании: {}{}'.format(score[1] * 100, '%'))
        # Прогноз
        y_pred = model.predict(
            x_test) if mode == 'test' else model.predict(x_train)

        # [6.8e-6 1.5e-10 7.6e-6 1.5e-3 7.0e-9 6.2e-5 2.2e-11 9.9e-1 3.0e-7 5.9e-6]
        # [0.     0.      0.     0.     0.     0.     0.      1.     0.     0.]
        # Заносим в массив predicted_classes метки классов, предсказанных моделью НС
        predicted_classes = np.array([np.argmax(m) for m in y_pred])
        if mode == 'test':
            true_classes = np.array([np.argmax(m) for m in y_test])
        else:
            true_classes = np.array([np.argmax(m) for m in y_train])

        n_test = len(y_test) if mode == 'test' else len(y_train)
        # Число верно классифицированных изображений
        true_classified = np.sum(predicted_classes == true_classes)
        # Число ошибочно классифицированных изображений
        false_classified = n_test - true_classified
        acc = 100.0 * true_classified / n_test
        print('Точность: {}{}'.format(acc, '%'))
        print('Неверно классифицированно:', false_classified)

        accuracyPerClass(predicted_classes, true_classes)

        showData(n_test, predicted_classes, true_classes, x_test if mode == 'test' else x_train,
                 True if show_true_predicted else False,
                 img_rows, img_cols)

        exit()
    #
    # Определяем форму входных данных
    input_shape = (img_rows, img_cols, 1)
    #
    # Создание модели нейронной сети
    model = create_model(36, input_shape, c_filters=[16, 16], d_units=[600, 600], kernel_size=3, c_strides=1,
                         pool_size=2,  p_strides=2, k_init=keras.initializers.RandomNormal(seed=seedVal), loss='mse', metrics=['accuracy'])
    #
    # Обучение нейронной сети
    history = model.fit(x_train, y_train, batch_size=128, epochs=epochs,
                        verbose=2, validation_data=(x_test, y_test))
    print('Модель сохранена в файле', fn_model)
    model.save(fn_model)
    # Запись истории обучения в текстовые файлы
    history = history.history
    ##for itm in history.items(): print(itm)
    with open(fn_loss, 'w') as output:
        for val in history['loss']:
            output.write(str(val) + '\n')
    with open(fn_acc, 'w') as output:
        for val in history['accuracy']:
            output.write(str(val) + '\n')
    with open(fn_val_loss, 'w') as output:
        for val in history['val_loss']:
            output.write(str(val) + '\n')
    with open(fn_val_acc, 'w') as output:
        for val in history['val_accuracy']:
            output.write(str(val) + '\n')
    # Вывод графиков обучения
    plt.figure(figsize=(9, 4))
    plt.subplots_adjust(wspace=0.5)
    one_plot(1, 'Потери', history['loss'], history['val_loss'])
    one_plot(2, 'Точность', history['accuracy'], history['val_accuracy'])
    plt.suptitle('Потери и точность')
    plt.show()


program()
print('Время вычислений:', time.time() - start)

# %%
