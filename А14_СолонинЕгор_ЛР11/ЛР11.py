# %%
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, ReLU, Add, DepthwiseConv2D, Conv2D, ZeroPadding2D, AveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow import image
import numpy as np
import cv2
import matplotlib.pyplot as plt


def resize_x(input):
    sample_shape = input.shape[0]
    x_imgs_2 = input.reshape(sample_shape, 1, 28, 28).transpose(0, 2, 3, 1)
    x_imgs = np.zeros((sample_shape, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i in range(sample_shape):
        x = cv2.resize(x_imgs_2[i], (IMG_SIZE, IMG_SIZE))
        x_imgs[i] = x
    x_imgs = x_imgs.reshape(sample_shape, IMG_SIZE, IMG_SIZE, 1)
    return x_imgs


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

        predicted = lst[1]
        real_class = lst[2]

        plt.title('{}/{}'.format(predicted, real_class))
        plt.axis('off')
    plt.show()


NUM_CLASSES = 10
IMG_SIZE = 32
EPOCHS = 60
train_model = not True
load_model = True
fn_model = "D:/mpei/LabsCGI/А14_СолонинЕгор_ЛР11/" + 'lk3.h5'

(x_train, y_train), (x_test, y_test) = load_data()
x_train = resize_x(x_train)
x_test = resize_x(x_test)

if train_model:
    base_model = MobileNetV2(weights=None, include_top=True,
                             input_shape=(IMG_SIZE, IMG_SIZE, 1))

    nL = len(base_model.layers)
    layer = base_model.layers[len(base_model.layers) - 1]
    layer.activation = None
    x = base_model.output
    x = Activation('relu')(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer='Adam', loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    # model.summary()

    history = model.fit(x_train, y_train, batch_size=128, epochs=EPOCHS,
                        verbose=2, validation_data=(x_test, y_test))
    print('Модель сохранена в файле', fn_model)
    model.save(fn_model)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if load_model:
    from keras.models import load_model
    model = load_model(fn_model)

    score_test = model.evaluate(x_test, y_test, verbose=0)
    score_train = model.evaluate(x_train, y_train, verbose=0)
    print('Потери при тестировании (test):', round(score_test[0], 4))
    print('Точность при тестировании (test): {}{}'.format(
        score_test[1] * 100, '%'))

    print('Потери при тестировании (train):', round(score_train[0], 4))
    print('Точность при тестировании (train): {}{}'.format(
        score_train[1] * 100, '%'))

    # Прогноз
    y_pred = model.predict(x_train)
    print(y_pred[0])
    print(y_train[0])
    predicted_classes = np.array([np.argmax(m) for m in y_pred])
    true_classes = y_train
    print("predicted_classes", predicted_classes)
    print("true_classes", true_classes)
    n_test = len(y_train)
    # Число верно классифицированных изображений
    true_classified = np.sum(predicted_classes == true_classes)
    # Число ошибочно классифицированных изображений
    false_classified = n_test - true_classified
    acc = 100.0 * true_classified / n_test
    print('Точность: {}{}'.format(acc, '%'))
    print('Неверно классифицированно:', false_classified)

    showData(n_test, predicted_classes, true_classes, x_train,
             False, IMG_SIZE, IMG_SIZE)


# %%
