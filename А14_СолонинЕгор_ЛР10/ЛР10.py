# %%

from sys import exit
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
#
pathToData = "D:/mpei/LabsCGI/А14_СолонинЕгор_ЛР10/mnistBinData/"
pathToEmnistData = "D:/mpei/LabsCGI/А14_СолонинЕгор_ЛР10/emnistBinData/"
img_rows = img_cols = 28
num_classes = 11
show_img = not True
batch_size = 256  # Размер обучающего (проверочного) пакета
train_model = not True
save_model = not True
load_model = True
show_model = True
fn_w = 'model.wei'  # Файл с весами НС
n_epochs = 300  # Число эпох обучения
criterion = nn.CrossEntropyLoss()  # Функция потерь
#
accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.max_pool2d1 = nn.MaxPool2d(2)
        self.max_pool2d2 = nn.MaxPool2d(2)
        self.conv2_drop = nn.Dropout2d(p=0.3)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(800, 32)  # 800 - считаем сами
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool2d1(x)
        x = self.conv2_drop(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2d2(x)
        x = x.view(-1, 800)  # 800 - считаем сами (flatten)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # torch.Size([256, 10])
        return F.log_softmax(x, dim=-1)


model = Net()  # Формируем модель НС
if show_model:
    print(model)


def load_bin_data(pathToData, img_rows, img_cols, show_img):
    print('Загрузка данных из двоичных файлов')
    with open(pathToData + 'imagesTrain.bin', 'rb') as read_binary:
        x_trn = np.fromfile(read_binary, dtype=np.uint8)
    with open(pathToData + 'labelsTrain.bin', 'rb') as read_binary:
        y_trn = np.fromfile(read_binary, dtype=np.uint8)
    with open(pathToData + 'imagesTest.bin', 'rb') as read_binary:
        x_tst = np.fromfile(read_binary, dtype=np.uint8)
    with open(pathToData + 'labelsTest.bin', 'rb') as read_binary:
        y_tst = np.fromfile(read_binary, dtype=np.uint8)

    pathToData = pathToData.replace("mnist", "emnist")

    with open(pathToData + 'imagesTrain.bin', 'rb') as read_binary:
        x_trn_e = np.fromfile(read_binary, dtype=np.uint8)
    with open(pathToData + 'labelsTrain.bin', 'rb') as read_binary:
        y_trn_e = np.fromfile(read_binary, dtype=np.uint8)
    with open(pathToData + 'imagesTest.bin', 'rb') as read_binary:
        x_tst_e = np.fromfile(read_binary, dtype=np.uint8)
    with open(pathToData + 'labelsTest.bin', 'rb') as read_binary:
        y_tst_e = np.fromfile(read_binary, dtype=np.uint8)

    x_trn = x_trn.reshape(-1, 1, img_rows, img_cols)
    x_tst = x_tst.reshape(-1, 1, img_rows, img_cols)
    x_trn_e = x_trn_e.reshape(-1, 1, img_rows, img_cols)
    x_tst_e = x_tst_e.reshape(-1, 1, img_rows, img_cols)

    x_tst_e, y_tst_e = build_11th_class(x_tst_e, y_tst_e, 1000)
    x_trn_e, y_trn_e = build_11th_class(x_trn_e, y_trn_e, 6000)

    x_trn = np.concatenate((x_trn, x_trn_e), axis=0)
    y_trn = np.concatenate((y_trn, y_trn_e), axis=0)
    x_tst = np.concatenate((x_tst, x_tst_e), axis=0)
    y_tst = np.concatenate((y_tst, y_tst_e), axis=0)

    if show_img:
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            img = x_trn[i].reshape(img_rows, img_cols)
            plt.imshow(img, cmap=plt.get_cmap('gray'))
            plt.title(y_trn[i])
            plt.axis('off')
        plt.subplots_adjust(hspace=0.5)
        plt.show()
    x_trn = np.array(x_trn, dtype='float32') / 255
    x_tst = np.array(x_tst, dtype='float32') / 255
    return x_trn, y_trn, x_tst, y_tst


def build_11th_class(x, y, ammount):
    repres_count = np.zeros(27)
    repres_1_max = 231 if ammount == 6000 else 39
    repres_2_max = 230 if ammount == 6000 else 38
    counter = 0
    border = 20 if ammount == 6000 else 12
    x_res = x

    for i in range(len(y)):
        if counter == ammount:
            break
        if y[i] > border and repres_count[y[i]] < repres_2_max:
            x_res[counter] = x[i]
            repres_count[y[i]] += 1
            counter += 1
        if y[i] <= border and repres_count[y[i]] < repres_1_max:
            x_res[counter] = x[i]
            repres_count[y[i]] += 1
            counter += 1

    x_res = x_res[:ammount]
    x_res = x_res.transpose(0, 1, 3, 2)
    y_res = np.ones(ammount) * 10
    return x_res, y_res


x_trn, y_trn, x_tst, y_tst = load_bin_data(
    pathToData, img_rows, img_cols, show_img)

trn_data = [[x, int(y)] for x, y in zip(x_trn, y_trn)]
tst_data = [[x, int(y)] for x, y in zip(x_tst, y_tst)]

# Формирование обучающих и проверочных пакетов
trn_loader = DataLoader(trn_data, batch_size=batch_size, shuffle=True)
tst_loader = DataLoader(tst_data, batch_size=batch_size, shuffle=True)
# Число примеров в обучающем и проверочном множествах
in_trn = len(trn_loader.sampler)  # 66000
in_tst = len(tst_loader.sampler)  # 11000
if show_img:
    trn_features, trn_labels = next(iter(trn_loader))
    img = trn_features[0].squeeze()
    ind = int(trn_labels[0])
    plt.imshow(img, cmap='gray')
    plt.title(ind)
    plt.axis('off')
    plt.show()


def train(epoch):
    trn_loss = tst_loss = 0
    trn_correct = tst_correct = 0
    model.train()  # Режим обучения
    for batch_no, (data, target) in enumerate(trn_loader):
        optimizer.zero_grad()  # Обнуляем градиенты
        output = model(data)  # forward
        loss = criterion(output, target)  # <class 'torch.Tensor'>
        loss.backward()  # Подготовка потерь для обратного шага
        optimizer.step()  # Обратное распространение ошибки, обновляем веса модели
        trn_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        trn_correct += (pred == target).sum().item()

    model.eval()  # Режим оценки
    for data, target in tst_loader:
        output = model(data)
        loss = criterion(output, target)
        tst_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        tst_correct += (pred == target).sum().item()

    trn_loss = trn_loss / in_trn
    tst_loss = tst_loss / in_tst

    trn_prec = trn_correct / in_trn * 100
    tst_prec = tst_correct / in_tst * 100

    loss_stats['train'].append(trn_loss)
    loss_stats['val'].append(tst_loss)
    accuracy_stats['train'].append(trn_prec)
    accuracy_stats['val'].append(tst_prec)

    print('\nЭпоха: {}\n\t Потери: обучение: {:.6f} \tпроверка: {:.6f}\n\t Точность: обучение: {:.2f}% \tпроверка: {:.2f}%'.format(
        epoch + 1, trn_loss, tst_loss, trn_prec, tst_prec))


if load_model:
    print('Загрузка весов из файла', fn_w)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(fn_w, map_location=torch.device(device)))
if train_model:
    print('Число эпох', n_epochs)
    # betas = (0.9, 0.98), eps = 1e-9
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print('Обучение')
    for ep in range(n_epochs):
        train(ep)
    if save_model:
        print('Сохранение весов модели')
        torch.save(model.state_dict(), fn_w)


def test():
    print('Проверка')
    tst_loss = 0  # Потери на проверочном множестве (ПМ)
    y_pred_list = []
    # Число верно классифицированных цифр в каждом классе (ПМ)
    cls_correct = [0]*num_classes
    # Число цифр в каждом классе (ПМ)
    cls_total = [0]*num_classes
    model.eval()  # Режим оценки
    for data, target in tst_loader:

        output = model(data)
        loss = criterion(output, target)
        tst_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        y_pred_list.append(pred.cpu().numpy())
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        for i in range(len(target)):
            label = target.data[i]
            cls_correct[label] += correct[i].item()
            cls_total[label] += 1
    # Средние потери
    tst_loss /= len(tst_loader.sampler)
    print('Потери: {:.6f}'.format(tst_loss))
    print('Точность: {:.4f}'.format(sum(cls_correct) / sum(cls_total)))

    print('Точность по классам:')
    cls = 0
    for cc, ct in zip(cls_correct, cls_total):
        print('{} - {:.4f}%'.format(cls, cc / ct * 100))
        cls += 1


def show_res():
    print('Смотрим начальный кусок первого пакета')
    data_iter = iter(tst_loader)  # Первый пакет
    images, labels = data_iter.next()
    output = model(images)
    _, pred = torch.max(output, 1)
    images = images.numpy()
    fig = plt.figure(figsize=(12, 3))
    N = 20
    for i in range(N):
        ax = fig.add_subplot(2, int(N / 2), i + 1, xticks=[], yticks=[])
        img = images[i]  # (1, 28, 28)
        img = np.squeeze(img)  # или: img = img.reshape(img_rows, img_cols)
        ax.imshow(img, cmap='gray')
        ttl = str(int(pred[i].item())) + ' (' + str(labels[i].item()) + ')'
        clr = 'green' if pred[i] == labels[i] else 'red'
        ax.set_title(ttl, color=clr)
    plt.show()

    show_graph()
    print(
        f"Classification report:\n"
        f"{classification_report(pred, labels)}\n"
    )


def show_err():
    data_iter = iter(tst_loader)  # Первый пакет
    images, labels = data_iter.next()
    output = model(images)
    _, pred = torch.max(output, 1)
    images = images.numpy()
    fig = plt.figure(figsize=(12, 3))
    N = 20
    i = count = 0
    while count < N:
        ax = fig.add_subplot(2, int(N / 2), count + 1, xticks=[], yticks=[])
        
        img = images[i]  # (1, 28, 28)
        img = np.squeeze(img)  # или: img = img.reshape(img_rows, img_cols)
        if pred[i] != labels[i]:
            ax.imshow(img, cmap='gray')
            ttl = str(int(pred[i].item())) + ' (' + str(labels[i].item()) + ')'
            ax.set_title(ttl, color='red')
            count += 1
            print(count)
        i += 1

        if i > 255:
            images, labels = data_iter.next()
            output = model(images)
            _, pred = torch.max(output, 1)
            images = images.numpy()
            i = 0

    plt.show()

    show_graph()
    print(
        f"Classification report:\n"
        f"{classification_report(pred, labels)}\n"
    )


def show_graph():
    # Create dataframes
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(
        id_vars=['index']).rename(columns={"index": "epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(
        id_vars=['index']).rename(columns={"index": "epochs"})
    # Plot the dataframes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    sns.lineplot(data=train_val_acc_df, x="epochs", y="value",
                 hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x="epochs", y="value",
                 hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')


test()  # Проверка
# show_res()  # Смотрим начальный кусок первого пакета
show_err()

# %%
