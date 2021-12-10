# %%
import numpy as np
import matplotlib.pyplot as plt

w = 80  # размер матрицы пикселей
vp = [[255]*w for i in range(w)]

add = 15
b = 70
x_left_bound = 2
x_right_bound = 10
y_left_bound = 2
y_right_bound = b


def getRand(l, r):
    return np.random.randint(l, r, size=1)[0]

# алгоритм Брезенхема
def draw_line(x1=0, y1=0, x2=0, y2=0):
    dx = x2 - x1
    dy = y2 - y1

    sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
    sign_y = 1 if dy > 0 else -1 if dy < 0 else 0

    if dx < 0:
        dx = -dx
    if dy < 0:
        dy = -dy

    if dx > dy:
        pdx, pdy = sign_x, 0
        es, el = dy, dx
    else:
        pdx, pdy = 0, sign_y
        es, el = dx, dy

    x, y = x1, y1

    error, t = el/2, 0
    vp[x][y] = 0

    while t < el:
        error -= es
        if error < 0:
            error += el
            x += sign_x
            y += sign_y
        else:
            x += pdx
            y += pdy
        t += 1
        vp[x][y] = 0


x = getRand(x_left_bound, x_right_bound)
y = getRand(y_left_bound, y_right_bound)
coords = [[x, y]]

# инициализация координат
for i in range(1, 6):
    x = getRand(x_left_bound + i * 10, i * add)
    y = getRand(y_left_bound, y_right_bound)
    coords.append([x, y])

# рисование линий
for i in range(5):
    x1 = coords[i][0]
    y1 = coords[i][1]
    x2 = coords[i+1][0]
    y2 = coords[i+1][1]
    draw_line(x1, y1, x2, y2)
draw_line(coords[0][0], coords[0][1], coords[5][0], coords[5][1])

# вывод матрицы пикселей
plt.matshow(vp)


# %%
