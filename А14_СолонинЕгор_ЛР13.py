# %%
import numpy as np
import matplotlib.pyplot as plt

w = 200


def draw_line(x1=0, y1=0, x2=0, y2=0):
    vp = [[255]*w for i in range(w)]

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

    return vp


line = draw_line(0, 0, 150, 100)
plt.matshow(line)


# %%
