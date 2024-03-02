import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

df = pd.read_csv("examp2.txt", sep=";", names=['coordinate', 'distance'])

coordinates = []
distances = []
steps = 681  # количество шагов лидара
angles = np.arange(-2*math.pi/3, 2*math.pi/3 + 4/3*math.pi/681, 4/3*math.pi/681)
self_detect = 0.3  # параметр, который отсекает слишком близкие точки (предположительно части самого робота)
precision = 2 # Коэффициент округления значениц

# Тут проиходит разбивка данных лидара на удобные для использования массивы
for i in range(len(df)):
    coordinates.append(df['coordinate'][i].split(', '))
    distances.append(df['distance'][i].split(', '))

coordinates = np.array(coordinates, dtype=float)
distances = np.array(distances, dtype=float)
robot_x, robot_y = coordinates[0:, 0], coordinates[0:, 1]
robot_angle = coordinates[0:, 2]
walls_x, walls_y = [], []


for i in range(len(df)):
    for j in range(steps):
        if distances[i][j] != 5.6 and distances[i][j] > self_detect:  # Перевод из полярной системы координат в декартову
            walls_x.append(robot_x[i] + (distances[i][j] * math.cos(robot_angle[i] - angles[j])))
            walls_y.append(robot_y[i] + (distances[i][j] * math.sin(robot_angle[i] - angles[j])))
        else:
            continue


MIN_X, MIN_Y = abs(min(walls_x)), abs(min(walls_y))

for i in range(len(walls_x)):  # Двигаю всю координатную сетку из отрицательной части
    walls_x[i] += MIN_X
    walls_x[i] = round(walls_x[i], precision)/2
    walls_y[i] += MIN_Y
    walls_y[i] = round(walls_y[i], precision)/2
    if i < 100:
        robot_x[i] += MIN_X
        robot_y[i] += MIN_Y

# plt.plot(walls_x, walls_y, "ro", ms=0.25)
# plt.plot(robot_x, robot_y)
# plt.show()

MAP_SIZE = (pow(10, precision) * math.ceil(max(walls_x) + abs(min(walls_x))),\
             pow(10, precision) * math.ceil(max(walls_y) + abs(min(walls_y))))
raw_map = np.zeros(MAP_SIZE)

for i in range(len(walls_x)):
    raw_map[int(pow(10, precision) * walls_x[i])][int(pow(10, precision) * walls_y[i])] = 1

my_map = cv2.GaussianBlur(raw_map, (7, 7), 0)

# for i in my_map:
#     for j in i:
#         j = round(j)

my_map = cv2.convertScaleAbs(my_map, alpha=1, beta=0)

plt.imshow(my_map, cmap='gray')
plt.show()

'''
TODO
Сделать карту по координатам точек стен
Фильтр Гаусса
Аппроксимировать прямые при помощи РДП
'''