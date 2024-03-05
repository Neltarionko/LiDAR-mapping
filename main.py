import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from rdp import rdp

df = pd.read_csv("examp2.txt", sep=";", names=['coordinate', 'distance'])

coordinates = []
distances = []
steps = 681  # количество шагов лидара
angles = np.arange(2*math.pi/3, -2*math.pi/3 + 4/3*math.pi/681, -4/3*math.pi/681)
self_detect = 0.3  # параметр, который отсекает слишком близкие точки (предположительно части самого робота)
precision = 2 # Коэффициент округления значениц
size = 8
half_size = size // 2
starting_point = [[70], [90], [1]]
end_point = [[140], [160], [1]]

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
            walls_x.append(robot_x[i] + (distances[i][j] * math.cos(robot_angle[i] + angles[j])))
            walls_y.append(robot_y[i] + (distances[i][j] * math.sin(robot_angle[i] + angles[j])))
        else:
            continue


MIN_X, MIN_Y = abs(min(walls_x)), abs(min(walls_y))

for i in range(len(walls_x)):  # Двигаю всю координатную сетку из отрицательной части
    walls_x[i] += MIN_X
    walls_x[i] = round(walls_x[i], precision)/(precision * 2)
    walls_y[i] += MIN_Y
    walls_y[i] = round(walls_y[i], precision)/(precision * 2)
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
my_map = cv2.convertScaleAbs(my_map, alpha=1, beta=0)

with open("map.txt", 'w') as f:
    for i in range(len(my_map)):
        for j in range(len(my_map[i]) - 1):
            f.write(f"{str(my_map[i][j])}, ")
        f.write(f"{str(my_map[i][j])}\n")
print(MAP_SIZE)
changed_x, changed_y = 0, 0
for i in range(len(my_map)):
    if not changed_y:
        for j in range(len(my_map[i])):
            if my_map[i][j] and not changed_x:
                for k in range(-half_size, half_size, 1):
                    if i+k >= len(my_map):
                        continue
                    else:
                        my_map[i+k][j-half_size:j+half_size] = 1
                changed_x = half_size
                changed_y = half_size
            elif changed_x:
                changed_x -= 1
            else:
                continue
    elif changed_y:
        changed_y -= 1

def add_edges(my_map):
    '''Ищет углы у объектов на карте
    '''
    for i in range(len(my_map)):
        for j in range(len(my_map[i])):
            if my_map[i, j] == 1 and np.any(my_map[i-1:i+1, j-1]) == False and np.any(my_map[i-1, j-1:j+1]) == False:
                my_map[i, j] = 2
            elif my_map[i, j] == 1 and np.any(my_map[i-1:i+1, j+1]) == False and np.any(my_map[i-1, j-1:j+1]) == False:
                my_map[i, j] = 2
            elif my_map[i, j] == 1 and np.any(my_map[i-1:i+1, j+1]) == False and np.any(my_map[i+1, j-1:j+1]) == False:
                my_map[i, j] = 2
            elif my_map[i, j] == 1 and np.any(my_map[i-1:i+1, j-1]) == False and np.any(my_map[i+1, j-1:j+1]) == False:
                my_map[i, j] = 2

def vision_graph(my_map, print=False):
    '''Создаёт и добавляет на карту углы объектов и граф видимости

    Parametres:
    my_map: карта
    print: флаг отрисовки линий на карте
    '''
    add_edges(my_map)
    edges_list = np.where(np.array(my_map)==2)
    start = []
    end = []

    edges_list = np.append(edges_list, np.zeros((1, len(edges_list[0]))), axis=0)
    edges_list = np.append(edges_list, end_point, axis=1)
    edges_list = np.insert(edges_list, [0], starting_point, axis=1)
    edges_list = np.array(edges_list, dtype=int)

    for i in range(len(edges_list[0])):
        if edges_list[2][i] == 1:
            for j in range(len(edges_list[0])):
                d_y = abs(edges_list[0][j] - edges_list[0][i])
                d_x = abs(edges_list[1][j] - edges_list[1][i])
                y = np.linspace(edges_list[0][i], edges_list[0][j], max(d_y, d_x), dtype=int)
                x = np.linspace(edges_list[1][i], edges_list[1][j], max(d_y, d_x), dtype=int)
                for k in range(len(x)):
                    if my_map[y[k]][x[k]] == 1:
                        break
                    elif k == len(x) - 1:
                        start.append(edges_list[0][i])
                        start.append(edges_list[1][i])
                        end.append(edges_list[0][j])
                        end.append(edges_list[1][j])
                        edges_list[2][j] = 1
    if print:
        X = []
        Y = []

        for i in range(0, len(start) - 1, 2):
            X.append(start[i+1])
            Y.append(start[i])
            X.append(end[i+1])
            Y.append(end[i])
            plt.plot(X, Y, 'ob--')
            X.clear()
            Y.clear()

# vision_graph(my_map)
plt.imshow(my_map)
plt.plot(starting_point[1], starting_point[0], 'or')
plt.plot(end_point[1], end_point[0], 'or')
plt.show()

'''
TODO
Сделать карту по координатам точек стен
Фильтр Гаусса
Аппроксимировать прямые при помощи РДП
'''