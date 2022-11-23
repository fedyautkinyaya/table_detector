
import numpy as np
import cv2 as cv
import math
from pdf2image import convert_from_path

# cv.namedWindow("result")  # создаем главное окно
# cv.namedWindow("settings")  # создаем окно настроек

# перевод pdf в jpg
pages = convert_from_path("/home/fedyautkin/PycharmProjects/pythonProject4py379/pose_estimation/1/task2/1.pdf")
for i, page in enumerate(pages):
    page.save(f"/home/fedyautkin/PycharmProjects/pythonProject4py379/pose_estimation/1/task2/img_{i}.jpg", 'JPEG')


fn = '/home/fedyautkin/PycharmProjects/pythonProject4py379/pose_estimation/1/task2/img_0.jpg'  # имя файла, который будем анализировать
fn1 = '/home/fedyautkin/PycharmProjects/pythonProject4py379/pose_estimation/1/task2/img_0_out.jpg' # имя файла, который т на выходе
img = cv.imread(fn)
hsv = cv.cvtColor(img, cv.COLOR_BGR2RGB)
(h, w) = img.shape[:2]

# # считываем значения бегунков
# h1 = cv.getTrackbarPos('h1', 'settings')
# s1 = cv.getTrackbarPos('s1', 'settings')
# v1 = cv.getTrackbarPos('v1', 'settings')
# h2 = cv.getTrackbarPos('h2', 'settings')
# s2 = cv.getTrackbarPos('s2', 'settings')
# v2 = cv.getTrackbarPos('v2', 'settings')
    # формируем начальный и конечный цвет фильтра
h1 = 0
s1 = 0
h2 = 255
s2 = 255
v2 = 255
v1 = 137
hsv_min = np.array((h1, s1, v1), np.uint8)
hsv_max = np.array((h2, s2, v2), np.uint8)
spisok = []


thresh = cv.inRange(hsv, hsv_min, hsv_max)  # применяем цветовой фильтр
contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # перебираем все найденные контуры в цикле
for cnt in contours0:
    rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
    box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
    box = np.int0(box)  # округление координат
    center = (int(rect[0][0]), int(rect[0][1]))
    center_color = center
    area = int(rect[1][0] * rect[1][1])  # вычисление площади

    # вычисление координат двух векторов, являющихся сторонам прямоугольника
    edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
    edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))

    # выясняем какой вектор больше
    usedEdge = edge1
    if cv.norm(edge2) > cv.norm(edge1):
        usedEdge = edge2
    reference = (1, 0)  # горизонтальный вектор, задающий горизонт

    # вычисляем угол между самой длинной стороной прямоугольника и горизонтом
    angle = 180.0 / math.pi * math.acos(
        (reference[0] * usedEdge[0] + reference[1] * usedEdge[1]) / (cv.norm(reference) * cv.norm(usedEdge)))

    if area > 40000:
        cv.drawContours(thresh, [box], 0, (255, 0, 0), 2)  # рисуем прямоугольник
        cv.circle(thresh, center, 5, (0, 0, 0), 2)  # рисуем маленький кружок в центре прямоугольника
        # выводим в кадр величину угла наклона
        cv.putText(thresh, "%d" % int(angle), (center[0] + 20, center[1] - 20),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.drawContours(img, [box], 0, (255, 255, 0), 2)  # рисуем прямоугольник
        cv.circle(img, center, 5, (0, 0, 0), 2)  # рисуем маленький кружок в центре прямоугольника
        spisok.append(angle)
        # выводим в кадр величину угла наклона
        cv.putText(img, "%d" % int(angle), (center[0] + 20, center[1] - 20),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 0), 2)

spisok1 = sorted(spisok)
print(spisok1)
center1 = (h / 2, w / 2)
image_center = tuple(np.array(img.shape[1::-1]) / 2)
rotation_matrix = cv.getRotationMatrix2D(center1, int(spisok1[-2]), 1)
rotated = cv.warpAffine(img, rotation_matrix, (w, h))
cv.imwrite(fn1, rotated)

# cv.imshow('contours', thresh)
# cv.imshow('result', img)

cv.waitKey()
cv.destroyAllWindows()

