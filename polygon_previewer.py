import cv2
import numpy as np

# Para que la ventana de OpenCV sea escalable en Windows
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1)

# Dimensiones del canvas (1920x1080, 16:9)
width, height = 1920, 1080

# Crear una imagen negra (canvas)
image = np.zeros((height, width, 3), dtype=np.uint8)

# Calcular el punto medio
mid_x, mid_y = width // 2, height // 2

# Dibujar las 4 zonas sin relleno (sólo bordes con thickness=1)
cv2.rectangle(image, (0, 0), (mid_x, mid_y), (0, 255, 0), thickness=1)         # Superior izquierda
cv2.rectangle(image, (mid_x, 0), (width, mid_y), (0, 255, 0), thickness=1)         # Superior derecha
cv2.rectangle(image, (0, mid_y), (mid_x, height), (0, 255, 0), thickness=1)        # Inferior izquierda
cv2.rectangle(image, (mid_x, mid_y), (width, height), (0, 255, 0), thickness=1)     # Inferior derecha

# Dibujar el borde exterior del canvas en azul para confirmar límites
cv2.rectangle(image, (0, 0), (width - 1, height - 1), (255, 0, 0), thickness=1)

cv2.imshow('Canvas 1920x1080 with 4 zones', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
