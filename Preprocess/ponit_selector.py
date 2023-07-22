import cv2

class PointSelector:

    # Inicializar la clase con la imagen y la ventana
    def __init__(self, img, window):
        self.img = img # Imagen original
        self.window = window # Nombre de la ventana
        self.points = [] # Lista de puntos seleccionados
        cv2.namedWindow(self.window) # Crear la ventana
        cv2.setMouseCallback(self.window, self.mouse_click) # Asignar la función del ratón

    # Definir una función para manejar el evento del ratón
    def mouse_click(self, event, x, y, flags, param):
        # Si se hace clic izquierdo, guardar el punto y dibujar un círculo
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.img, (x, y), 5, (0, 0, 255), -1)
            # Mostrar la imagen actualizada
            cv2.imshow(self.window, self.img)

    # Definir una función para mostrar la imagen y esperar a que se presione la tecla enter para salir
    def show_and_wait(self):
        cv2.imshow(self.window, self.img) # Mostrar la imagen original
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13: # Tecla enter
                break
        cv2.destroyAllWindows() # Liberar recursos y cerrar ventanas
