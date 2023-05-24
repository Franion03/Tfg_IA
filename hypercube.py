import os
import sys
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


class Hypercube:
    def __init__(self):
        self.dataPath = ""
        self.hipercubo = None
        self.sghc = None
        self.numBands = 224
        self.numPixels = 640
        self.numLines = 0
        self.badPixels = None
        self.badPixelsProcessed = False
        self.basename = ''
        self.filename = ''
        self.ext = ''
        self.bindata = None

    def Load(self, filename):        
        self.dataPath = filename
        self.filename, self.ext = os.path.splitext(filename)
        if not self.ext:
            self.ext = '.mat'
        
        if os.path.dirname(filename):
            self.basename = os.path.dirname(self.dataPath)
        else:
            self.basename = filename
        # Leer el archivo binario en una matriz self.hipercubo
        self.hipercubo = np.fromfile(filename, dtype=np.uint16)
        self.numLines = len(self.hipercubo) // (self.numPixels * self.numBands)
        self.hipercubo = self.hipercubo.reshape((self.numLines,self.numBands,self.numPixels)).astype("float")    
    # Grafica los valores como un grafico lineal
    def PlotDimLine(self,dim_a_visualizar):
        plt.plot(self.hipercubo[dim_a_visualizar,:, dim_a_visualizar])
        plt.show()
    # Grafica los valores como un histograma
    def PlotDimHist(self,dim_a_visualizar):
        valores_a_graficar = self.hipercubo[dim_a_visualizar,:,dim_a_visualizar].flatten()
        plt.hist(valores_a_graficar, bins=1000)
        plt.title('Valores en la dimensión {}'.format(dim_a_visualizar))
        plt.show()
    # Grafica la imagen de la dimension dada
    def PlotIMG(self,dim_a_visualizar):
        matriz_a_visualizar = self.hipercubo[:, dim_a_visualizar, :]
        # Mostrar la matriz como una imagen utilizando la función `imshow`
        plt.imshow(matriz_a_visualizar, cmap='gray')
        plt.title('Visualización de la dimensión {}'.format(dim_a_visualizar))
        plt.colorbar()
        plt.show()
    # A function that crops the image to the given dimensions
    def Crop(self, y1, y2, x1, x2):
        self.hipercubo = self.hipercubo[:, x1:x2, y1:y2]
        self.numPixels = x2 - x1
        self.numLines = y2 - y1
    # A function that adjust color gamma in x image
    def imadjust(self,x,a,b,c,d,gamma=1):
        # Similar to imadjust in MATLAB.
        # Converts an image range from [a,b] to [c,d].
        # The Equation of a line can be used for this transformation:
        #   y=((d-c)/(b-a))*(x-a)+c
        # However, it is better to use a more generalized equation:
        #   y=((x-a)/(b-a))^gamma*(d-c)+c
        # If gamma is equal to 1, then the line equation is used.
        # When gamma is not equal to 1, then the transformation is not linear.

        y = (((self.hipercubo - a) / (b - a)) ** gamma) * (d - c) + c
        return y

if __name__ == '__main__':  
    
    matplotlib.use('TkAgg',force=True)
    if len(sys.argv) == 2:
        hipercubo = Hypercube()
        hipercubo.Crop(94,528,10,214)
        hipercubo.Load(sys.argv[1])
        hipercubo.PlotDimLine(100)
        hipercubo.PlotIMG(100)
        hipercubo.imadjust(94,528,10,214)

