import os
import sys
import glob
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2
import json
import csv
from segment_anything_meta import Segmenter


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
        self.bidimensionalHipercube = None
        self.medianHipercube = None 

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
        self.hipercubo = self.hipercubo.reshape((self.numLines,self.numBands,self.numPixels)).astype("uint16")    
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
    # Grafica la imagen de la media de las bandas
    def PlotIMGBidimensional(self):
        matriz_a_visualizar = self.bidimensionalHipercube
        # Mostrar la matriz como una imagen utilizando la función `imshow`
        plt.imshow(matriz_a_visualizar, cmap='gray')
        plt.title('Visualización de la media de las bandas')
        plt.colorbar()
        plt.show()
    # A function that crops the image to the given dimensions
    def Crop(self, y1, y2, x1, x2):
        self.hipercubo = self.hipercubo[:, x1:x2, y1:y2].astype(np.uint16)
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
    # A function that select roi in image
    def selectROI(self):
        img = cv2.cvtColor(self.hipercubo[:, 100, :].astype(np.uint8), cv2.COLOR_RGB2BGR)

        roi = cv2.selectROI("Selecciona el area", img)
        self.hipercubo = self.hipercubo[roi[1]:roi[1]+roi[3], :, roi[0]:roi[0]+roi[2]].astype(np.uint16)
        return roi
    # A function that makes the average of bands in the hipercube
    def average(self):
        self.bidimensionalHipercube = np.mean(self.hipercubo, axis=1,dtype=np.uint16)
    # A function that makes the average of bands in the hipercube
    def median(self):
        self.medianHipercube = np.median(self.hipercubo, axis=1)
    # A function that keeps in csv roi values giveds from the hipercube in actual path
    def saveROI(self, roi):
        np.savetxt(self.basename + ".csv", roi, delimiter=",")
    
    def returnImage(self):
        return cv2.cvtColor(self.hipercubo[:, 100, :].astype(np.uint8), cv2.COLOR_RGB2BGR)
    # una función que te extraiga los valores de la imagen en una mascara dada
    def extractValues(self, mask): 
        returnedmask = []
        maskLabeled = np.zeros_like(mask)
        for i in range(len(mask)):
            for j in range(len(mask[i])):
                if mask[i][j] == True:
                    returnedmask.append(self.hipercubo[i, :, j])
                    maskLabeled[i][j] = 1 # TODO: Aqui debe de salir la etiqueta de la imagen deseada
                else:
                    maskLabeled[i][j] = 0
        return returnedmask,maskLabeled
    # A function that saves in csv the values of the hipercube and assigns a value to the hole image
    # TODO: Añadir filtro de mediana para filtrar los valores mas bajos
    def saveValues(self, values, name):
        array = np.array(values)
        with open(name + '.csv', 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';')
            for  row in array:
                # write the label and the row as a list
                spamwriter.writerow(  list(row) + [3])
    # A function that saves the mask in .bin format
    def saveMask(self, mask,filename):
        savedMask = np.array(mask).astype(np.uint8)
        filename = filename.replace(".bin","")
        np.savetxt(filename + "_mask" + ".bin", savedMask, delimiter=",")



if __name__ == '__main__':  
    
    matplotlib.use('TkAgg',force=True) 
    if len(sys.argv) == 2:
        all_files = glob.glob(os.path.join(sys.argv[1] , "*.bin"))
        maxValue = 0
        minValue = 1000
        # for filename in all_files:
        #     hipercubo = Hypercube()
        #     hipercubo.Load(filename)
        #     hipercubo.median()
        #     if maxValue < np.max(hipercubo.medianHipercube):
        #         maxValue = np.max(hipercubo.medianHipercube)
        #     if minValue > np.min(hipercubo.medianHipercube): 
        #         minValue = np.min(hipercubo.medianHipercube)

        for filename in all_files:
            hipercubo = Hypercube()
            hipercubo.Load(filename)
            hipercubo.Crop(94,528,10,214)
            # hipercubo.PlotIMG(100)
            hipercubo.average()
            # hipercubo.PlotDimLine(100)
            # hipercubo.PlotIMG(100)
            # hipercubo.PlotIMGBidimensional()
            # hipercubo.selectROI()
            # hipercubo.PlotIMG(100)
            segment = Segmenter("vit_h","sam_vit_h_4b8939.pth")
            image = hipercubo.returnImage()
            mask = segment.segmentRoi(image)
            finalmask,maskLabeled = hipercubo.extractValues(mask)
            hipercubo.saveValues(finalmask, "pp_pe")
            hipercubo.saveMask(maskLabeled,filename)
            print("Done")
            # hipercubo.imadjust(94,528,10,214)

