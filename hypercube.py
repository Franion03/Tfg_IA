import os
import struct
import sys
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
import numpy as np
import struct
import array

class Load:
    def __init__(self, filename):
        
        # Get the base name of the file and its extension
        name, ext = os.path.splitext(filename)
        
        # If no extension is provided, assume it's a .mat file
        if not ext:
            ext = '.mat'
        
        # If a path is provided, load the file from that path
        # Otherwise, load it from the dataPath
        if os.path.dirname(filename):
            pathstr = os.path.dirname(filename)
        else:
            pathstr = filename
        
        # Define the file names with the appropriate path
        # .filename = os.path.join(pathstr, name + '.bin')
        # .matFilename = os.path.join(pathstr, name + '.mat')
        
        # If the .bin file doesn't exist, try loading the .mat file
        # if not os.path.exists(.filename):
        #     ext = '.mat'
        
        # if ext == '.bin':
        #     # Clear previous data
        # hc = []
        #     .sghc = []
        #     .badPixelsProcessed = False
            
        #     # When loading a .bin file, we assume 224 bands and 640 pixels.
        numBands = 224
        numPixels = 640
        #     .badPixels = [] # We don't have bad pixels when loading a .bin file
            
        #     print(f'Loading {.filename}')
        
        # Leer el archivo binario en una matriz hipercubo
        hipercubo = np.fromfile(filename, dtype=np.uint16)

        # Darle forma al hipercubo
        # hipercubo = hipercubo.reshape(forma)

        # with open(filename, 'rb') as f:
        #     num_bytes = os.path.getsize(filename)

        #     # Calculate the number of 16-bit integers in the file
        #     num_ints = num_bytes // 2

        #     # Read in the data using struct
        #     data = f.read(num_bytes)
        #     bindata = array.array('H', data)
        
        # # with open(filename, 'rb') as f:
        # #     hipercubo = np.fromfile(f, dtype=np.uint16)
        
        # # Convert the binary data to a list of integers
        # bindata = struct.unpack(f'>{len(data)//2}H', data)
        
        # hc = list(bindata)
        
        # # Calculate the number of lines
        # numLines = len(hc) // (numPixels * numBands)
        #  # Convert the binary data to a 3D array
        # hc = [[[] for _ in range(numBands)] for _ in range(numLines)]
        # for i in range(numLines):
        #     for j in range(numBands):
        #         for k in range(numPixels):
        #             index = i * numBands * numPixels + j * numPixels + k
        #             hc[i][j].append(bindata[index])
        
        # # Convert to numpy array for easier manipulation (optional) //TODO: Cambiar todos los uint16 por uint 16
        # hc = np.array(hc, dtype=np.uint16)

        # # Calculate the number of lines in the data
        # numLines = hc.shape[0]

        # # Transpose the array so that the first two dimensions define a CubeFrame
        # # and the third dimension represents the spectral bands
        # hc = np.transpose(hc, (0, 2, 1))
        numLines = len(hipercubo) // (numPixels * numBands)
        hipercubo = hipercubo.reshape((numLines,numPixels,numBands))
        valores_a_graficar = hipercubo[2,:, :].flatten()

        # Grafica los valores como un histograma
        plt.hist(valores_a_graficar, bins=100)
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.title('Valores en la dimensión {}'.format(2))
        plt.show()

        # Selecciona una dimensión para visualizar y extrae la matriz correspondiente
        dim_a_visualizar = 0  # Seleccione la primera dimensión
        matriz_a_visualizar = hipercubo[dim_a_visualizar, :, :]

        # Mostrar la matriz como una imagen utilizando la función `imshow`
        plt.imshow(matriz_a_visualizar, cmap='gray')
        plt.title('Visualización de la dimensión {}'.format(dim_a_visualizar))
        plt.show()

        print(hipercubo[1][1][1])


        
        

#         if not os.path.exists(.matFilename):
#             # .SaveAsMat(.matFilename)
# #                 print('  Deleting .bin file\n')
# #                 os.remove(.filename)
#             pass
        
#         # If a badpixels file exists for the base name, load it
#         badpixelsFilename = os.path.join(.dataPath, name + '_BadPixels.mat')
#         if os.path.exists(badpixelsFilename):
#             .LoadBadPixels(badpixelsFilename)

if __name__ == '__main__':  
    if len(sys.argv) == 2:
        Load(sys.argv[1])
