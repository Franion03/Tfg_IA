import numpy as np
import time
from sklearn import preprocessing
from scipy.signal import savgol_filter
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import sys
import glob
import os


#Recibe un spectrum y lo reescala al rango [0..1]
def SpectrumPreprocessor(spectrum):
    """
     Preprocesses a spectrum to make it easier to visualize. Savitzky - Golay filters are applied to the spectrum before scaling.
     
     Args:
     	 spectrum: The spectrum to preprocess. Must be a list of length 1.
     
     Returns: 
     	 A list of length 1 containing the preprocessed spectrum. It is assumed that the spectrum has been filtered
    """
    spectrum=spectrum[0]
    spectrum=savgol_filter(spectrum, 15,2)
    spectrum=preprocessing.minmax_scale(spectrum)
    return np.array([spectrum])


#Dado un FileDescriptor abierto lee batch_size lineas
#  Para cada linea elimina el \n final y obtiene los valores con split por ','
def ReadBatchFromFile(openFileDescriptor,batch_size,separator):
    """
     Reads a batch of data from a file. This is a generator function that will be called by the Read () function of the data source.
     
     Args:
     	 openFileDescriptor: An open file descriptor to the file to read
     	 batch_size: The number of samples to read
     	 separator: The separatin between samples and labels e. g.
     
     Returns: 
     	 A tenga definidos y yBatch : Numpy arrays containg the xBatch y
    """
    #Definimos aquí por si el fichero esta vacio que el return tenga definidos los valores de retorno.
    xBatch=np.array([])
    yBatch=np.array([])
    # El el linea de la linea.
    for l in range(batch_size):
        line= openFileDescriptor.readline()
        if not line:
            break
        else:
            #Se elimina el último caracter el \n leido en cada linea
            line = line[:-1]
            
            # Se separa el string en datos. Separador la coma.
            lineData=[float(v) for v in line.split(separator)]

            if l==0: #Si hay al menos una linea se definen aquí los arrays de retorno

                #El numero de muestras es el numero de datos de la linea menos el último que es el label
                numSamples=len(lineData)-1 # el ultimo valor es el label 0 o 1

                #Definimos los arrays. En la primera iteración que es cuando sabemos que longitud tienen
                xBatch = np.empty((1,numSamples),dtype=float)
                yBatch = np.empty((1,1), dtype=float)
            
            spectrum=np.array([lineData[:-1]], dtype=float)
            label=np.array([lineData[-1:]], dtype=float)
       
            #spectrum=SpectrumPreprocessor(spectrum)
            #spectrum=NormalizeSpectrum(spectrum)

            xBatch = np.append(xBatch, spectrum, axis=0)
            yBatch = np.append(yBatch, label, axis=0)

            if l==0: #Como hacemos append sobre un np que tiene un primer elemento vacio, ahora lo quitamos
                xBatch = np.delete(xBatch, 0, axis=0)            
                yBatch = np.delete(yBatch, 0, axis=0)            

    return xBatch, yBatch


def BatchGenerator(fileList, batch_size, separator, validation):
    """
     Generator for batch files. Iterates over the list of files and yields batches and labels in a generator
     
     Args:
     	 fileList: List of files to process
     	 batch_size: Size of each batch in bytes ( default 10 )
     	 separator: Separator between files ( default'' ) e. g.
     	 validation: True if validation is enabled False if not ( default
    """
    ixFile=0
    numFiles=len(fileList)
    #print(fileList[0])
    print("Files to process: {}".format(numFiles))
    print(f"Separator {separator}")
    while True:
        for file in fileList:
            # print("Reading from {}".format(file))
            fileDesc = open(file, 'r')
            xBatch, yBatch = ReadBatchFromFile(fileDesc,batch_size,separator)
            numLines=xBatch.shape[0]
            # Caso de que el primer batch venga vacio (no hay datos en el fichero) saltamos al siguiente fichero.
            if numLines==0:
                continue
            # First Batch-------------------------------------------
            le = LabelEncoder()
            label = le.fit_transform(yBatch)
            np.array(label)
            #np.ravel(label)

# printing label
            label
            yield xBatch, label

            # Iterating through lines in file
            while numLines>0:
                #print("Next Batch")
                xBatch, yBatch = ReadBatchFromFile(fileDesc,batch_size,separator)
                #print(batch.shape)
                numLines=xBatch.shape[0]
                #print(f"  numLines = {numLines}")
                # Batch-------------------------------------------
                if (numLines==0):
                    break

                #Returning batch. Last batch could have less than batch_size lines.
                le = LabelEncoder()
                label = le.fit_transform(yBatch)
                np.array(label)
                #np.ravel(label)
                yield xBatch, label
            fileDesc.close()
        if validation:
            break;

def ValidationGenerator(fileList, batch_size, separator):
    """
     Generator for validation. This is a generator that yields batches of data and labels from a list of files
     
     Args:
     	 fileList: List of files to process
     	 batch_size: Batch size in rows and columns ( int )
     	 separator: Separator between file names ( str ) e. g
    """
    # print(fileList[0])
    # print("Files to process: {}".format(numFiles))
    for file in fileList:
        # print("Reading from {}".format(file))
        fileDesc = open(file, 'r')
        xBatch, yBatch = ReadBatchFromFile(fileDesc,batch_size, separator)
        numLines=xBatch.shape[0]
        # Caso de que el primer batch venga vacio (no hay datos en el fichero) saltamos al siguiente fichero.
        if numLines==0:
            continue
        # First Batch-------------------------------------------
        #to_categorical(yBatch, num_classes=4)
        le = LabelEncoder()
        label = le.fit_transform(yBatch)
        np.array(label)
        yield xBatch, label

        # Iterating through lines in file
        while numLines>0:
            #print("Next Batch")
            xBatch, yBatch = ReadBatchFromFile(fileDesc,batch_size,separator)
            #print(batch.shape)
            numLines=xBatch.shape[0]
            #print(f"  numLines = {numLines}")
            # Batch-------------------------------------------
            if (numLines==0):
                break

            #Returning batch. Last batch could have less than batch_size lines.
            le = LabelEncoder()
            label = le.fit_transform(yBatch)
            np.array(label)
            yield xBatch, label
        fileDesc.close()


#FUNCion para calcular el numero de lineas de un CSV
def FileNumLines (filename):
    """
     Count the number of newlines in a file. This is useful for checking the size of a file in order to avoid reading the whole file multiple times.
     
     Args:
     	 filename: name of file to check. Must be a string
     
     Returns: 
     	 number of newlines in
    """
    chunk = 1024*1024   # Process 1 MB at a time.
    f = np.memmap(filename)
    num_newlines = sum(np.sum(f[i:i+chunk] == ord('\n'))
                       for i in range(0, len(f), chunk))
    del f
    return num_newlines


def FilesNumLines(fileList):
    """
     Get the number of lines in a list of files. This is useful for debugging and to determine how many lines are in each file
     
     Args:
     	 fileList: list of files to count
     
     Returns: 
     	 int of total number of lines in each file in the list in the same order as they appeared
    """
    totalLines=0
    # tik = time.time()
    for file in fileList:
        numLines=FileNumLines(file)
        totalLines+=numLines
    # tok = time.time()
    # print(tok-tik)
    return totalLines

def SpectrumsCounter(fileList):
    """
     Counts the number of spectra in each file and returns the total number of lines. This is useful for debugging
     
     Args:
     	 fileList: List of files to be analysed
     
     Returns: 
     	 Total number of Spectrums in each file ( 1 or 0 if there are no files in the
    """
    totalLines=0
    # tik=time.time()
    for file in fileList:
        #print(f"Reading from {file}")
        fileDesc = open(file, 'r')
        fileLines=0
        line= fileDesc.readline()
        if not(line):
            continue

        totalLines+=1
        fileLines+=1

        # Iterating through lines in file
        while line:
            line= fileDesc.readline()
            if not(line):
                break

            #Returning batch. Last batch could have less than batch_size lines.
            totalLines+=1
            fileLines+=1
        fileDesc.close()
        #print(f"  #file:{fileLines} #total:{totalLines}")
    # tok=time.time()
    # print(tok-tik)
    return totalLines

if __name__ == '__main__':  
    
    if len(sys.argv) == 3:
        trainFiles= glob.glob(os.path.join(sys.argv[1] , "f*.csv"))
        testFiles= glob.glob(os.path.join(sys.argv[2] , "f*.csv"))
        filelist=trainFiles
        batchsize=1000
        something = BatchGenerator(filelist, batchsize, ';', validation=False)
        x,y=next(something) 
        print("Train files: {}".format(len(trainFiles)))
