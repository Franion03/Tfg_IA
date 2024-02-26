import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

def mean_centering(hipercubo):
    """
    Mean centering is a method to remove the mean from the data. 
    This is done by subtracting the mean of each band from the data. 
    Mean centering is useful to remove the effects of different illumination conditions 
    and to highlight the spectral differences between different pixels.
    """
    # Calcular la media de cada banda espectral
    media = np.mean(hipercubo, axis=(0, 1))

    # Restar la media a los datos originales
    mean_centered_data = hipercubo - media

    # Devolver los datos centrados
    return mean_centered_data

def pca(hipercubo,varianza = None, componentes=10):
    """
    Principal Component Analysis (PCA) is a method to reduce the dimensionality of data. 
    It does so by finding the principal components of the data, which are the directions 
    along which the data varies the most. These principal components can then be used to 
    transform the data into a new coordinate system, where the first coordinate (first principal 
    component) captures the most variation, the second coordinate (second principal component) 
    captures the second most variation, and so on. This new coordinate system can be used to 
    visualize the data in a lower dimension, or to perform other tasks such as clustering, classification, etc.

    We have to specify the number of principal components to keep, which is a hyperparameter of the PCA algorithm.
    """
    dataImagen = hipercubo.hipercubo.copy()
    if varianza != None :
        imageTemp = dataImagen.reshape((dataImagen.shape[1],dataImagen.shape[0]*dataImagen.shape[2])).T
        pca = PCA()
        pca.fit(imageTemp)
        imageTemp = pca.transform(imageTemp)
        #Evaluar el numero de coeficientes en base a los datos de varianza
        var = 0
        num_componentes = 0
        for i in range(pca.explained_variance_ratio_.shape[0]):
            var += pca.explained_variance_ratio_[i]
            if var > varianza:
                break
            else:
                num_componentes += 1
        imageTemp = imageTemp.reshape( (dataImagen.shape[1], dataImagen.shape[2],dataImagen.shape[0]) )
        imagePCA = np.zeros( (num_componentes, dataImagen.shape[1], dataImagen.shape[2]) )
        for i in range(imagePCA.shape[0]):
            imagePCA[i] = imageTemp[:,:,i]
    if componentes != None:
        imageTemp = dataImagen.reshape((dataImagen.shape[0],dataImagen.shape[1]*dataImagen.shape[2])).T
        c_pca = PCA(n_components=componentes)
        c_pca.fit(imageTemp)
        imageTemp = c_pca.transform(imageTemp)
        imageTemp = imageTemp.reshape( (dataImagen.shape[1], dataImagen.shape[2],imageTemp.shape[1]) )
        imagePCA = np.zeros( (componentes, dataImagen.shape[1], dataImagen.shape[2]) )
        for i in range(imagePCA.shape[0]):
            imagePCA[i] = imageTemp[:,:,i]
    return imagePCA

def standardNormalizationVariate(hipercubo):
    """
    Standard normalization is a method to scale the data so that each band has zero mean and unit variance. 
    This is done by subtracting the mean of each band from the data, and then dividing by the standard deviation of each band. 
    Standard normalization is useful to remove the effects of different illumination conditions and to highlight the spectral differences between different pixels.
    """
    normalized_datas = []
    # Calcular la media de cada banda espectral
    for i in range(hipercubo.numBands):
        media = np.mean(hipercubo.hipercubo[:, i, :])
    # Calcular la desviación estándar de cada banda espectral
        desviacion_estandar = np.std(hipercubo.hipercubo[:, i, :])
    # Normalizar los datos
        normalized_data = (hipercubo.hipercubo[:, i, :] - media) / desviacion_estandar
        normalized_datas.append(normalized_data)

    # Devolver los datos normalizados
    return normalized_datas

def filterSavinzkyGolay ( hipercubo):
    """
    The Savitzky-Golay filter is a method to smooth data. 
    It does so by fitting a low-degree polynomial to the data, and then using this polynomial to estimate the smoothed values. 
    The Savitzky-Golay filter is useful to remove noise from data, while preserving the shape and features of the signal.
    """
    return savgol_filter(hipercubo, 5, 2)

def minmaxnormalization(hipercubo):
    """
    Min-max normalization is a method to scale the data to a specific range, typically [0, 1]. 
    This is done by subtracting the minimum value of each band from the data, and then dividing by the range of each band (i.e., the maximum value minus the minimum value). 
    Min-max normalization is useful to remove the effects of different illumination conditions and to highlight the spectral differences between different pixels.
    """
    minmax_normalized_datas = []
    for i in range(hipercubo.numBands):
    # Calcular el mínimo de cada banda espectral
        minimos = np.min(hipercubo.hipercubo[:, i, :])

    # Calcular el máximo de cada banda espectral
        maximos = np.max(hipercubo.hipercubo[:, i, :])

    # Normalizar los datos
        minmax_normalized_data = (hipercubo.hipercubo[:, i, :] - minimos) / (maximos - minimos)
        minmax_normalized_datas.append(minmax_normalized_data)
    # Devolver los datos normalizados
    return minmax_normalized_datas

def msc(hypercube):
    """
    Multiplicative scatter correction (MSC) is a method to remove the effects of light scattering from the data. 
    It does so by dividing each spectrum by a reference spectrum, which is typically the mean spectrum of the data. 
    MSC is useful to remove the effects of different illumination conditions and to highlight the spectral differences between different pixels.
    """
    normalized_hypercubes = []
    for i in range(hypercube.numBands):
        # Calcular la media de cada banda espectral
        mean = np.mean(hypercube.hypercube[:, i, :])
        # Normalizar cada banda dividiéndola por la banda media
        normalized_hypercube = hypercube.hypercube[:, i, :] / mean[np.newaxis, :, np.newaxis]
        normalized_hypercubes.append(normalized_hypercube)


    # Devolver el hipercubo normalizado
    return normalized_hypercubes

def emsc(hypercube):
    """
    Extended multiplicative scatter correction (EMSC) is a method to remove the effects of light scattering from the data. 
    It does so by dividing each spectrum by a reference spectrum, which is typically the mean spectrum of the data. 
    EMSC is useful to remove the effects of different illumination conditions and to highlight the spectral differences between different pixels.
    """
    normalized_hypercubes = []
    for i in range(hypercube.numBands):
        # Calcular la media de cada banda espectral
        mean = np.mean(hypercube.hypercube[:, i, :])
        # Calcular la desviación estándar de cada banda espectral
        std = np.std(hypercube.hypercube[:, i, :])
        # Normalizar cada banda dividiéndola por la banda media y la desviación estándar
        normalized_hypercube = (hypercube.hypercube[:, i, :] - mean[np.newaxis, :, np.newaxis]) / std[np.newaxis, :, np.newaxis]
        normalized_hypercubes.append(normalized_hypercube)

    # Devolver el hipercubo normalizado
    return normalized_hypercubes

def assymetricLeastSquares(hypercubo):
    """
    Assymetric Least Squares (ALS) is a method to remove the effects of light scattering from the data. 
    It does so by dividing each spectrum by a reference spectrum, which is typically the mean spectrum of the data. 
    ALS is useful to remove the effects of different illumination conditions and to highlight the spectral differences between different pixels.
    """
    normalized_hypercubes = []
    for i in range(hypercubo.numBands):
        # Calcular la media de cada banda espectral
        mean = np.mean(hypercubo.hypercube[:, i, :])
        # Calcular la desviación estándar de cada banda espectral
        std = np.std(hypercubo.hypercube[:, i, :])
        # Normalizar cada banda dividiéndola por la banda media y la desviación estándar
        normalized_hypercube = (hypercubo.hypercube[:, i, :] - mean[np.newaxis, :, np.newaxis]) / std[np.newaxis, :, np.newaxis]
        normalized_hypercubes.append(normalized_hypercube)

    # Devolver el hipercubo normalizado
    return normalized_hypercubes