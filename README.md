# Tfg_IA

This is a project where we are going to prove different methods to process different Hyperespectral images

# Data Preparation

This Python code allows you to load and analyze hypercube data.

# Preprocess

Here are the methods used to reduce the noise in the data provided it is in a different folder because it is not necessary for you to use it in order to prove different types of preproccess.

# Classifiers

In this file you will find different types of neural networks which are used to classify the different types of materials. 

you will find this methods:

## CNN

## Multilayer Perceptron



# Usage

## Prerequisites

Before running the code, ensure that you have the following dependencies installed:

- Python 3.x
- NumPy
- Matplotlib
- Tensorflow

## Installation

1. Clone the repository or download the `hypercube_analysis.py` file.
2. Install the required dependencies using pip:
   ```
   pip install numpy matplotlib
   ```

## Usage

1. Import the necessary modules and classes:
   ```python
   import os
   import sys
   import matplotlib
   from matplotlib import pyplot as plt
   import numpy as np
   ```

2. Create an instance of the `Hypercube` class:
   ```python
   hipercubo = Hypercube()
   ```

3. Load a hypercube file using the `Load()` method:
   ```python
   hipercubo.Load("path/to/hypercube.bin")
   ```

4. Perform analysis and visualization on the loaded hypercube. Here are some example methods you can use:

   - `PlotDimLine(dim_a_visualizar)`: Plots the values along a specific dimension as a line graph.
   - `PlotDimHist(dim_a_visualizar)`: Plots the values along a specific dimension as a histogram.
   - `PlotIMG(dim_a_visualizar)`: Displays the image of a given dimension.

   Example usage:
   ```python
   hipercubo.PlotDimLine(150)
   hipercubo.PlotIMG(150)
   ```

5. Run the script with the hypercube file as a command-line argument:
   ```
   python hypercube_analysis.py path/to/hypercube.bin
   ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgments

- This code was developed as part of a hypercube analysis project.
- Thanks to the contributors of NumPy and Matplotlib for their fantastic libraries.