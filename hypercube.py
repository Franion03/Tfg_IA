import os
import struct
import sys

class Load:
    def __init__(self, obj, filename):
        self.obj = obj
        
        # Get the base name of the file and its extension
        name, ext = os.path.splitext(filename)
        self.obj.basename = name
        
        # If no extension is provided, assume it's a .mat file
        if not ext:
            ext = '.mat'
        
        # If a path is provided, load the file from that path
        # Otherwise, load it from the dataPath
        if os.path.dirname(filename):
            pathstr = os.path.dirname(filename)
        else:
            pathstr = obj.dataPath
        
        # Define the file names with the appropriate path
        self.obj.filename = os.path.join(pathstr, name + '.bin')
        self.obj.matFilename = os.path.join(pathstr, name + '.mat')
        
        # If the .bin file doesn't exist, try loading the .mat file
        if not os.path.exists(self.obj.filename):
            ext = '.mat'
        
        if ext == '.bin':
            # Clear previous data
            self.obj.hc = []
            self.obj.sghc = []
            self.obj.badPixelsProcessed = False
            
            # When loading a .bin file, we assume 224 bands and 640 pixels.
            self.obj.numBands = 224
            self.obj.numPixels = 640
            self.obj.badPixels = [] # We don't have bad pixels when loading a .bin file
            
            print(f'Loading {self.obj.filename}')
            # Read the binary data
            with open(self.obj.filename, 'rb') as f:
                data = f.read()
            
            # Convert the binary data to a list of integers
            self.obj.bindata = struct.unpack(f'>{len(data)//2}H', data)
            
            self.obj.hc = list(self.obj.bindata)
            
            # Calculate the number of lines
            self.obj.numLines = len(self.obj.hc) // (self.obj.numPixels * self.obj.numBands)
            
            # Define the cube size as it's loaded from disk
            cubesize = [self.obj.numPixels, self.obj.numBands, self.obj.numLines]
            
            # Fill the cube with the loaded data
            self.obj.hc = [[self.obj.hc[(x*self.obj.numPixels*self.obj.numBands) + (y*self.obj.numPixels) + z] for y in range(self.obj.numBands)] for x in range(self.obj.numLines) for z in range(self.obj.numPixels)]
            
            # Reshape the cube to work more comfortably with it
            # The first two dimensions define a CubeFrame
            # The third dimension (analogous to time) are the spectral bands.
            self.obj.hc = [[self.obj.hc[(y*self.obj.numPixels*self.obj.numLines) + (x*self.obj.numPixels) + z] for y in range(self.obj.numBands)] for x in range(self.obj.numLines) for z in range(self.obj.numPixels)]
            
            if not os.path.exists(self.obj.matFilename):
                # self.obj.SaveAsMat(self.obj.matFilename)
#                 print('  Deleting .bin file\n')
#                 os.remove(self.obj.filename)
                pass
            
            # If a badpixels file exists for the base name, load it
            badpixelsFilename = os.path.join(obj.dataPath, obj.basename + '_BadPixels.mat')
            if os.path.exists(badpixelsFilename):
                self.obj.LoadBadPixels(badpixelsFilename)

if __name__ == '__main__':  
    if len(sys.argv) != 2:
        Load(sys.argc[1])
