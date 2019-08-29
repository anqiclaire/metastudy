from PIL import Image, ImageDraw
import numpy as np

class drawMM(object):
    # by default 80 pixels per unit square
    def __init__(self, array, unitSize = 80, colorZero = (0, 0, 0), colorOne = (0, 0, 0), colorTwo = (192, 192, 192), needDup = False, imageName = 'output.png'):
        if len(array.shape) != 1:
            raise ValueError("Input array should be 1 by n")
        self.imageName = imageName
        self.colorZero = colorZero # gridline
        self.colorOne = colorOne
        self.colorTwo = colorTwo
        self.unitSize = unitSize
        self.unitNum = self._sizeValid(array.shape[0]) # number of unit squares along one direction of 1/4 of the full square
        self.fullNum = 2*self.unitNum # number of unit squares along one direction of the full square
        self.gridWidth = 5 # 5 px = 0.5 mm
        self.margin = 2*self.unitSize # margin at one end
        self._imageGen()
        self.array = array
        self.coords = self.arrayToCoords(array)
        self.needDup = needDup # need duplicate or not
        self.draw()

    def _imageGen(self):
        # generate image based on self.margin
        self.imageSize = self.margin*2 + self.gridWidth*(self.fullNum+1) + self.unitSize*self.fullNum
        self.image = Image.new('RGB', (self.imageSize, self.imageSize), color = 'white')
        self.pixels = self.image.load()

    def _sizeValid(self, size):
        if size == 0:
            raise ValueError("Array size is not acceptable, must be 1,3,6,10,15 ...")
        increment = 0
        sumOfInc = 0
        while sumOfInc < size:
            increment += 1
            sumOfInc += increment
        if sumOfInc == size:
            return increment
        else:
            raise ValueError("Array size is not acceptable, must be 1,3,6,10,15 ...")

    def arrayToCoords(self, array):
        coords = []
        nonZeroIndex = np.flatnonzero(array)
        for index in nonZeroIndex:
            coords.append(self.indexToCoords(index+1))
        return coords
    
    def indexToCoords(self, index):
        sumOfInc = 0
        realIndex = len(self.array) - index
        row = self.unitNum - 1
        col = self.unitNum - 1
        for y in range(1, self.unitNum+1):
            sumOfInc += y
            if sumOfInc > realIndex:
                col = y - 1
                row = realIndex - (sumOfInc - y)
                return (row, col)

    def draw(self):
        # draw the colorTwo background
        self.drawBox(self.margin, self.margin, self.imageSize - self.margin, self.imageSize - self.margin, self.colorTwo)
        # draw the colorOne unit boxes
        self.drawUnit()
        # draw the gridline
        self.drawGrid()

    # draw a box from left top (x0,y0) to right bottom (x1,y1), exclude right & bottom (delta_mode off)
    # draw a box from left top (x0,y0) to right bottom (x0+x1,y0+y1), exclude right & bottom (delta_mode on)
    def drawBox(self, x0, y0, x1, y1, color, delta_mode = False):
        if delta_mode:
            x1 = x1 + x0
            y1 = y1 + y0
        for x in range(x0, x1):
            for y in range(y0, y1):
                self.pixels[x, y] = color # color should be a tuple in RGB (255, 255, 255)

    # draw the unit squares of using colorOne
    def drawUnit(self):
        for coord in self.coords:
            # original
            x0 = self.margin + self.gridWidth + coord[0]*(self.gridWidth+self.unitSize)
            y0 = self.margin + self.gridWidth + coord[1]*(self.gridWidth+self.unitSize)
            self.drawBox(x0, y0, self.unitSize, self.unitSize, self.colorOne, True)
            # diag mirror
            x0 = self.margin + self.gridWidth + coord[1]*(self.gridWidth+self.unitSize)
            y0 = self.margin + self.gridWidth + coord[0]*(self.gridWidth+self.unitSize)
            self.drawBox(x0, y0, self.unitSize, self.unitSize, self.colorOne, True)
            # vertical mirror
            x0 = self.margin + self.gridWidth + (self.fullNum - 1 - coord[0])*(self.gridWidth+self.unitSize)
            y0 = self.margin + self.gridWidth + coord[1]*(self.gridWidth+self.unitSize)
            self.drawBox(x0, y0, self.unitSize, self.unitSize, self.colorOne, True)
            x0 = self.margin + self.gridWidth + (self.fullNum - 1 - coord[1])*(self.gridWidth+self.unitSize)
            y0 = self.margin + self.gridWidth + coord[0]*(self.gridWidth+self.unitSize)
            self.drawBox(x0, y0, self.unitSize, self.unitSize, self.colorOne, True)
            # horizontal mirror
            x0 = self.margin + self.gridWidth + coord[0]*(self.gridWidth+self.unitSize)
            y0 = self.margin + self.gridWidth + (self.fullNum - 1 - coord[1])*(self.gridWidth+self.unitSize)
            self.drawBox(x0, y0, self.unitSize, self.unitSize, self.colorOne, True)
            # diag mirror
            x0 = self.margin + self.gridWidth + coord[1]*(self.gridWidth+self.unitSize)
            y0 = self.margin + self.gridWidth + (self.fullNum - 1 - coord[0])*(self.gridWidth+self.unitSize)
            self.drawBox(x0, y0, self.unitSize, self.unitSize, self.colorOne, True)
            # vertical mirror
            x0 = self.margin + self.gridWidth + (self.fullNum - 1 - coord[0])*(self.gridWidth+self.unitSize)
            y0 = self.margin + self.gridWidth + (self.fullNum - 1 - coord[1])*(self.gridWidth+self.unitSize)
            self.drawBox(x0, y0, self.unitSize, self.unitSize, self.colorOne, True)
            x0 = self.margin + self.gridWidth + (self.fullNum - 1 - coord[1])*(self.gridWidth+self.unitSize)
            y0 = self.margin + self.gridWidth + (self.fullNum - 1 - coord[0])*(self.gridWidth+self.unitSize)
            self.drawBox(x0, y0, self.unitSize, self.unitSize, self.colorOne, True)
            
    # draw the gridline
    def drawGrid(self):
        # horizontal
        for row in range(self.fullNum + 1):
            x0 = self.margin
            y0 = self.margin + row*(self.gridWidth+self.unitSize)
            x1 = self.imageSize - self.margin
            y1 = y0 + self.gridWidth
            self.drawBox(x0, y0, x1, y1, self.colorZero)
        # vertical
        for col in range(self.fullNum + 1):
            x0 = self.margin + col*(self.gridWidth+self.unitSize)
            y0 = self.margin
            x1 = x0 + self.gridWidth
            y1 = self.imageSize - self.margin
            self.drawBox(x0, y0, x1, y1, self.colorZero)


    # duplicate the current image horizontally n times and vertically m times
    def dupNbyM(self, n, m):
        self.needDup = True
        self.margin = 0 # overwrite margin
        self._imageGen()
        self.draw()
        # create a blank image with enlarged size
        new_image = Image.new('RGB', (self.imageSize*n, self.imageSize*m))
        for row in range(n):
            for col in range(m):
                new_image.paste(self.image, (row*self.imageSize, col*self.imageSize))
        # save to self.dupImage
        self.dupImage = new_image

    def save(self):
        if self.needDup:
            self.dupImage.save(self.imageName)
        else:
            self.image.save(self.imageName)