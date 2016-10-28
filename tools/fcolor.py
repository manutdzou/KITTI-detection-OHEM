import cv

def Color(image):
    w = image.width
    h = image.height
    size = (w,h)
    iColor = cv.CreateImage(size,8,3)
    for i in range(h):
        for j in range(w):
            r = GetR(image[i,j])
            g = GetG(image[i,j])
            b = GetB(image[i,j])
            iColor[i,j] = (r,g,b)
    return iColor
    
def GetR(gray):
    if gray < 127:
        return 0
    elif gray > 191:
        return 255
    else:
        return (gray-127)*4-1

        
def GetG(gray):
    if gray < 64:
        return 4*gray
    elif gray > 191:
        return 256-(gray-191)*4
    else:
        return 255
        
def GetB(gray):
    if gray < 64:
        return 255
    elif gray > 127:
        return 0
    else:
        return 256-(gray-63)*4

FCArray = [(0,51,0),(0,51,102),(51,51,102),(51,102,51),\
            (51,51,153),(102,51,102),(153,153,0),(51,102,153),\
            (153,102,51),(153,204,102),(204,153,102),(102,204,102),\
            (153,204,153),(204,204,102),(204,255,204),(255,255,204)]        
def FColor(image,array=FCArray):
    w = image.width
    h = image.height
    size = (w,h)
    iColor = cv.CreateImage(size,8,3)
    for i in range(h):
        for j in range(w):
            iColor[i,j] = array[int(image[i,j]/16)]
    return iColor
        




