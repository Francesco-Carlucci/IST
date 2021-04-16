import io
import os
import cv2
import matplotlib as plt
import numpy as np
#from PIL import Image

imgdir = './'
fn='7Collines19_100 NL D FINALE A BLUME HEEMSKERK PELLEGRINI_349Inpainted'

if __name__ == '__main__': #generate image from bit
    height=900
    width=1600
    ymin=351
    xmin=52
    ymax=598
    xmax=140
    path=imgdir+fn
    img=cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    ax = plt.subplot(1, 1, 1)
    ax.set_title('img')
    ax.imshow(img)
    plt.show()
    """
    template = np.zeros((height, width, 3), dtype=np.uint8)
    #xmin=int(xmin/1600)
    #ymin=int(ymin/900)
    #xmax=int(xmax/1600)
    #ymax=int(ymax/900)
    n=255
    template[xmin:xmin + xmax, ymin:ymin + ymax] = [255,255,255]
    np.set_printoptions(threshold=np.inf)
    #print(template)
    tempByte = bytes(template)
    #print(tempByte)
    imgsize=(1600,900)
    img= Image.fromarray(template, mode='RGB')
    #img = Image.frombytes('L', imgsize,  tempByte)
    #img = Image.new('RGB', (60, 30), color='red')
    img.save('mask.jpg')
    img2=np.asarray(img)
    print(img2)
    """
