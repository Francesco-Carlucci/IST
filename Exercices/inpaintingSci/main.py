import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#from skimage import io
from skimage.restoration import inpaint


if __name__ == '__main__':
    #filename = os.path.join('prova.jpg')
    #image_orig = io.imread(filename)
    xmin = 351  #xmin
    ymin = 52   #ymin
    xmax = 598
    ymax = 140
    img = cv.imread('prova.jpg')
    h, w= img.shape[:-1]
    # Create mask with a bounding box defect regions
    #mask=  np.zeros(img.shape[:-1])
    cvmask = np.zeros((h,w,1), dtype=np.uint8)
    cvmask[ymin:ymin + ymax, xmin:xmin + xmax] = 1

    #image_result = inpaint.inpaint_biharmonic(image_defect, mask, multichannel=True)
    image_result=cv.inpaint(img, cvmask, 3, cv.INPAINT_TELEA)
    #cv.imshow('dst', image_result)
    image_result2 = cv.inpaint(img, cvmask, 3, cv.INPAINT_NS)
    #cv.imshow('dst', image_result2)

    fig, axes = plt.subplots(ncols=2, nrows=2)
    ax = axes.ravel()

    ax[0].set_title('Original image')
    ax[0].imshow(img)

    ax[1].set_title('Mask')
    ax[1].imshow(cvmask, cmap=plt.cm.gray)

    #ax[2].set_title('Defected image')
    #ax[2].imshow(image_defect)

    ax[2].set_title('Inpainted image TELEA')
    ax[2].imshow(image_result)

    ax[3].set_title('NS image')
    ax[3].imshow(image_result2)

    for a in ax:
        a.axis('off')

    fig.tight_layout()
    plt.show()
    imgres = Image.fromarray(image_result, mode='RGB'
                                                '')
    imgres.save("provaInpainted.jpg")



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
