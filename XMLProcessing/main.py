import os
#from skimage import io
#from skimage.restoration import inpaint
import numpy as np
import xmltodict
import cv2 as cv


imgdir="./testset"
xmldir="./labelset"
dstdir="./imgTestInp"

if __name__ == '__main__':
    fid=0
    for filename in os.listdir(imgdir):
        fid+=1
        print("file n.", fid, " :",filename)
        imgname=imgdir+"/"+filename  #the name include the relative path of the image
        xmlname=xmldir+"/"+filename  #the path and name of the xml file
        xmlname=xmlname.replace("jpg", "xml")   #change extension
        dstname=dstdir+"/"+filename             #path of the result image
        dstname=dstname.replace(".jpg", "Inpainted")
        dstname+=".jpg"
        img = cv.imdecode(np.fromfile(imgname, dtype=np.uint8), cv.IMREAD_UNCHANGED) #use that instead of cv.imread because this last gives None sometimes
        #imgSci= io.imread(imgname)
        #h, w = img.shape[:-1]                          #reads dimensions from the xml
        currentxml= open(xmlname, 'r')
        my_dict = xmltodict.parse(currentxml.read())    #reads the xml file in a python dictionary
        #I've found xmltodict easier that the recommended ElementTree
        w=int(my_dict['annotation']['size']['width'])
        #print("width:", w)
        h=int(my_dict['annotation']['size']['height'])
        #print("Height:", h)
        d=0
        mask = np.zeros((h, w, 1), dtype=np.uint8)
        #sciMask=np.zeros(imgSci.shape[:-1])             #mask for the scikit inpainting function
        for i in my_dict['annotation']['object'] :
            xmin = int(i['bndbox']['xmin'])
            ymin = int(i['bndbox']['ymin'])
            xmax = int(i['bndbox']['xmax'])
            ymax = int(i['bndbox']['ymax'])
            mask[ymin:ymax, xmin:xmax] = 255 #to distinguish the boundingboxes on the mask, giving them different coulours
            #sciMask[ymin:ymax, xmin:xmax] = 1
            #show the coordinates and each bounding box on the mask
            """
            d+=1
            print("bndbox n.:", d)
            print("xmin :",xmin)
            print("ymin :",ymin)
            print("xmax :",xmax)
            print("ymax :",ymax)
            maskshw=mask.copy()  #show the mask resized to fit the screen
            maskshw = cv.resize(maskshw, (800, 450))
            #cv.imshow('shw', maskshw)
            #cv.waitKey(0)
            """
        blueMask= np.full((h, w, 3), [255,231,76] , dtype=np.uint8)
        hsv=cv.cvtColor(np.full((h, w, 3),[90, 100, 200], dtype=np.uint8), cv.COLOR_HSV2BGR)  # convert image from HSV to BGR to test colours
        #print(hsv)
        """
        image_resShw = cv.resize(hsv, (800, 450))
        cv.imshow('hsv', image_resShw)
        cv.waitKey(0)
        image_resShw = cv.resize(blueMask, (800, 450))
        cv.imshow('bluecolor', image_resShw)
        cv.waitKey(0)
        cv.destroyAllWindows()
        """

        res1 = cv.bitwise_and(blueMask, blueMask, mask=mask)
        res2 = cv.bitwise_and(img, img, mask=cv.bitwise_not(mask))
        #imageBlue = cv.add(res1,res2)
        image_result = cv.inpaint(img, mask, 60, cv.INPAINT_TELEA)
        print(dstname)
        cv.imwrite(dstname, image_result)

        """#show the inpainted image to check it
        image_resShw=cv.resize(res1, (800, 450))
        cv.imshow('bluemask', image_resShw)
        cv.waitKey(0)
        image_resShw = cv.resize(res2, (800, 450))
        cv.imshow('cutimg', image_resShw)
        cv.waitKey(0)
        cv.destroyAllWindows()
        """


        """   #tried with scikit biharmonic inpainting, but it's too slow
        image_defect = imgSci.copy()
        for layer in range(image_defect.shape[-1]):
            image_defect[np.where(sciMask)] = 0
        #cv.imshow('shw', image_defect)
        #cv.waitKey(0)
        
        image_resultSci = inpaint.inpaint_biharmonic(image_defect, sciMask, multichannel=True)
        image_resShw = cv.resize(image_resultSci, (800, 450))
        cv.imshow('shw', image_resShw)
        cv.waitKey(0)
        """







