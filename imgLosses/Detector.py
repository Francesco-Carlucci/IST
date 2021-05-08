import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from functions import filterLines, points2rhotheta

basedir='./Aug2009Losses50'
imgdir='../../test_set'
maskdir='../../test_setnewMasks'
dstdir = '.'

d=0
for fn in os.listdir(basedir):
    filename = os.path.join(basedir, fn)
    image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    imgname=os.path.join(imgdir,fn)
    img= cv2.imdecode(np.fromfile(imgname, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (image.shape[1], image.shape[0]), cv2.INTER_AREA)
    maskname = fn.replace("loss", "")
    maskname = fn.replace("Inpainted", "")
    maskpath = os.path.join(maskdir, maskname)
    try:
        mask = cv2.imdecode(np.fromfile(maskpath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except:
        maskpath = maskpath.replace('centre', 'centrÃƒÂ©')
        maskpath = maskpath.replace('centrÃ©', 'centrÃƒÂ©')
        # maskpath = maskpath.replace('centrÃƒÂ©', 'centre')   # 1
        # maskpath = maskpath.replace('centrÃƒÂ©', 'centrÃ©')  # 2
        mask = cv2.imdecode(np.fromfile(maskpath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), cv2.INTER_AREA)
    if (len(mask.shape) == 3):
        image = image * (mask // 255)
    else:
        image = cv2.bitwise_and(image, image, mask=mask)
    #cv2.imshow('loss', np.array(np.sum(image, axis=2), np.uint8))
    cv2.waitKey()
    #cv2.imshow("img", image)
    #cv2.waitKey()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert img to grayscale
    #---------------------Lines used to cover the red pool lanes-------------------------------
    line_length_threshold = int(0.03 * image.shape[1])  # segments shorter than 2% of width will be discarded
    lsd = cv2.ximgproc.createFastLineDetector(line_length_threshold, _canny_th1=20, _canny_th2=40, _do_merge=True)
    lines = lsd.detect(gray)

    angle_offset = 90  # not used         90    # 90° -> horizontal
    angle_tolerance = 20  # not used         30   # +- 45° tolerance
    ih = image.shape[0]
    iw = image.shape[1]

    if not lines is None:                 #filter lines
        new_lines = filterLines(lines, (iw/2,ih/2), 0.5 * iw, iw, ih, angle_tolerance, angle_offset)

    avgTheta = 0
    cntLen = 0
    for i in range(new_lines.shape[0]):
        x1, y1, x2, y2 = new_lines[i]
        line_length = np.math.sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1))
        rho, theta = points2rhotheta(new_lines[i, 0:2], new_lines[i, 2:4])
        avgTheta += theta * line_length  # weighted average using the length of the line
        cntLen += line_length
    avgTheta /= cntLen
    avgTheta = (((avgTheta) * 180.0 / np.math.pi) - 180) % 180

    angle_offset = avgTheta  # not used         90    # 90° -> horizontal
    angle_tolerance = 10  # not used         30   # +- 45° tolerance

    if not lines is None:                 #filter lines
        lines = filterLines(lines, (iw/2,ih/2), 0.5 * iw, iw, ih, angle_tolerance, angle_offset)
    #print(lines,new_lines)
    for i in range(lines.shape[0]):          #write the lines
        image = cv2.line(image, tuple(lines[i, 0:2].astype(int)), tuple(lines[i, 2:4].astype(int)), (30, 0, 0), 4)

    cv2.imshow("lines", image)
    cv2.waitKey()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert image to HSV

    colourMask = cv2.inRange(hsv, (0, 160, 80), (32, 255, 255)) #looks for dark red
    mask2 = cv2.inRange(hsv, (150, 120, 60), (180, 255, 255))
    mask2 = cv2.add(cv2.erode(mask2,(1,1)),colourMask)

    i = 0
    while (i < 5):
        i += 1
        dilatedMask = cv2.dilate(mask2, kernel2)
        dilatedMask = cv2.erode(dilatedMask, kernel)

    cnt, hierarchy = cv2.findContours(dilatedMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    cnt.sort(key=cv2.contourArea, reverse=True)
    #cntAcc = cnt[:int(0.1 * len(cnt)) if len(cnt) >= 1 else 1]
    #print(len(cntAcc))
    colouredMask=dilatedMask.copy()
    colouredMask=cv2.fillPoly(colouredMask, cnt[int(0.2 * len(cnt)):], (0))
    """
    cntAcc=[]
    colouredMask=colourMask.copy()
    for obj in cnt:
        if(cv2.contourArea(obj)<100):
            colouredMask=cv2.fillPoly(colouredMask, obj, (0))
        else:
            cntAcc.append(np.array(obj))
    """
    """
    i = 0
    while (i < 5):
        i += 1
        dilatedMask = cv2.dilate(colouredMask, kernel2)
        dilatedMask = cv2.erode(dilatedMask, kernel)
    """
    cv2.imshow("mascheraOriginal", colourMask)
    cv2.imshow("maschera2", mask2)
    cv2.imshow("mascheraColoured", colouredMask)
    cv2.imshow("mascheraDilated", dilatedMask)
    cv2.waitKey()

    cnt, hierarchy = cv2.findContours(dilatedMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    cnt.sort(key=cv2.contourArea, reverse=True)
    #cntAcc = cnt[:int(0.8 * len(cnt)) if len(cnt) >= 1 else 1]
    #cntAcc = []
    for obj in cnt:
        x, y, w, h = cv2.boundingRect(obj)
        print(h*w)
        if (cv2.contourArea(obj)<20 or w>2.5*h or h>3*w or h*w>600):    #control h>3*w
            colouredMask = cv2.fillPoly(colouredMask, obj, (0))
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cntAcc.append(np.array(obj))

                  #show contours found
    #imgcnt = cv2.resize(image, image.shape, cv2.INTER_AREA)
    cv2.imshow("img", img)
    #cv2.drawContours(image, cntAcc, -1, (0, 255, 0), 3)
    #cv2.imshow("contorni", image)
    cv2.waitKey()
    cv2.imwrite(dstdir+str(d)+'.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    d+=1


