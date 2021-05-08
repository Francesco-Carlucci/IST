import os
import random

import numpy as np
import torch
import torchvision
import cv2
PoolOnly=True
imgdir = './trainsetInpainted'
maskdir= './imgMaskOnly'
dstdir= './AugDataset2009'
dstmaskdir= './AugMasks'
for filename in os.listdir(imgdir):
    iw = int(1600)
    ih = int(900)

    mask = np.empty((ih ,iw ,3))  # in case of Dataset without border removal, we fill it with empty masks
    path = os.path.join(imgdir, filename)
    print(path)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    iw=image.shape[1]
    ih=image.shape[0]

    maskname = filename.replace("Inpainted", "")
    maskpath = os.path.join(maskdir, maskname)
    try:
        mask = cv2.imdecode(np.fromfile(maskpath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except:
        maskpath =maskpath.replace('centré', 'centrÃƒÂ©')
        maskpath = maskpath.replace('centrÃ©', 'centrÃƒÂ©')
        mask = cv2.imdecode(np.fromfile(maskpath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    mask = cv2.resize(mask, (iw, ih), cv2.INTER_AREA)

    dstpath = os.path.join(dstdir, filename)
    dstmask= os.path.join(dstmaskdir, maskname)
    cv2.imwrite(dstpath, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(dstmask, mask, [cv2.IMWRITE_JPEG_QUALITY, 90])

    #RANDOM CROP

    width=int(iw/5*3)
    height=int(ih/5*3)
    cx,cy=800,450
    x = random.randint(0, iw - width)
    y = random.randint(0, ih - height)
    cropped = image[y:y + height, x:x + width]
    maskcropped = mask[y:y + height, x:x + width]
    """
    cnt1 = np.array([[cx-width,cy-height],[cx,cy-height],[cx,cy],[cx-width,cy]]) #sx-up
    cnt2 = np.array([[cx,cy-height],[cx+width,cy-height],[cx+width,cy],[cx,cy]]) #dx-up
    cnt3 = np.array([[cx,cy],[cx+width,cy],[cx+width,cy+height],[cx,cy+height]]) #dx-down
    cnt4 = np.array([[cx,cy],[cx,cy+height],[cx-width,cy+height],[cx-width,cy]]) #sx-down
    image = cv2.drawContours(image, [cnt1], -1, (0, 0, 255), 2)
    image = cv2.drawContours(image, [cnt2], -1, (0, 255, 0), 2)
    image = cv2.drawContours(image, [cnt3], -1, (255, 0, 0), 2)
    image = cv2.drawContours(image, [cnt4], -1, (128, 128, 0), 2)
    mask = cv2.drawContours(mask, [cnt1], -1, (0, 0, 255), 2)
    mask = cv2.drawContours(mask, [cnt2], -1, (0, 255, 0), 2)
    mask = cv2.drawContours(mask, [cnt3], -1, (255, 0, 0), 2)
    mask = cv2.drawContours(mask, [cnt4], -1, (128, 128, 0), 2)
    image= cv2.resize(image, (width,height))
    mask = cv2.resize(mask, (width, height))
    """
    #cv2.imshow('image', image)
    #cv2.imshow('cropped',cropped)
    #cv2.imshow('mask', maskcropped)
    #cv2.waitKey()

    cropped = cv2.resize(cropped, (iw, ih))
    maskcropped = cv2.resize(maskcropped, (iw,ih))

    fncrop = filename.replace(".jpg", "Crop.jpg")
    dstCrop = os.path.join(dstdir, fncrop)
    #dstCrop2 =os.path.join('AugmentedDataset2009',fncrop)
    fnmask= maskname.replace(".jpg", "Crop.jpg")
    dstmask=os.path.join(dstmaskdir, fnmask)
    cv2.imwrite(dstCrop, cropped, [cv2.IMWRITE_JPEG_QUALITY, 90])             #RANDOM
    #cv2.imwrite(dstCrop2, cropped, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(dstmask, maskcropped,[cv2.IMWRITE_JPEG_QUALITY, 90])

    #Changing light and saturation

    light = np.random.choice(np.array([ -40, -30, 30, 40]))
    sat = np.random.choice(np.array([ -40, -30, 30, 40],dtype=np.uint8))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v=v+light
    v[v>255] = 255
    v[v<0] = 0
    v=v.astype(np.uint8)
    s=s+sat
    s[s>255] = 255
    s[s<0] = 0
    s = s.astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    coloured = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    fncol = filename.replace(".jpg", "Colo.jpg")
    fnmask = maskname.replace(".jpg", "Colo.jpg")
    dstmask = os.path.join(dstmaskdir, fnmask)
    dstCol = os.path.join(dstdir, fncol)
    cv2.imwrite(dstCol, coloured,[cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(dstmask, mask,[cv2.IMWRITE_JPEG_QUALITY, 90])               #no changes needed to the mask
    #cv2.imshow('coloured', coloured)
    #cv2.waitKey()

    #Rotation
    M = cv2.getRotationMatrix2D(((iw - 1) / 2.0, (ih - 1) / 2.0), 180, 1)
    rotated = cv2.warpAffine(image, M, (iw, ih))
    rotmask = cv2.warpAffine(mask, M, (iw, ih))

    #cv2.imshow('rotated', rotated)
    #cv2.waitKey()

    fnrot = filename.replace(".jpg", "Rot.jpg")
    fnmask = maskname.replace(".jpg", "Rot.jpg")
    dstmask = os.path.join(dstmaskdir, fnmask)
    dstRot = os.path.join(dstdir, fnrot)
    cv2.imwrite(dstRot, rotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(dstmask, rotmask, [cv2.IMWRITE_JPEG_QUALITY, 90])

    #Noise

    gauss = np.random.normal(0, 0.7, image.shape)
    gauss = gauss.astype('uint8')
    noise = cv2.add(image, gauss)

    #cv2.imshow('noise', noise)
    #cv2.waitKey()

    #fnnois = filename.replace(".jpg", "Nois.jpg")
    #fnmask = maskname.replace(".jpg", "Nois.jpg")
    #dstmask = os.path.join(dstmaskdir, fnmask)
    #dstnois = os.path.join(dstdir, fnnois)
    #cv2.imwrite(dstnois, noise, [cv2.IMWRITE_JPEG_QUALITY, 90])
    #cv2.imwrite(dstmask, mask, [cv2.IMWRITE_JPEG_QUALITY, 90])              # no changes needed to the mask

    #Flip

    flipped1=cv2.flip(image,0)
    mask1 = cv2.flip(mask,0)
    flipped2=cv2.flip(image,1)
    mask2 = cv2.flip(mask,1)

    #cv2.imshow('img',cv2.resize(image,(int(iw/5),int(ih/5))))
    #cv2.imshow('flip1',cv2.resize(flipped1,(int(iw/5),int(ih/5))))
    #cv2.imshow('flip2', cv2.resize(flipped2, (int(iw / 5), int(ih / 5))))
    #cv2.waitKey()

    fnflip1 = filename.replace(".jpg", "Flipvert.jpg")
    fnflip2 = filename.replace(".jpg", "Fliporiz.jpg")
    fnmask1 = maskname.replace(".jpg", "Flipvert.jpg")
    fnmask2 = maskname.replace(".jpg", "Fliporiz.jpg")
    dstmask1 = os.path.join(dstmaskdir, fnmask1)
    dstmask2 = os.path.join(dstmaskdir, fnmask2)
    dstflip1 = os.path.join(dstdir, fnflip1)
    dstflip2 = os.path.join(dstdir, fnflip2)
    cv2.imwrite(dstflip1, flipped1, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(dstflip2, flipped2, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(dstmask1, mask1, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(dstmask2, mask2, [cv2.IMWRITE_JPEG_QUALITY, 90])

    #Filter

    filtered=cv2.blur(image, (10, 10))

    #cv2.imshow('filtered', cv2.resize(filtered, (int(iw / 5), int(ih / 5))))
    #cv2.waitKey()

    fnfilt = filename.replace(".jpg", "Blur.jpg")
    fnmask = maskname.replace(".jpg", "Blur.jpg")
    dstmask = os.path.join(dstmaskdir, fnmask)
    dstfilt = os.path.join(dstdir, fnfilt)
    cv2.imwrite(dstfilt, filtered, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(dstmask, mask, [cv2.IMWRITE_JPEG_QUALITY, 90])                    # no changes needed to the mask
