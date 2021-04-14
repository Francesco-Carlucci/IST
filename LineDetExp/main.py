import numpy as np
import math
import cv2
from matplotlib import image
from matplotlib import pyplot as plt
import cv2.xfeatures2d
import sys
import os
from functions import filterLines, points2rhotheta

#if len(sys.argv)<2:
  #print('Usage: pool_lines.py <infile>')
  #sys.exit()

#filename = sys.argv[1]

if len(sys.argv)<2:
  print('Usage: pool_lines.py <dir>')
  sys.exit()

basedir = sys.argv[1]

dstdir="./test_setPoolOnly"

i=0
for fn in os.listdir(basedir):
  maskname = fn.replace("Inpainted", "")
  dstname = dstdir + "/" + maskname             # path of the result image
  filename = os.path.join(basedir, fn)
  i+=1
  print(i," :", fn)
  np.set_printoptions(threshold=sys.maxsize)

  image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8),cv2.IMREAD_UNCHANGED)  # use that instead of cv.imread because this last gives None sometime
  oih = image.shape[0]
  oiw = image.shape[1]
  ih = int(oih)  # resize image to this size for processing
  iw = int(oiw)  # resize image to this size for processing

  image2 = cv2.resize(image, (iw, ih), cv2.INTER_AREA)
  gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) #convert img to grayscale
  kernel_size=5
  edges = cv2.Canny(image2, 50, 150)
  #cv2.imshow('edges', edges)
  #cv2.waitKey()
  # image2 = cv2.GaussianBlur(image2, (kernel_size, kernel_size), 0) #tried to filter the image to reduce glares

  hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV) #convert image to HSV

  mask = cv2.inRange(hsv, (70, 40, 40), (116, 255, 255)) #Filter image looking for light blue
  #mask2= cv2.inRange(hsv,(0,80,80),(255,255,255))  #second mask for bright points in the water, doesn't work
  #mask=cv2.bitwise_or(mask,mask2)
  #res2 = cv2.bitwise_and(image2, image2, mask=cv2.bitwise_not(mask)) #show selected areas on the image
  #cv2.imshow("mask", mask)
  #cv2.imshow("img and mask", res2)
  #cv2.waitKey()

  #dilate mask to make bigger contours grow and pool lanes connect
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
  kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
  i=0
  """
  while (i<5):
    cnt, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    imgcnt = cv2.resize(image, (iw, ih), cv2.INTER_AREA)
    cv2.drawContours(imgcnt, cnt, -1, (0, 255, 0), 3)
    #cv2.imshow("contorni 1", imgcnt)
    #cv2.waitKey()
    cnt.sort(key=cv2.contourArea, reverse=True)  # selecting contours by percentage without a fixed threshold
    cntAcc = cnt[:int(0.5 * len(cnt))]
    cv2.fillPoly(mask, pts=cnt[max(1,int(0.5 * len(cnt))):], color=(0, 0, 0))
    # show filtered contours
    imgcnt = cv2.resize(image, (iw, ih), cv2.INTER_AREA)
    cv2.drawContours(imgcnt, cntAcc, -1, (0, 255, 0), 3)
    #cv2.imshow("contorni filtrati", mask)
    #cv2.waitKey()
    i+=1
    mask = cv2.dilate(mask, kernel2)
    #cv2.imshow("mascheraDilated", mask)
    mask = cv2.erode(mask, kernel)
  cv2.imshow("mascheraEroded", mask)
  cv2.waitKey()
  cnt=cnt[0]
  hull = cv2.convexHull(cnt)
  """


  while (i<2):
    i+=1
    mask = cv2.dilate(mask, kernel2)
    #cv2.imshow("mascheraDilated", mask)
    mask = cv2.erode(mask, kernel2)
  #cv2.imshow("mascheraEroded", mask)
  #cv2.waitKey()

  cnt, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   #find contours

  """  #show contours found
  imgcnt = cv2.resize(image, (iw, ih), cv2.INTER_AREA)
  cv2.drawContours(imgcnt, cnt, -1, (0, 255, 0), 3)
  cv2.imshow("contorni", imgcnt)
  cv2.waitKey()
  """

  points= np.empty((0,2), dtype=int)  #It would be better to detect if there are multiple clusters of big contours
                                      #before flattening all their points in points array
  for obj in cnt :
    if cv2.contourArea(obj) > 800:   #filter contours by area
      #print("obj:", obj)
      for couple in obj :
        points=np.vstack((points,np.array([couple[0,0], couple[0,1]])))  #accumulate all the the points in an array

  #cnt.sort(key=cv2.contourArea, reverse=True)   #selecting contours by percentage without a fixed threshold
  #cntAcc=cnt[:int(0.05*len(cnt))]               #results quite similar to the method above
  """   #show filtered contours
  imgcnt = cv2.resize(image, (iw, ih), cv2.INTER_AREA)
  cv2.drawContours(imgcnt, cntAcc, -1, (0, 255, 0), 3)
  cv2.imshow("contorni filtrati", imgcnt)
  cv2.waitKey()
  """

  #print("points:", points) #checking that cnt has been flattened
  centre=(iw/2,ih/2)        #setting centrea as the image centre
  if (points is not None and points.shape[0] != 0):
    """
    xmin,ymin=np.amin(points, axis=0)     #selects some points in the image with x or y minimum
    xmax, ymax= np.amax(points, axis=0)   #better to just do the convex hull of all the points

    Idxmin=np.where(points[...,0] == xmin)
    Idymin=np.where(points[...,1] == ymin)
    Idxmax = np.where(points[...,0] == xmax)
    Idymax = np.where(points[...,1] == ymax)

    y1 = np.amin(points[Idxmin[0]],axis=0)[1]    # minimum y of x= xmin
    y2 = np.amax(points[Idxmin[0]], axis=0)[1]  # maximum y with x=xmin
    x1 = np.amin(points[Idymin[0]], axis=0)[0]  # minimum x of y= ymin
    x2 = np.amax(points[Idymin[0]], axis=0)[0]  # maximum x with y=ymin
    y3 = np.amin(points[Idxmax[0]],axis=0)[1]   # minimum y of x= xmax
    y4 = np.amax(points[Idxmax[0]], axis=0)[1]  # maximum y with x=xmax
    x3 = np.amin(points[Idymax[0]], axis=0)[0]  # minimum x of y= ymax
    x4 = np.amax(points[Idymax[0]], axis=0)[0]  # maximum x with y=ymax

    print("xmin: (",xmin,y1,") (",xmin, y2,")" )
    print("ymin: (",x1,ymin,") (",x2, ymin,")" )
    print("xmax: (", xmax, y3, ") (", xmax, y4, ")")
    print("ymax: (", x3, ymax, ") (", x4, ymax, ")")
    plt.plot(xmin, y1, marker='v', color="white")
    plt.plot(xmin, y2, marker='v', color="white")
    plt.plot(x1, ymin, marker='v', color="white")
    plt.plot(x2, ymin, marker='v', color="white")
    plt.plot(xmax, y3, marker='v', color="white")
    plt.plot(xmax, y4, marker='v', color="white")
    plt.plot(x3, ymax, marker='v', color="white")
    plt.plot(x4, ymax, marker='v', color="white")
    plt.imshow(image2)
    plt.show()
    limes=np.array([[xmin,y1], [xmin,y2],[x1,ymin],[x2,ymin],[xmax,y3],[xmax,y4],[x3,ymax],[x4,ymax]])
    """

    hull=cv2.convexHull(points)
    M = cv2.moments(hull)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    centre = (cx,cy)                #calculate centroid of the hull, then used to filter lines
    """ #show position of the centre
    plt.plot(cx, cy, marker='v', color="red")
    plt.imshow(image2)
    plt.show()
    """

    #imgcnt = cv2.resize(image, (iw, ih), cv2.INTER_AREA)  #show the shape detected only by colour search
    #if(hull is not None and hull.shape[0]!=0) :
      #cv2.drawContours(imgcnt, [hull], -1, (0, 0, 255), 2)
    #cv2.imshow("locus ad limes", imgcnt)
    #cv2.waitKey()
    #rect = cv2.minAreaRect(points)   #instead of hull calculate the rotated bounding box, less precise
    #box = cv2.boxPoints(rect)
    #box = np.int0(box)
    """  #show the rotated bounding box calculated by minAreaRect
    cv2.imshow("box", cv2.drawContours(image2, [box], 0, (0, 0, 255), 2))
    cv2.waitKey()
    """

  """    #show the convex hull of biggest contours filtered by area in cntAcc
  imgcnt = cv2.resize(image, (iw, ih), cv2.INTER_AREA)
  for obj in cntAcc :
    hull = cv2.convexHull(obj)
    cv2.drawContours(imgcnt, [hull], -1, (0, 0, 255), 2)
    #cv2.imshow("hull of big contours", imgcnt)
  cv2.waitKey()
  """
             #********************************LINES*****************************************#
  line_length_threshold = int(0.05*iw)   #segments shorter than 5% of width will be discarded
  lsd = cv2.ximgproc.createFastLineDetector(line_length_threshold, _canny_th1=20, _canny_th2=40, _do_merge=True)

  lines = lsd.detect(gray)

  try: lines
  except NameError: lines = None
  #eliminates vertical or lamost vertical lines
  angle_offset = 90  # not used         90    # 90° -> horizontal
  angle_tolerance = 20    # not used         30   # +- 45° tolerance

  if not lines is None:
    new_lines = filterLines(lines, centre, 0.2*iw, iw, ih, angle_tolerance, angle_offset)
  else:
    new_lines = np.array([])

  """      #filter lines by their average theta
  avgTheta = 0
  cntLen = 0
  for i in range(new_lines.shape[0]):
    x1, y1, x2, y2 = new_lines[i]
    line_length = math.sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1))
    rho, theta = points2rhotheta(new_lines[i, 0:2], new_lines[i, 2:4])
    avgTheta += theta * line_length   #weighted average using the length of the line
    cntLen += line_length
  avgTheta /= cntLen
  avgTheta = (((avgTheta) * 180.0 / math.pi) - 180) % 180
  
  angle_offset = avgTheta  # used         90    # 90° -> horizontal
  angle_tolerance = 4  # used         30   # +- 45° tolerance
  if not lines is None:
    new_lines = filterLines(lines, (iw / 2, ih / 2), 0.5 * iw, iw, ih, angle_tolerance, angle_offset)
  else:
    new_lines = np.array([])
  print("filter2", len(lines), len(new_lines))
  """
  d=0
  linePoints = np.empty((0, 2), dtype=int) #flat all the points extremes of the lines
  for i in range(new_lines.shape[0]):
    linePoints=np.vstack((linePoints,np.array([new_lines[i,0],new_lines[i,1]])))
    linePoints = np.vstack((linePoints, np.array([new_lines[i, 2], new_lines[i, 3]])))
    #image2 = cv2.line(imgcnt, tuple(new_lines[i,0:2].astype(int)), tuple(new_lines[i,2:4].astype(int)), (0,0+d*30,255), 2)
    """  #print length and theta of each line, just to check
    d+=1
    rho, theta = points2rhotheta(new_lines[i,0:2], new_lines[i,2:4])
    theta=(((theta) * 180.0 / math.pi)-180)%180
    x1, y1, x2, y2 = new_lines[i]
    line_length = math.sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1))
    print(new_lines[i], line_length,theta)
    cv2.imshow('result', image2)
    if cv2.waitKey(-1) & 0xFF == ord('q'):
      break
    """
  #cv2.imshow('lines', imgcnt)
  linesHull = cv2.convexHull(linePoints)

  # choose the biggest hull between the one find with colour and the lines one, take biggest area
  # because this way is less probable that it won't cut some swimmers
  imgmask= np.zeros((ih,iw,3), dtype=np.uint8)
  if(cv2.contourArea(linesHull)>cv2.contourArea(hull)) :
    print("lines")
    #cv2.drawContours(image2, [linesHull], -1, (0, 0, 255), 2)
    cv2.fillPoly(imgmask, pts=[linesHull], color=(255,255,255))
  else :
    print("colour")
    #cv2.drawContours(image2, [hull], -1, (0, 0, 255), 2)
    cv2.fillPoly(imgmask, pts=[hull], color=(255,255,255))
  #cv2.imshow("pool shape in white", imgmask)
  if cv2.waitKey(-1) & 0xFF == ord('q'):
    break
  mask = cv2.cvtColor(imgmask, cv2.COLOR_BGR2GRAY)

  #maskdir = './imgMaskOnly'
  #maskname = fn.replace("Inpainted", "")
  #maskpath = os.path.join(maskdir, maskname)
  #mask = cv2.imdecode(np.fromfile(maskpath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
  mask = cv2.resize(mask, (iw, ih), cv2.INTER_AREA)
  """
  cv2.imshow("mask", mask)
  cv2.waitKey()
  """
  imgres = cv2.bitwise_and(image2, image2, mask=cv2.bitwise_not(mask))
  """
  cv2.imshow("imgres", imgres)
  cv2.waitKey()
  """
  image2 = cv2.bitwise_and(image2, image2, mask=mask)
  """
  cv2.imshow("image2-1", image2)
  cv2.waitKey()
  image2 = cv2.add(imgres,image2)
  cv2.imshow("image2fin", image2)
  cv2.waitKey()
  """

   #show final shape detected
  #cv2.imshow("mask", imgmask)
  """
  cv2.imshow("Frontier", image2)
  if cv2.waitKey(-1) & 0xFF == ord('q'):
    break
  cv2.destroyWindow('result')
  """


  cv2.imwrite(dstname, image2)  #save file in destination directory


