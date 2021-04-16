import numpy as np
import math
import cv2
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


for fn in os.listdir(basedir):

  filename = os.path.join(basedir, fn)

  image = cv2.imread(filename)
  oih = image.shape[0]
  oiw = image.shape[1]
  ih = int(oih/3)  # resize image to this size for processing
  iw = int(oiw/3)  # resize image to this size for processing

  image2 = cv2.resize(image, (iw, ih), cv2.INTER_AREA)
  gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) #convert img to grayscale
  kernel_size=5
  #gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
  #cv2.imshow('afterBur', gray)

  line_length_threshold = int(0.05*iw)   #segments shorter than 5% of width will be discarded
  lsd = cv2.ximgproc.createFastLineDetector(line_length_threshold, _canny_th1=20, _canny_th2=40, _do_merge=True)

  lines = lsd.detect(gray)

  try: lines
  except NameError: lines = None

  angle_offset = 90  # not used         90    # 90° -> horizontal
  angle_tolerance = 10    # not used         30   # +- 45° tolerance

  if not lines is None:
    new_lines = filterLines(lines, (iw/2,ih/2), 0.1*iw, iw, ih, angle_tolerance, angle_offset)
  else:
    new_lines = np.array([])
  print(len(lines), len(new_lines))

  for i in range(new_lines.shape[0]):
    image2 = cv2.line(image2, tuple(new_lines[i,0:2].astype(int)), tuple(new_lines[i,2:4].astype(int)), (0,0,255), 2)
    """
    cv2.imshow('result', image2)
    if cv2.waitKey(-1) & 0xFF == ord('q'):
      break
    rho, theta = points2rhotheta(lines[i,0,0:2], lines[i,0,2:4])
    theta=(((theta) * 180.0 / math.pi)-180)%180
    print(lines[i], new_lines[i],rho,theta)
    """

  #cv2.imwrite("prova.jpg", image2)
  cv2.imshow('result',image2)

  if cv2.waitKey(-1) & 0xFF == ord('q'):
    break


cv2.destroyWindow('result')

#input("Press key to continue...")


