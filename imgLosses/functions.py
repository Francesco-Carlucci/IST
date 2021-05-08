import math
import numpy as np

def filterLines(lines, centre, dist_thresh, img_width, img_height, angle_tolerance, angle_offset):
  new_lines = np.empty((0,4), dtype=int)
  for i in range(lines.shape[0]):
    rho, theta = points2rhotheta(lines[i,0,0:2], lines[i,0,2:4])
    th = (theta)*180.0/math.pi  #+ angle_offset
    if th>360:
     th -= 360
    #print(i," ",rho," ",th)
    x1, y1, x2, y2 = lines[i,0]
    x0 = centre[0]
    y0 = centre[1]
    # length of line
    line_length = math.sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1))
    # distance from the point centre
    dist = np.abs( (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1 )/math.sqrt((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1)) # source: wikipedia
    """  #i comment to study what the script does
    select the angle closer than the angle tolerance from the angle offset
      -first condition also:
    and lines closer to the centre than the threshold 
    and the lines should be in the low section of the image (below 0.6 height)
    and at the right of the centre
    
    and (y1>0.4*img_height or y2>0.4*img_height) and x1>x0-0.05*img_width and x2>x0-0.05*img_width:
    and ((th>360-angle_tolerance and th<angle_tolerance) or (th>180-angle_tolerance and th<180+angle_tolerance))
    """
    if dist < dist_thresh :
      #if ((th>360-angle_tolerance and th<angle_tolerance) or (th>180-angle_tolerance and th<180+angle_tolerance)):
      th2 = (th+360)%360    #bring the angle to [0,360)
      th2 = (th2+180)%180   #convert the angles to [0,180)
      """
      remove the lines with a angle between (20,160) or (40,140), so almost vertical
      and all the lines with an angle farther than the angle tolerance from the angle offset
      """
      if th<360 and abs(th2)>40 and abs(th2-180)>40 and ((abs(th-angle_offset)%360)<angle_tolerance or (abs(-th+angle_offset)%360)<angle_tolerance):
        #if th<360 and abs(th2)>20 and abs(th2-180)>20:
        new_lines = np.vstack((new_lines, np.array(lines[i,0,0:4].astype(int))))
      #print('added %f   (%d, %d) (%d, %d) *******************************' % (th2, x1, y1, x2, y2))
  return new_lines;

def points2rhotheta(p1, p2):
  np1 = np.array(p1)
  np2 = np.array(p2)
  if (np2[0]-np1[0])==0:
    theta = 1e16
    rho = np1[0]
  else:
    a = (np2[1]-np1[1])/(np2[0]-np1[0])
    b = np1[1] - a*np1[0]
    #print(a, b)
    (rho, theta) = ab2rhotheta(a, b)
  return (rho, theta)

def ab2rhotheta(a, b):
  """ convert line from form y = a*x + b into Hesse normal form (rho, theta (rad)) """
  """ also : y - ax - b = 0 """
  """        y*sin(theta) + x*cos(theta) - rho = 0 """
  #print("a: %f  b: %f" % (a, b))
  theta = math.atan(a) + math.pi/2.0
  rho = b*math.sin(theta)
  #print("a: %f  b: %f  rho: %f  theta: %f" % (a, b, rho, theta))
  return (rho, theta)



