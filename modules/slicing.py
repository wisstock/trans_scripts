#!/usr/bin/env python3

""" Copyright © 2020 Borys Olifirov

Functions for extract pixel values by line.

"""

import sys
import os
import logging

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage.external import tifffile
from skimage import data, img_as_float


def lineSlice(img, angle=1, cntr_coord="center"):
    """ Returns coordinates of intersection points  for the image edges
    and the line which go through point with set up coordinates
    at the set up angle.
    Requires cell image mass center coordinates
    (default set up frame center)
    and angle value (in degree).


    Angles annotation:
                

           Vertical line (v)     
           180 
    Up(u)   | mass centre coord
            |/
    270 ----+---- 90 Horizontal line (h) 
            |
    Down(d) |
            0

    y = a * x + b
    a = tg alpha
    b = y | x = 0
    

    IF deltha y < lim y

      y
      ^
      |
     Y|------------------
      |                  |
      |                  |
      |                  |
     B| *                |
      |   *              |
      |     *            |
      |       *          |
      |        a* O      |
     A|-----------*------|A'
      |           | *    |
      |           |   *  |
      |           |     *|B'
     S+-----------------------> x
                         X
    LEFT
    AB = AO * tg(a)

    RIGHT
    A'B' = OA' * tg(a)



    IF deltha y > lim y

      y
      ^
      |
     Y| *
      |  *
      |   *                
      |    *               
      |     *C              
     B|------* ----------               
      |       *          |
      |        *         |
      |         *        |
      |         a* O     |
     A|-----------*------|A'
      |           |*     |
      |           | *    |
      |           |  *   |B'
     S+---------------*-------> x
      |              C'* | 
      |                 *|X
      |

    LEFT
    BC = AO - AB/tg(a)

    RIGHT
    B'C' = OA' - A'B'/tg(a)                    

    """

    def anglPars(angl):
        """Parse input angle value.
        Real angle range is from 0 till 180 degree
        (because slice is diameter, not radius)

        """

        if 0 < angl < 90 or 180 < angl < 270:
            logging.debug("anglPars done! d")
            return("d", math.radians(90 - angl))

        elif 90 < angl < 180 or 270 < angl < 360:
            logging.debug("anglPars done! u")
            return("u", math.radians(angl-90))

        elif angl == 0 or angl == 180:
            logging.debug("anglPars done! v")
            return("v", 0)

        elif angl == 90 or angl == 270:
            logging.debug("anglPars done! h")
            return("h", 90)






    x0, y0, x1, y1 = 0, 0, 0, 0  # init line ends coordinates
    img_shape = np.shape(img)

    logging.info("Frame shape: %s", img_shape)
    logging.info("Slice angle: %s", angle)

    x_lim = img_shape[1]-1  # create global image size var 
    y_lim = img_shape[0]-1  # "-1" because pixels indexing starts from 0

    indicator, angl_rad = anglPars(angle)

    if cntr_coord == "center":
        cntr_coord = [np.int(x_lim/2),
                      np.int(y_lim/2)]  # [x, y]

        logging.info("Center mode, coord: %s" % cntr_coord)

    AO_left = cntr_coord[0]
    OA_right = x_lim - cntr_coord[0]

    x_cntr = cntr_coord[0]
    y_cntr = cntr_coord[1]
    

    logging.info("Custom center, coord: %s" % cntr_coord)


    if indicator == "h":  # boundary cases check
        x0, y0 = 0, cntr_coord[1]
        x1, y1 = x_lim, cntr_coord[1] 

        logging.debug("coordinate h")
        logging.debug("0 point %s, 1 point %s" % ([x0, y0], [x1, y1]))

        return([x0, y0], [x1, y1])

    elif indicator == "v":  # boundary cases check
        x0, y0 = cntr_coord[0], y_lim
        x1, y1 = cntr_coord[0], 0

        logging.debug("coordinate v")
        logging.debug("0 point %s, 1 point %s" % ([x0, y0], [x1, y1]))

        return([x0, y0], [x1, y1])


    elif indicator == "d":
        AB_left = np.int(cntr_coord[0]*math.tan(angl_rad))  # AB, see on comment above
        
        logging.debug("AB: %s" % AB_left)

        if  cntr_coord[1] - AB_left > 0:  # calculate down left (0-90)
            x0 = 0
            y0 = cntr_coord[1] - AB_left

            logging.debug("AB > 0. x0, y0 = %s, %s" % (x0, y0))
        elif cntr_coord[1] - AB_left <= 0:
            BO_left = np.int(cntr_coord[1]/math.tan(angl_rad))  # BO, see comment above
            x0 = cntr_coord[0] - BO_left
            y0 = 0

            logging.debug("AB < y max. x0, y0 = %s, %s" % (x0, y0))

        AB_right = np.int((x_lim - cntr_coord[0])*math.tan(angl_rad))  # A'B', see comment above

        logging.debug("A'B': %s" % AB_right)
        
        if AB_right < y_lim - cntr_coord[1]:  # calculate down right
            x1 = x_lim
            y1 = cntr_coord[1] + AB_right

            logging.debug("A'B' > y_max. x1, y1 = %s, %s" % (x1, y1))
        elif AB_right >= y_lim - cntr_coord[1]:
            OC_right = np.int((y_lim - cntr_coord[1])/math.tan(angl_rad))
            BC_right = ((x_lim - cntr_coord[0]) - OC_right)  # B'C', see comment above
            x1 = x_lim - BC_right
            y1 = y_lim

            logging.debug("A'B' > y max. x1, y1 = %s, %s" % (x1, y1))


    elif indicator == "u":
        AB_left = np.int(cntr_coord[0]*math.tan(angl_rad))  # AB, see comment above

        logging.debug("AB: %s" % AB_left)

        if  AB_left < y_lim - cntr_coord[1]:  # calculate up left (90-180)
            x0 = 0
            y0 = cntr_coord[1] + AB_left
            logging.debug("AB < y max. x0, y0 = %s, %s" % (x0, y0))
        elif AB_left >= y_lim - cntr_coord[1]:
            BO_left = np.int((y_lim - cntr_coord[1])/math.tan(angl_rad))  # BC, see comment above
            x0 = cntr_coord[0] - BO_left
            y0 = y_lim
            logging.debug("AB > y max. x0, y0 = %s, %s" % (x0, y0))

        AB_right = np.int((x_lim - cntr_coord[0])*math.tan(angl_rad))  # A'B', see comment above

        logging.debug("A'B': %s" % AB_right)

        if cntr_coord[1] - AB_right > 0:  # calculate up right
            x1 = x_lim
            y1 = cntr_coord[1] - AB_right

            logging.debug("A'B' > 0. x1, y1 = %s, %s" % (x1, y1))
        elif cntr_coord[1] - AB_right <= 0:
            BC_right = (cntr_coord[0] - np.int(cntr_coord[1]/math.tan(angl_rad)))  # B'C', see comment above
            x1 = x_lim - BC_right
            y1 = 0

            logging.debug("A'B' < 0. x1, y1 = %s, %s" % (x1, y1))

    return([x0, y0], [x1, y1])

def radiusSlice(img, angl=1, cntr_coord="center"):
    """ Returns coordinates of intersection points for the image edges
    and the line starting from center point at the set up angle. 

    Algorithm is the same as the function lineSlice

     y^
      |
      |
      |------------------------
      |                        |
      |                        |
      |                        |
      |                        |
      |                        |
     A|--------- ** * O        |
      |       **   *|          |
      |    **    *  |          |
     a|**      *    |          |
      |      *      |          |
      |    *        |          |
      +-------------------------------->
          b        B                  x


     II | III
        |
    ----+----
        |
      I | IV
    0

    """ 

    def anglPars(angl):
        """Parse input angle value.
        Real angle range is from 0 till 180 degree
        (because slice is diameter, not radius)

        """

        if 0 <= angl <90:
            return("I")  # , math.radians(angl))

        elif 90 <= angl < 180:
            return("II")  # , math.radians(angl))

        elif 180 <= angl < 270:
            return("III")  # , math.radians(angl))

        elif 270 <= angl <= 360:
            return("IV")  # , math.radians(angl))

    img_shape = np.shape(img)

    x_lim = img_shape[1]-1  # create global image size var 
    y_lim = img_shape[0]-1  # "-1" because pixels indexing starts from 0

    if cntr_coord == "center":
        cntr_coord = [np.int(x_lim/2),
                      np.int(y_lim/2)]  # [x, y]

        logging.info("Center mode, coord: %s" % cntr_coord)
    else:
        logging.info("Custom center, coord: %s" % cntr_coord)

    x0, y0 = 0, 0  # init line ends coordinates

    x_cntr = cntr_coord[0]  # AO_left = x_cntr, OA_right = x_lim - x_cntr
    y_cntr = cntr_coord[1]  # BO_down = y_cntr, OB_up = y_lim - y_cntr

    indicator = anglPars(angl)
    logging.debug("Square quartile: %s" % indicator)

    logging.info("Frame shape: %s, slice angle: %s" % (img_shape, angl))

    if indicator == 'I':
      Bb = np.int(y_cntr * math.tan(math.radians(angl)))

      logging.debug("Bb = %s" % Bb)

      if Bb <= x_cntr:
        x0 = x_cntr - Bb
        y0 = 0

      if Bb > x_cntr:
        Aa = np.int(x_cntr * math.tan(math.radians(90 - angl)))
        
        logging.debug("Aa = %s" % Aa)

        x0 = 0
        y0 = y_cntr - Aa


    elif indicator == 'II':
      Aa = np.int(x_cntr * math.tan(math.radians(angl - 90)))

      logging.debug("Aa = %s" % Aa)

      if Aa <= y_cntr + Aa:
        x0 = 0
        y0 = y_cntr + Aa

      if Aa > y_cntr:
        Bb = np.int((y_lim-y_cntr) * math.tan(math.radians(180 - angl)))
        
        logging.debug("Bb = %s" % Bb)

        x0 = x_cntr - Bb
        y0 = y_lim

    elif indicator == 'III':
      Bb = np.int((y_lim - y_cntr) * math.tan(math.radians(angl - 180)))

      logging.debug('Bb = %s' % Bb)

      if Bb <= x_lim - x_cntr:
        x0 = x_cntr + Bb
        y0 = y_lim

      if Bb > x_lim - x_cntr:
        Aa = np.int((x_lim - x_cntr) * math.tan(math.radians(270 - angl)))

        logging.debug('Aa = %s' % Aa)

        x0 = x_lim
        y0 = y_cntr + Aa

    elif indicator == 'IV':
      Aa = np.int((x_lim - x_cntr) * math.tan(math.radians(angl - 270)))

      logging.debug('Aa = %s' % Aa)

      if Aa <= y_cntr:
        x0 = x_lim
        y0 = y_cntr - Aa

      if Aa > y_cntr:
        Bb = np.int(y_cntr * math.tan(math.radians(360 - angl)))

        logging.debug('Bb = %s' % Bb)

        x0 = x_cntr + Bb
        y0 = 0

    return([x_cntr, y_cntr], [x0, y0])

def lineExtract(img, start_coors, end_coord):
    """ Returns values ​​of pixels intensity along the line
    with specified ends coordinates.

    Requires cell image, and line ends coordinate (results of lineSlice).

    """
    x0, y0 = start_coors[1], start_coors[0]
    x1, y1 = end_coord[1], end_coord[0]
    line_length = int(np.hypot(x1-x0, y1-y0))  # calculate line length

    # logging.debug("line length: %s" % (np.hypot(x1-x0, y1-y0)))

    x, y = np.linspace(x0, x1, line_length), np.linspace(y0, y1, line_length)  # calculate projection to axis

    logging.debug("X and Y length: %s, %s" % (np.shape(x)[0], np.shape(y)[0]))

    output = img[x.astype(np.int), y.astype(np.int)]

    return output

def bandExtract(img, start_coors, end_coord, band_width=2, mode="mean"):
    """ Returns values ​​of pixels intensity in fixed neighborhood (in pixels)
    for each points in the the line with specified ends coordinates.
    
    Requires cell image, and line ends coordinate (results of lineSlice).

    """

    def neighborPix(img, coord, shift):
      """ Mean value of matrix with setting shifr.
      Default 4x4.

      """
      sub_img = img[coord[0]-shift:coord[0]+shift:1,
                    coord[1]-shift:coord[1]+shift:1]

      mean_val = np.mean(sub_img)

      return mean_val

    def parallelPix(img, coord, length, shift):
      """ Calculate mean value for points with same indexes in several
      slice lines. Ends coordinates for each line 
      """
      pass


    x0, y0 = start_coors[1], start_coors[0]
    x1, y1 = end_coord[1], end_coord[0]
    line_length = int(np.hypot(x1-x0, y1-y0))  # calculate line length

    x, y = np.linspace(x0, x1, line_length), np.linspace(y0, y1, line_length)  # calculate projection to axis

    if mode == "mean":

      logging.info("Band side shift: %s px" % band_width)

      i = 0
      output = []

      while i < np.shape(x)[0]:
        x_point = np.int(x[i])
        y_point = np.int(y[i])

        output.append(neighborPix(img, [x_point, y_point], band_width))

        i += 1

      return output

    elif mode == 'parallel':
      pass  


if __name__=="__main__":
  pass
