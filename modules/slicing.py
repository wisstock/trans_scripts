#!/usr/bin/env python3

"""Copyright © 2020 Borys Olifirov

Functions for extract pixel values by line

scipy.ndimage.measurements.center_of_mass

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


logging.basicConfig(filename="sample.log",  # logging options
                    level=logging.DEBUG,
                    filemode="w")


def lineSlice(img, angle=1, cntr_coord="center"):
    """ Returns coordinates of intersection points custom line with frame edges.
    Requires cell image mass center coordinates (default set up frame center)
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

    logging.debug("array shape: %s", img_shape)

    x_lim = img_shape[1]-1  # create global image size var 
    y_lim = img_shape[0]-1  # "-1" because pixels indexing starts from 0

    logging.debug("X coord lim: %s, Y coord lim: %s" % (x_lim, y_lim))

    indicator, angl_rad = anglPars(angle)

    if cntr_coord == "center":
        cntr_coord = [np.int(x_lim/2),
                      np.int(y_lim/2)]  # [x, y]

        logging.debug("center mode")

    AO_left = cntr_coord[0]
    OA_right = x_lim - cntr_coord[0]

    x_cntr = cntr_coord[0]
    y_cntr = cntr_coord[1]
    

    logging.debug("center coord: %s" % cntr_coord)


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

def lineExtract(img, start_coors = [0, 0], end_coord = [100, 100]):
    """ Extraction pixel values thoward the guide line
    Require img im NumPy array format and line ends coordinate (in px).

    """

    logging.debug("start point, end point: %s, %s" % (start_coors, end_coord))

    x0, y0 = start_coors[1], start_coors[0]
    x1, y1 = end_coord[1], end_coord[0]
    line_length = int(np.hypot(x1-x0, y1-y0))  # calculate line length

    logging.debug("line length: %s" % (np.hypot(x1-x0, y1-y0)))

    x, y = np.linspace(x0, x1, line_length), np.linspace(y0, y1, line_length)  # calculate projection to axis

    logging.debug("x and y length: %s, %s" % (np.shape(x), np.shape(y)))

    output_img = img[x.astype(np.int), y.astype(np.int)]
    return(output_img)


# Generate some data...
# x, y = np.mgrid[-5:5:0.1, -5:5:0.1]
# z = np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2)

if __name__=="__main__":
    input_file = 'temp/data/Fluorescence_435nmDD500_cell1.tiff'
    angle = 45

    img = getTiff(input_file, 1, 10)
    # data_shape = img.shape()
    # print(data_shape[1], data_shape[2])

    # cntr = [250,800]

    start, end = lineSlice(img, angle)
    line_slice = lineExtract(img, start, end)
    shape = np.shape(img)
    cntr = [np.int((shape[1]-1)/2),
            np.int((shape[0]-1)/2)]


    fig, (ax0, ax1) = plt.subplots(nrows=2,
                              ncols=1,
                              figsize=(8, 8))

    ax0.imshow(img)  #, cmap='gray')
    ax0.plot([start[0], end[0]], [start[1], end[1]], 'ro-')
    ax0.scatter(cntr[0],cntr[1],color='r')
    # ax0.scatter(start[0]+5, start[1]+5)

    ax1.plot(line_slice)

    # plt.gca().invert_yaxis()
    plt.show()