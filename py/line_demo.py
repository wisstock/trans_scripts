#!/usr/bin/env python3

'''
Copyright © 2020 Borys Olifirov

Script for extract pixel values by line

scipy.ndimage.measurements.center_of_mass

'''

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


def getTiff(file_name, channel=0, frame_number=0, camera_offset=250):
    wd_path = os.path.split(os.getcwd())
    os.chdir(wd_path[0] + '/temp/data/')
    tiff_tensor = tifffile.imread(file_name)
    # print(tiff_tensor.shape, np.max(tiff_tensor))

    channel_one = tiff_tensor[0::2, :, :]
    channel_two = tiff_tensor[1::2, :, :]
    # print(channel_one.shape)
    # print(channel_two.shape)

    if channel == 0:
    	frame_amount = channel_one.shape
    	try:
    		img = channel_one[frame_number] - camera_offset
    		return(img)
    	except ValueError:
    		if frame_number > frame_amount[0]:
    			print("Frame number out of range!")
    else:
    	frame_amount = channel_two.shape
    	try:
    		img = channel_two[frame_number] - camera_offset
    		return(img)
    	except ValueError:
    		if frame_number > frame_amount[0]:
    			print("Frame number out of range!")

def lineSlice(img, angle=0, cntr_coord="center"):
    '''
    Returns coordinates of intersection points custom line with frame edges.
    Requires cell image mass center coordinates (default set up frame center)
    and angle value (in degree).


    Angles annotation:
                

           Vertical line (v)     
          180 
    Up(u)  | mass centre coord
           |/
    90 ----+---- Horizontal line (h) 
           |
    Down(d)|
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
      |  Up   *          |
      |        a* O      |
     A|-----------*------|A'
      |           | *    |
      |  D        |   *  |
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
     Y|*
      | *
      |  *                
      |   *               
      |    *C              
     B|-----* -----------               
      |      *           |
      |       *          |
      |        *         |
      |        a* O      |
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

    '''

    def anglPars(angl):
        '''
        Parse input angle value.
        Real angle range is from 0 till 180 degree
        (because slice is diameter, not radius)

        '''
        if 0 < angl < 90:
            logging.debug("anglPars done! d")
            return("d", math.radians(angl))
        elif 90 < angl < 180:
            logging.debug("anglPars done! u")
            return("u", math.radians(180-angl))
        elif angl == 0 | angl == 180:
            logging.debug("anglPars done! v")
            return("v", "NaN")
        elif angl == 90:
            logging.debug("anglPars done! h")
            return("h", "NaN")

    x0, y0, x1, y1 = 0, 0, 0, 0  
    img_shape = np.shape(img)

    logging.debug("array shape: %s", img_shape)

    x_lim = img_shape[0]-1
    y_lim = img_shape[1]-1

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
    logging.debug("indicator = %s" % indicator)


    if indicator == "h":  # boundary cases check
        x0, y0 = cntr_coord[0], 0
        x1, y1 = cntr_coord[0], y_lim
        logging.debug("coordinate h")
        logging.debug("0 point %s, 1 point %s" % ([x0, y0], [x1, y1]))
        return([x0, y0], [x1, y1])

    elif indicator == "v":  # boundary cases check
        x0, y0 = x_lim, cntr_coord[1]
        x1, y1 = 0, cntr_coord[1]
        logging.debug("coordinate v")
        logging.debug("0 point %s, 1 point %s" % ([x0, y0], [x1, y1]))
        return([x0, y0], [x1, y1])

    elif indicator == "u":
        # calculate up left (90-180)
        AB_left = np.int(cntr_coord[0]*math.tan(angl_rad))
        if  AB_left <= y_lim - cntr_coord[1]:
            logging.debug("AB left: %s" % AB_left)
            x0 = 0
            y0 = cntr_coord[1] + AB_left
        elif AB_left > y_lim - cntr_coord[1]:
            x0 = cntr_coord[0] - np.int((y_lim - cntr_coord[1])/math.tan(angl_rad))
            y0 = y_lim

        # calculate up right
        AB_right = np.int((x_lim - cntr_coord[0])*math.tan(angl_rad))
        if cntr_coord[1] - AB_right >= 0:
            x1 = x_lim
            y1 = cntr_coord[1] - AB_right
        elif cntr_coord[1] - AB_right < 0:
            x1 = x_lim  - np.int(cntr_coord[1]/math.tan(angl_rad))
            y1 = 0

    return([x0, y0], [x1, y1])

def lineExtract(img, start_coors = [0, 0], end_coord = [100, 100]):
    '''
    Extraction pixel values thoward the guide line
    Require img im NumPy array format and line ends coordinate (in px)

    '''

    logging.debug("start point, end point: %s, %s" % (start_coors, end_coord))

    x0, y0 = start_coors[0], start_coors[1]
    x1, y1 = end_coord[0], end_coord[1]
    line_length = int(np.hypot(x1-x0, y1-y0))  # calculate line length

    logging.debug("line length: %s" % (np.hypot(x1-x0, y1-y0)))

    x, y = np.linspace(x0, x1, line_length), np.linspace(y0, y1, line_length)  # calculate projection to axis

    logging.debug("x and y length: %s, %s" % (np.shape(x), np.shape(y)))

    output_img = img[x.astype(np.int), y.astype(np.int)]
    return(output_img)


# Generate some data...
# x, y = np.mgrid[-5:5:0.1, -5:5:0.1]
# z = np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2)


input_file = 'Fluorescence_435nmDD500_cell1.tiff'
angle = 120


img = getTiff(input_file, 1, 10)
# data_shape = img.shape()
# print(data_shape[1], data_shape[2])

start, end = lineSlice(img, angle)
print(start, end)
# line_slice = lineExtract(img, start, end)
shape = np.shape(img)
cntr = [np.int((shape[0]-1)/2),
        np.int((shape[1]-1)/2)]
print(cntr)


fig, (ax0) = plt.subplots(nrows=1,
                          ncols=1,
                          figsize=(8, 8))

ax0.imshow(img)  #, cmap='gray')
ax0.plot([start[1], end[1]], [start[0], end[0]], 'ro-')
ax0.scatter(cntr[1],cntr[0],color='r')

plt.gca().invert_yaxis()
plt.show()