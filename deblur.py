from pylab import *
import numpy
import scipy.ndimage

width = 100
height = 100
depth = 10
imgs = zeros((height, width, depth))

# prepare test input, a stack of images which is zero except for a point which has been blurred by a 3D gaussian
#sigma = 3
#imgs[height/2,width/2,depth/2] = 1
#imgs = scipy.ndimage.filters.gaussian_filter(imgs, sigma)

# read real input from stack of images img_0000.png, img_0001.png, ... (total number = depth)
# these must have the same dimensions equal to width x height above
# if imread reads them as having more than one channel, they need to be converted to one channel
for k in range(depth):
    imgs[:,:,k] = scipy.ndimage.imread( "img_%04d.png" % (k) )

# prepare output array, top and bottom image in stack don't get filtered
out_imgs = zeros_like(imgs)
out_imgs[:,:,0] = imgs[:,:,0]
out_imgs[:,:,-1] = imgs[:,:,-1]

# apply nearest neighbor deconvolution
alpha = 0.4 # adjustabe parameter, strength of filter
sigma_estimate = 3 # estimate, just happens to be same as the actual

for k in range(1, depth-1):
    # subtract blurred neighboring planes in the stack from current plane
    # doesn't have to be gaussian, any other kind of blur may be used: this should approximate PSF
    out_imgs[:,:,k] = (1+alpha) * imgs[:,:,k]  \
        - (alpha/2) * scipy.ndimage.filters.gaussian_filter(imgs[:,:,k-1], sigma_estimate) \
        - (alpha/2) * scipy.ndimage.filters.gaussian_filter(imgs[:,:,k+1], sigma_estimate)

# show result, original on left, filtered on right
compare_img = copy(out_imgs[:,:,depth/2])
compare_img[:,:width/2] = imgs[:,:width/2,depth/2]
imshow(compare_img)
show()