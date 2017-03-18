from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # Defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending
    # on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Filling pixels inside the polygon defined by "vertices" with
    # the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once
    you want to average/extrapolate the line segments you detect to map
    out the full extent of the lane (going from the result shown in
    raw-lines-example.mp4 to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel)
    # for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    # Get the image size
    imshape = image.shape

    # Canny parameters
    low_threshold = 60
    high_threshold = 180
    blur_kernel = 5

    # Image make parameters
    vertices = np.array([[(150, imshape[0]),
                          (450, 310),
                          (490, 310),
                          (900, imshape[0])]],
                        dtype=np.int32)
    # Hough transform parameters
    rho = 2
    theta = np.pi/180
    hough_threshold = 15
    min_line_length = 40
    max_line_gap = 20

    # # # BEGIN PIPELINE # # #
    # Convert the image to grayscale, and apply a gaussian blur
    gs_img = grayscale(image)
    blur_img = gaussian_blur(gs_img, blur_kernel)
    # Perform Canny edge detection on the blurred grayscale
    canny_img = canny(gs_img, low_threshold, high_threshold)
    # Mask off polygonal area
    # Determine dynamically in the future?
    masked_img = region_of_interest(canny_img, vertices)
    # Retrieve lines from Hough transform
    img_lines = hough_lines(masked_img, rho, theta, hough_threshold,
                            min_line_length, max_line_gap)
    result = weighted_img(img_lines, np.copy(image))
    # # # END PIPELINE # # #
    return result


# solidWhiteRight
# solidWhiteCurve
# solidYellowCurve
# solidYellowCurve2
# solidYellowLeft
# whiteCarLaneSwitch
for image in os.listdir("test_images/"):
    output_img = process_image(mpimg.imread("test_images/" + image))
    plt.imshow(output_img, cmap='Greys_r')
    plt.show()
