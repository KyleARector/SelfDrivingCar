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


def calc_slope(x1, y1, x2, y2):
    return (y2 - y1)/(x2 - x1)


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
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
    right_lines_x = []
    right_lines_y = []
    left_lines_x = []
    left_lines_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calculate slope to determine what side the line is on
            # Minimum length check - again?
            # Check if no lines per side?
            slope = (y2 - y1)/(x2 - x1)
            if calc_slope(x1, y1, x2, y2) < 0:
                left_lines_x.extend([x1, x2])
                left_lines_y.extend([y1, y2])
            else:
                right_lines_x.extend([x1, x2])
                right_lines_y.extend([y1, y2])

    # Create best fit lines based on points from either side
    # Left side
    l_x = np.array(left_lines_x)
    l_y = np.array(left_lines_y)
    l_A = np.vstack([l_x, np.ones(len(l_x))]).T
    l_m, l_b = np.linalg.lstsq(l_A, l_y)[0]
    l_x_bottom = int((img.shape[0] - l_b)/l_m)
    l_x_top = int((320-l_b)/l_m)
    cv2.line(img, (l_x_bottom, img.shape[0]), (l_x_top, 320), color, thickness)
    # Right side
    r_x = np.array(right_lines_x)
    r_y = np.array(right_lines_y)
    r_A = np.vstack([r_x, np.ones(len(r_x))]).T
    r_m, r_b = np.linalg.lstsq(r_A, r_y)[0]
    r_x_bottom = int((img.shape[0] - r_b)/r_m)
    r_x_top = int((320-r_b)/r_m)
    cv2.line(img, (r_x_bottom, img.shape[0]), (r_x_top, 320), color, thickness)
    # cv2.line(img, (x1, y1), (x2, y2), color, thickness)


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
    # Get the image size
    imshape = image.shape

    # Canny parameters
    low_threshold = 60
    high_threshold = 180
    blur_kernel = 5

    # Image make parameters
    vertices = np.array([[(150, imshape[0]),
                          (450, 320),
                          (490, 320),
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


# Cycle through images, process them, and save in same directory
# Directory could be defined from cmdln or config file
in_directory = "test_images"
out_directory = in_directory + "_output/"
in_directory += "/"
for image in os.listdir(in_directory):
    output_img = process_image(mpimg.imread(in_directory + image))
    # Reorder color channels before saving
    cv2.imwrite(out_directory + image,
                cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

white_output = 'test_videos_output/solidYellowLeft.mp4'
clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
