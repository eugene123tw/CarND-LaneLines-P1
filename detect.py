#importing some useful packages
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

DEBUG = True

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
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, cache, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    n = 10
    H, W, _ = img.shape

    def getSlope(x1, y1, x2, y2):
        slope = (y2 - y1) / (x2 - x1 + 0.00000001)
        return slope

    def getYInt(x1, y1, x2, y2):
        m = getSlope(x1, y1, x2, y2)
        b = y1 - m * x1
        return m, b

    x1_ls, y1_ls, x2_ls, y2_ls, x1_rs, y1_rs, x2_rs, y2_rs = [], [], [], [], [], [], [], []

    if DEBUG:
        debug_img = img.copy()

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = ((y2 - y1) / (x2 - x1 + 0.000001))
            if 0.1<math.fabs(slope)<1:
                if slope < 0:  # Left line
                    x1_ls.append(x1), y1_ls.append(y1), x2_ls.append(x2), y2_ls.append(y2)
                else:  # Right line
                    x1_rs.append(x1), y1_rs.append(y1), x2_rs.append(x2), y2_rs.append(y2)

            if DEBUG:
                if 0.1 < math.fabs(slope) < 1:
                    print("Slope: %3f, (x1,y1):(%3f,%3f), (x2,y2):(%3f,%3f)"%(slope,x1,y1,x2,y2))

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(debug_img, "circles", (10, 30), font, 1, (0, 255, 0))
                    # cv2.line(debug_img, (x1, y1), (x2, y2), color, thickness=2)
                    cv2.circle(debug_img, ((x2+x1)//2, (y2+y1)//2), 1, color, thickness=5)
                    plt.imsave("point.jpg",debug_img)

    # plt.imshow(debug_img), plt.show()

    x1_ls, y1_ls, x2_ls, y2_ls = np.mean([x1_ls, y1_ls, x2_ls, y2_ls], axis=1)
    slp_lt, intcpt_lt = getYInt(x1_ls, y1_ls, x2_ls, y2_ls)

    x1_rs, y1_rs, x2_rs, y2_rs = np.mean([x1_rs, y1_rs, x2_rs, y2_rs], axis=1)
    slp_rgt, intcpt_rgt = getYInt(x1_rs, y1_rs, x2_rs, y2_rs)

    cache["slp_lt_cache"].append(slp_lt)
    cache["slp_rgt_cache"].append(slp_rgt)
    cache["intcpt_lt_cache"].append(intcpt_lt)
    cache["intcpt_rgt_cache"].append(intcpt_rgt)

    mv_slp_lt = np.mean(cache["slp_lt_cache"][-n:])
    mv_slp_rgt = np.mean(cache["slp_rgt_cache"][-n:])
    mv_intcpt_lt = np.mean(cache["intcpt_lt_cache"][-n:])
    mv_intcpt_rgt = np.mean(cache["intcpt_rgt_cache"][-n:])



    # Draw left lines
    cv2.line(img, ((W//2-50), int((W//2-50) * mv_slp_lt + mv_intcpt_lt)), (int((H - mv_intcpt_lt) / mv_slp_lt), H),
             color,
             thickness)

    # Draw right lines
    cv2.line(img, ((W//2+50), int((W//2+50) * mv_slp_rgt + mv_intcpt_rgt)), (int((H - mv_intcpt_rgt) / mv_slp_rgt), H), color,
             thickness)

    if DEBUG:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Lt Slp %.3f,Rgt Slp %.3f" % (mv_slp_lt, mv_slp_rgt)
        cv2.putText(img, text, (10, 30), font, 1, (0, 255, 0))

    return img, cache


def hough_lines(img, cache, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, cache=cache, color=[255, 0, 0], thickness=5)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., _lambda=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, _lambda)


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
class VideoProcessor:
    def __init__(self):
        self.cache = {}
        self.cache["slp_lt_cache"] = []
        self.cache["slp_rgt_cache"] = []
        self.cache["intcpt_lt_cache"] = []
        self.cache["intcpt_rgt_cache"] = []

    def mv_line_tracking(self, image):
        # Retrieve cache
        cache = self.cache

        # Read image -> Grayscale
        gray_img = grayscale(image)

        # Canny
        blurred_img = gaussian_blur(gray_img, kernel_size=3)
        edges_img = canny(blurred_img, low_threshold=100, high_threshold=240)

        # Create ROI vertices
        imshape = image.shape

        vertices = np.array([[(100, imshape[0]), (imshape[1]/2-50, imshape[0]/2+30), (imshape[1]/2+50, imshape[0]/2+30), (imshape[1] - 100, imshape[0])]], dtype=np.int32)
        roi_img = region_of_interest(edges_img, vertices)

        # Hough Transform + Get left,right line slope and intercept
        line_img = hough_lines(roi_img, cache, rho=1, theta=np.pi / 180, threshold=15, min_line_len=20,max_line_gap=400)
        result = weighted_img(line_img, image, alpha=0.8, beta=1., _lambda=0.)

        return result


if __name__ == "__main__":
    img_collection = os.listdir("debug_frames/")
    test_imgdir = "debug_frames/" + img_collection[0]

    # img_collection = os.listdir("test_images/")
    # test_imgdir = "test_images/" + img_collection[0]


    for imgname in img_collection:
        inputdir = "debug_frames/" + imgname
        outdir = "test_images_output/" + imgname

        image = mpimg.imread(inputdir)
        vp = VideoProcessor()
        result = vp.mv_line_tracking(image)
        mpimg.imsave(outdir, result)