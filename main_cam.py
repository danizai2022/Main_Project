import cv2
import numpy as np
import os
from scipy import stats
import heatMap
import time
import ConvayerBase
import camera_connection

GRADIANT_SIZE = 100
MAX_ERROR = 20
gradiant = heatMap.G2.generate_gradiant(GRADIANT_SIZE)
gradiant = gradiant.reshape((-1, 3))
defect_image_list = {}


def pts2img(pts, shape):
    pts = pts.astype(np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[pts[:, 1], pts[:, 0]] = 255
    return mask


def perspective_correction(pts, angle):
    pts[:, 1] = pts[:, 1] / np.cos(np.deg2rad(angle))
    pts = pts.astype(np.int32)
    return pts


def linearregression(pts):
    x = np.expand_dims(pts[:, 0], axis=-1)
    y = np.expand_dims(pts[:, 1], axis=-1)

    featurs = np.hstack((np.ones_like(x), x))  # , x**2))

    theta = np.linalg.inv(np.matmul(featurs.transpose(), featurs))
    theta = np.matmul(theta, featurs.transpose())
    theta = np.matmul(theta, y)
    slope = theta[1]
    intercept = theta[0]
    return slope, intercept


image_path = "image"
res = np.zeros((500, 640, 3), dtype=np.uint8)
t_mean = 0
frame_idx = 0
cam = camera_connection.Collector("23287291")
cam.start_grabbing()
fps = 0


def CreateContour(
    res, frame_idx2
):  # This function is used to create the contour from the stored frame as well as extract the characteristic of the defect like height, width, area and perimeter
    frame_idx2 = frame_idx2 + 1
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.erode(thresh_img, np.ones((3, 3)), iterations=1)
    thresh_img = cv2.dilate(thresh_img, np.ones((3, 3)), iterations=3)
    thresh_img = cv2.erode(thresh_img, np.ones((3, 3)), iterations=2)
    contours, hierarchy = cv2.findContours(
        thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        if h > 0:
            rr = cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 5)

        default_details = {
            ### "depth": "{:.2f}".format(depth1 * 0.01),
            "height": "{:.2f}".format(0.05 * h),
            "width": "{:.2f}".format(0.03 * w),
            "area": "{:.2f}".format(area * 0.001),
            "perimeter": "{:.2f}".format(perimeter * 0.037),
        }

        defect_image_list.append_mylist(default_details)

    return defect_image_list


while True:
    ret, img = cam.getPictures()
    if not ret:
        continue
    frame_idx += 1
    fps += 1
    # -----------------------------------------------------------------------5
    t = time.time()
    # -----------------------------------------------------------------------
    img = cv2.blur(img, (5, 1))
    print(img.shape)
    pts = ConvayerBase.extract_points(
        img, thresh=50, perspective_angle=70
    )  # This function is used to extract points from the image

    res_y = ConvayerBase.moving_avrage(
        pts[:, 1], 10
    )  # This function is used to apply the moving average function as a filter to remove any noise from the pts[:, 1] as well as smooth the pts[:, 1]
    pts[: res_y.shape[0], 1] = res_y

    # -----------------------------------------------------------------------
    slope, intercept = linearregression(
        pts
    )  # This function is used to apply linearregression to the extracted point in order to extract the slop as well as an intercept of the extracted point "pts"

    good_y = (pts[:, 0] * slope + intercept).astype(
        np.int32
    )  # This is a line which is extracted from the linearregression function
    error_y = (
        pts[:, 1] - good_y
    )  # This is a difference between the extracted point pts[:, 1] and the line extracted from the linearregression
    error_y = error_y / MAX_ERROR * GRADIANT_SIZE // 2

    # -----------------------------------------------------------------------
    step = 2
    if frame_idx > res.shape[0] // step - step - 1:
        frame_idx = res.shape[0] // step - step - 1
        res[:-step] = res[step:]

    res[frame_idx * step : frame_idx * step + step, pts[:, 0]] = gradiant[
        np.clip(error_y + GRADIANT_SIZE // 2, 0, GRADIANT_SIZE - 1).astype(np.int32)
    ]  # This function is used to stroe the frame in res array in order to find the contour of defect in image.
    # -----------------------------------------------------------------------

    t = time.time() - t
    t_mean += t
    if t_mean > 1:
        print(
            fps,
            t_mean,
        )
        t_mean = 0
        fps = 0

    cv2.imshow("res", res)
    cv2.waitKey(1)


t_mean /= frame_idx - 1
print("t mean", t_mean)
cv2.waitKey(5)
