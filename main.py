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
gradiant = heatMap.G3.generate_gradiant(GRADIANT_SIZE)
gradiant = gradiant.reshape((-1, 3))


def extract_points_mean(mask):
    ys, xs = np.nonzero(mask)
    ys_sum_per_xs = np.bincount(xs, ys)
    ys_count_per_xs = np.bincount(xs, np.ones(ys.shape, dtype=np.int32))

    exist_points = ys_count_per_xs > 0
    mean_y = ys_sum_per_xs[exist_points] / ys_count_per_xs[exist_points]

    xs = np.argwhere(exist_points)

    mean_y = np.expand_dims(mean_y, axis=-1)

    mean_pts = np.hstack((xs, mean_y)).astype(np.int32)
    return mean_pts


def pts2img(pts, shape):
    pts = pts.astype(np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[pts[:, 1], pts[:, 0]] = 255
    return mask


def perspective_correction(pts, angle):
    pts[:, 1] = pts[:, 1] / np.cos(np.deg2rad(angle))
    pts = pts.astype(np.int32)
    return pts


def linregress(pts):
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

cam = camera_connection.Collector("23287291")
cam.start_grabbing()
while True:
    img = cam.getPictures()
    cv2.imshow("img", img)
    cv2.waitKey(0)

for frame_idx, fname in enumerate(os.listdir(image_path)):
    fpath = os.path.join(image_path, fname)
    img = cv2.imread(fpath, 0)
    # -----------------------------------------------------------------------5
    t = time.time()
    # -----------------------------------------------------------------------
    img = cv2.blur(img, (5, 1))
    pts = ConvayerBase.extract_points(img, thresh=50, perspective_angle=60)
    # _,thresh_img = cv2.threshold( img, 50, 255, cv2.THRESH_BINARY )
    ##thresh_img = cv2.erode(thresh_img, np.ones((1,3)))

    # pts = extract_points_mean(thresh_img)
    # pts = perspective_correction(pts, 60)

    # -----------------------------------------------------------------------
    img1 = pts2img(pts, (400, 640))
    res_y = ConvayerBase.moving_avrage(pts[:, 1], 10)
    pts[: res_y.shape[0], 1] = res_y
    img2 = pts2img(pts, (400, 640))

    # -----------------------------------------------------------------------
    slope, intercept = linregress(pts)
    # slope, intercept, r, p, std_err = stats.linregress(pts[:,0], pts[:,1])

    good_y = (pts[:, 0] * slope + intercept).astype(np.int32)
    error_y = pts[:, 1] - good_y
    error_y = error_y / MAX_ERROR * GRADIANT_SIZE // 2
    # -----------------------------------------------------------------------
    step = 5
    if frame_idx > res.shape[0] // step - step - 1:
        frame_idx = res.shape[0] // step - step - 1
        res[:-step] = res[step:]

    res[frame_idx * step : frame_idx * step + step, pts[:, 0]] = gradiant[
        np.clip(error_y + GRADIANT_SIZE // 2, 0, GRADIANT_SIZE - 1).astype(np.int32)
    ]
    # -----------------------------------------------------------------------
    t = time.time() - t
    print(t)
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.imshow("img", img)
    cv2.imshow("res", res)
    cv2.waitKey(10)
    t_mean += t

t_mean /= frame_idx - 1
print("t mean", t_mean)
cv2.waitKey(5)
