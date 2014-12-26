import cv2

def find_features(img, hessian_threshold=500):
    """Given a gray OpenCV image (such as from imread) return a list
    of OpenCV features and their descriptors as a tuple

    Params:
    img -- OpenCV image
    hessian_threshold -- threshold for feature selection
      (larger means fewer features)

    Returns:
    ([keypoints], [descriptors])
    """
    surf = cv2.SURF(400)
    kp, des = surf.detectAndCompute(img, None) # Second param is mask
    return (kp, des)

def display_features(img, kp):
    """Draw features onto image
    """
    kpimg = cv2.drawKeypoints(img, kp)
    cv2.imshow("Feature Keypoints", kpimg)
    cv2.waitKey(0)
    cv2.destroyWindow("Feature Keypoints")

img = cv2.imread("test.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
(kp, _) = find_features(gray)
display_features(gray, kp)
