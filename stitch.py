import cv2

def find_matching_features(features1, features2):
    """Given two lists of features in tuple form [(kp, desc)],
    return a pair of the best matching features

    Runtime:
    O(m * n) -- m is len(features1)
                n is len(features2)

    Params:
    features1 -- list of (kp, desc) pairs
    features2 -- list of (kp, desc) pairs

    Returns:
    ((kp, desc), (kp, desc))
    """
    kp1, desc1 = features1
    kp2, desc2 = features2

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    best_match = sorted(bf.match(desc1, desc2), key=lambda x: x.distance)[0]

    best1 = features1[best_match.queryIdx]
    best2 = features2[best_match.trainIdx]

    return (best1, best2)

def stitch_images(images):
    """Given a list of images and their corresponding features
    in tuple form [(image, (kp, desc))], return a stitched image
    """

    # TODO: find_matching_features() on all pairs of images
    # TODO: Calculate new image size
    # TODO: Use keypoints for translation stitching
