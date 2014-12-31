"""Finds matching features and pickles matching information to files"""
import source
import detect
import numpy as np
import numpy.lib.scimath as npmath
import pickle
import cv2
from disjoint_set import DisjointSet
from matplotlib import pyplot as plt

def correlation(x, y):
    """x,y features. Uses ratio test"""
    matches = matcher.knnMatch(x[1], y[1], k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)

    # keepMatches = []
    # # ratio test as per Lowe's paper
    # for i,(m,n) in enumerate(matches):
    #     if m.distance < 0.7*n.distance:
    #         keepMatches.append((m, n))

    return len(good)

def show_groups():
    groups = pickle.load(open('groups.txt'))
    print groups
    print [len(g) for g in groups]
    for g in groups:
        for f in g:
            cv2.imshow(f, cv2.imread(f))
        cv2.waitKey(0)
        for f in g:
            cv2.destroyWindow(f)

if __name__ == "__main__":
    N = 50 # number of images to use
    # only need to download once:
    # source.download_images('yosemite', N, -119.583650, 37.720424, -119.563650, 37.740424)
    files = source.get_images('yosemite')[:N]
    features = [detect.get_features(f, sx=1, sy=1) for f in files]

    # set up FLANN matcher
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)
    # matcher = cv2.FlannBasedMatcher(index_params,search_params)

    # create BFMatcher
    matcher = cv2.BFMatcher(cv2.NORM_L2) # , crossCheck=True)

    grid = np.zeros((len(features), len(features)), 'int')
    matches = {} # {filename: (filename, correlation)}

    groups = DisjointSet(files)
    for i, f1 in enumerate(features):
        print("matching images with image", i)
        for j, f2 in enumerate(features):
            if i >= j: # do not double compute
                grid[i, j] = correlation(f1, f2)
                if i != j and grid[i, j] >= 5:
                    print(files[i], files[j])
                    matches.setdefault(files[i], []).append((files[j], grid[i, j]))
                    matches.setdefault(files[j], []).append((files[i], grid[i, j]))
                    groups.union(files[i], files[j])
    print(grid)

    pickle.dump(files, open('files.txt', 'w'))
    pickle.dump(matches, open('matches.txt', 'w'))
    pickle.dump(groups.get_sets(), open('groups.txt', 'w'))

    show_groups()
