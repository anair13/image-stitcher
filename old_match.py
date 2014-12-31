"""Finds matching features and pickles matching information to files"""
import source
import detect
import numpy as np
import numpy.lib.scimath as npmath
import pickle
import cv2
from disjoint_set import DisjointSet
from matplotlib import pyplot as plt
#
# The code below reinvents the wheel, now we use OpenCV library functions
#


def match_oneway(features_1, features_2):
    """One way descriptor matching image f1 to f2, adapted from Solem"""
    f1 = features_1
    f2 = features_2

    ratio = 0.6
    size = f1.shape[0]

    scores = np.zeros((size, 1), 'int')

    for i in range(size):
        product = 0.9999 * np.dot(f1[i, :], f2)
        cosines = npmath.arccos(product)
        index = np.argsort(cosines)

        if cosines[index[0]] < ratio * cosines[index[1]]:
            scores[i] = int(index[0])

    return scores


def match(features_1, features_2):
    """Computes two way matches, removes matches that are not symmetric"""
    matches_12 = match_oneway(features_1[0], features_2[1])
    matches_21 = match_oneway(features_2[0], features_1[1])

    index_12 = matches_12.nonzero()[0]

    for n in index_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0 # zero if not symmetric

    return matches_12


def correlation(features_1, features_2):
    return sum(match(features_1, features_2) > 0)


def show_groups():
    groups = pickle.load(open('groups.txt'))
    for g in groups:
        for f in g:
            cv2.imshow(f, cv2.imread(f))
        cv2.waitKey(0)
        for f in g:
            cv2.destroyWindow(f)


if __name__ == "__main__":
    N = 20

    # only need to download once:
    # source.download_images('yosemite', N, -119.583650, 37.720424, -119.563650, 37.740424)
    files = source.get_images('yosemite')[:N]
    features = [detect.get_features(f, sx=0.5, sy=0.5) for f in files]

    norm_features = []
    for f in features:
        f_normalized = np.array([v/np.linalg.norm(v) for v in f[1]])
        norm_features.append((f_normalized, f_normalized.T))

    grid = np.zeros((len(features), len(features)), 'int')
    matches = {} # {filename: (filename, correlation)}

    groups = DisjointSet(files)
    for i, f1 in enumerate(norm_features):
        print("matching images with image", i)
        for j, f2 in enumerate(norm_features):
            if i >= j: # do not double compute
                grid[i, j] = correlation(f1, f2)
                if i != j and grid[i, j] >= 2:
                    print(files[i], files[j])
                    matches.setdefault(files[i], []).append((files[j], grid[i, j]))
                    matches.setdefault(files[j], []).append((files[i], grid[i, j]))
                    groups.union(files[i], files[j])

    pickle.dump(files, open('files.txt', 'w'))
    pickle.dump(matches, open('matches.txt', 'w'))
    pickle.dump(groups.get_sets(), open('groups.txt', 'w'))

    show_groups()
