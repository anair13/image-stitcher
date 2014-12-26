"""Stitches images together"""
import source
import feature_detect
import numpy as np
import numpy.lib.scimath as npmath

def match_oneway(features_1, features_2):
    """One way descriptor matching image f1 to f2, adapted from Solem"""
    f1 = np.array([v/np.linalg.norm(v) for v in features_1])
    f2 = np.array([v/np.linalg.norm(v) for v in features_2]).T

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
    matches_12 = match_oneway(features_1, features_2)
    matches_21 = match_oneway(features_2, features_1)

    index_12 = matches_12.nonzero()[0]

    for n in index_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0 # zero if not symmetric

    return matches_12

def correlation(features_1, features_2):
    return sum(match(features_1, features_2) > 0)

if __name__ == "__main__":
    # only need to download once:
    # files = source.download_images('campanile', 100, -122.261434, 37.870816, -122.257434, 37.874816)
    files = source.get_images('campanile')
    features = [feature_detect.get_features(f) for f in files[:20]]

    grid = np.zeros((len(features), len(features)), 'int')
    for i, f1 in enumerate(features):
        print("matching images with image", i)
        for j, f2 in enumerate(features):
            grid[i, j] = correlation(f1[1], f2[1])

    print(grid)