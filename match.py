"""Stitches images together"""
import source
import detect
import numpy as np
import numpy.lib.scimath as npmath
import pickle
import cv2

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
    # only need to download once:
    # source.download_images('yosemite', 200, -119.583650, 37.720424, -119.563650, 37.740424)
    files = source.get_images('yosemite')[:20]
    features = [detect.get_features(f) for f in files]

    norm_features = []
    for f in features:
        f_normalized = np.array([v/np.linalg.norm(v) for v in f[1]])
        norm_features.append((f_normalized, f_normalized.T))

    grid = np.zeros((len(features), len(features)), 'int')
    matches = {} # {filename: (filename, correlation)}

    group_map = {} # {filename: group_index}
    groups = [] # [{filename1, filename2}, {}]
    for i, f1 in enumerate(norm_features):
        print("matching images with image", i)
        for j, f2 in enumerate(norm_features):
            if i >= j: # do not double compute
                grid[i, j] = correlation(f1, f2)
                if i != j and grid[i, j] >= 2:
                    print(files[i], files[j])
                    matches.setdefault(files[i], []).append((files[j], grid[i, j]))
                    matches.setdefault(files[j], []).append((files[i], grid[i, j]))
                    index = None
                    if files[i] in group_map:
                        index = group_map[files[i]]
                    elif files[j] in group_map:
                        index = group_map[files[j]]
                    if index:
                        groups[index].add(files[i])
                        groups[index].add(files[j])
                    else:
                        groups.append(set([files[i], files[j]]))
                        group_map[files[i]] = len(groups) - 1
                        group_map[files[j]] = len(groups) - 1

    pickle.dump(files, open('files.txt', 'w'))
    pickle.dump(matches, open('matches.txt', 'w'))
    pickle.dump(groups, open('groups.txt', 'w'))
    pickle.dump(group_map, open('group_map.txt', 'w'))

    show_groups()
