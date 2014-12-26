"""Stitches images together"""
import source
import feature_detect

if __name__ == "__main__":
    # only need to download once:
    # files = source.download_images('campanile', 100, -122.261434, 37.870816, -122.257434, 37.874816)
    files = source.get_images('campanile')
    features = [feature_detect.get_features(f) for f in files]
    print("found features")

