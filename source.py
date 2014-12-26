"""Filesystem and online API interactions to grab images"""

import os
import urllib, urlparse
import json

def download_images(name, n = 20, minx=-77.037564, miny=38.896662, maxx=-77.035564, maxy=38.898662):
    """This Panoramio image download code adapted from Jan Erik Solem
    Stores the images to folder img/{name}
    Arguments are longitudes (x) and latitudes (y), default is white house.
    """

    # query for images
    url = 'http://www.panoramio.com/map/get_panoramas.php?order=popularity&set=public&'
    args = 'from=0&to=%d&minx=%f&miny=%f&maxx=%f&maxy=%f&size=medium' % (n, minx, miny, maxx, maxy)
    c = urllib.urlopen(url + args)

    # get the urls of individual images from JSON
    j = json.loads(c.read())
    imurls = []
    for im in j['photos']:
        imurls.append(im['photo_file_url'])

    # ensure directory exists
    dir = "img/" + name + "/"
    d = os.path.dirname(dir)
    if not os.path.exists(d):
        os.makedirs(d)

    # download images
    for url in imurls:
        image = urllib.URLopener()
        image.retrieve(url, dir + os.path.basename(urlparse.urlparse(url).path))
        print 'downloading:', url

if __name__ == "__main__":
    # download_images('washington', 50, minx=-77.037564, miny=38.896662, maxx=-77.035564, maxy=38.898662)
    download_images('campanile', 50, -122.260434, 37.871816, -122.258434, 37.873816)