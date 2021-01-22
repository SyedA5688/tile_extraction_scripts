# Script to run extract_tiles.py for each image in current directory
import os
import extract_tiles


def extract_tiles_from_all_tifs():
    for filename in os.listdir('/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/AC_set_a/'):
        if filename.endswith(".tif"):
            # print(filename)
            # Run with threshold of 85%, noise was getting in around 82%
            extract_tiles.load_and_extract_tiles('/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/AC_set_a/' + filename, .75)

if __name__ == "__main__":
    extract_tiles_from_all_tifs()
