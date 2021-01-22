import pyvips
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from skimage.transform import resize
from numpy.random import RandomState
import cv2
from scipy import stats
import statistics
from kneed import KneeLocator
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw
import os
import math
from delete_tiles_by_filename import delete_files

rng = RandomState(42)  # Reproducibility
TILE_HEIGHT = 2048
TILE_WIDTH = 2048
GRID_INCREMENT = min(TILE_WIDTH, TILE_HEIGHT) // 20
N = 0.05  # Thumbnail scaling factor
DISQUALIFY_BORDER = 3000  # Don't accept tiles from edge border of WSI. This number is border to skip

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

def optimalK(data, nrefs=3, maxClusters=3, minClusters=2):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(minClusters, maxClusters+1)),))
    kmeans = []
    for gap_index, k in enumerate(range(minClusters, maxClusters+1)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            print('Computing', k, 'clusters on ref', i, '...')
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        print('Computing', k, 'clusters on input ...')
        km = KMeans(k)
        km.fit(data)
        kmeans.append(km)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

    return (gaps.argmax() + minClusters, kmeans, gaps)

def load_and_extract_tiles(filename, TISSUE_THRESHOLD_PERCENTAGE):
    # Loading large tiff using pyvips library
    img = pyvips.Image.new_from_file(filename)
    # print("Image Width:", img.width, "Image Height:", img.height, "Image Bands:", img.bands,
    #       "Image Format:", img.format, '\n')

    # Isolating image filename now from filepath passed to function
    filename = filename.split("/")[-1]  # For pietro's filepath
    filename = filename.split(".tif")[0]

    # Converting image to numpy array
    print("Script running on image", filename, "now.")
    print("Converting image to numpy array...")
    # Some samples need cropping
    if filename == "MJK_2229_100720190202":
        np_img = np.ndarray(buffer=img.write_to_memory(), dtype=format_to_dtype[img.format],
                            shape=[img.height, img.width, img.bands])[:107000, :, 0:3]
    elif filename == "MJK_2234_100720190211":
        np_img = np.ndarray(buffer=img.write_to_memory(), dtype=format_to_dtype[img.format],
                            shape=[img.height, img.width, img.bands])[20000:, :, 0:3]
    elif filename == "Unknown(10.7.19)_2197_100720191217":
        np_img = np.ndarray(buffer=img.write_to_memory(), dtype=format_to_dtype[img.format],
                            shape=[img.height, img.width, img.bands])[:, :32000, 0:3]
    else:
        np_img = np.ndarray(buffer=img.write_to_memory(), dtype=format_to_dtype[img.format],
                            shape=[img.height, img.width, img.bands])[:, :, 0:3]  # Channels needs to be last dimension
    TOP_STD = 3.0
    BOT_STD = 3.0

    print("Shape of whole image numpy array:", np_img.shape, '\n')
    print("Generating thumbnail...")
    idx = [1, 0]  # cv2 takes height and width in reverse order
    np_img_mini = cv2.resize(np_img, dsize=tuple((N * np.array(np_img.shape)[0:2][idx]).astype(int)))
    # Create PIL image from thumbnail to draw boxes for indicating where we saved tiles from
    thumbnailImg = Image.fromarray(np_img_mini)
    draw = ImageDraw.Draw(thumbnailImg)

    np_img_mini = np_img_mini.reshape(np_img_mini.shape[0] * np_img_mini.shape[1], 3)

    # print("Identifying Gap-Statistic ideal K-Means cluster count...")
    # k, kmeans, gaps = optimalK(np_img_mini, nrefs=2, maxClusters=3, minClusters=2)
    print('Generating input clusters...')
    k = 2
    kmeans = []
    km = KMeans(k, random_state=rng)
    kmeans.append(km.fit(np_img_mini))

    print("Ran with k equal to 2")
    # TISSUE_THRESHOLD_PERCENTAGE pass as parameter to function
    # print("Optimal Cluster Count:", k, " (Clean Sample, Tissue cleanly distinguished from background)")

    # tissue cluster
    tissue = 0 if (np.mean(kmeans[0].cluster_centers_[0]) < np.mean(kmeans[0].cluster_centers_[1])) else 1
    tissue_c0 = stats.describe(list(np_img_mini[kmeans[0].labels_ == tissue][:, 0]))
    tissue_c1 = stats.describe(list(np_img_mini[kmeans[0].labels_ == tissue][:, 1]))
    tissue_c2 = stats.describe(list(np_img_mini[kmeans[0].labels_ == tissue][:, 2]))

    # noise cluster
    noise_c0 = stats.describe(list(np_img_mini[kmeans[0].labels_ == 1 - tissue][:, 0]))
    noise_c1 = stats.describe(list(np_img_mini[kmeans[0].labels_ == 1 - tissue][:, 1]))
    noise_c2 = stats.describe(list(np_img_mini[kmeans[0].labels_ == 1 - tissue][:, 2]))
    # Syed - changing cutoff from  "tissue_c0[2] + TOP_STD * (np.sqrt(tissue_c0[3]))"  to 2.5 standard deviations
    # below mean of noise distribution
    c0_cut_top = noise_c0[2] - 2.5 * np.sqrt(noise_c0[3])  # mean - 3 standard devs
    c1_cut_top = noise_c1[2] - 2.5 * np.sqrt(noise_c1[3])
    c2_cut_top = noise_c2[2] - 2.5 * np.sqrt(noise_c2[3])
    c0_cut_bot = 0
    c1_cut_bot = 0
    c2_cut_bot = 0

    print("Tissue pixel thresholds: ", [c0_cut_bot, c0_cut_top], [c1_cut_bot, c1_cut_top], [c2_cut_bot, c2_cut_top])

    # rgb_white_pixels = np_img[0:10, 0:10, 0:3]
    # threshold = np.min(rgb_white_pixels) - 3  # Minimum of the values
    # print("Threshold for pixel decisions (close to 255 is white):", threshold, '\n')

    print('Entering loop, calculating tissue percentage and saving tiles > {}%'.format(100*TISSUE_THRESHOLD_PERCENTAGE))
    num_saved_tiles = 0
    # Start from disqualify border, go to WSI height minus 1 tile width (saving tiles) minus disqualify border again.
    # Increment by tile size
    for k in range(DISQUALIFY_BORDER, np_img.shape[0] - TILE_HEIGHT - DISQUALIFY_BORDER, TILE_HEIGHT):
        print("Looking on height:", k)
        i = DISQUALIFY_BORDER
        # Percentage variable
        percentage = -1.0

        while i < (np_img.shape[1] - TILE_WIDTH - DISQUALIFY_BORDER):  # From 0 to width, inc by 50 at end
            # Extracting 1 tile
            tile = np_img[k:k+TILE_HEIGHT, i:i+TILE_WIDTH, :]
            # print("Tile shape: ", tile.shape)

            # Calculating tissue pixel percentage by looping through gridlines on image and
            total_pixels = 0
            tissue_pixels = 0
            max_dim = max(TILE_WIDTH, TILE_HEIGHT)

            for grid_line in range(GRID_INCREMENT, max_dim, GRID_INCREMENT):  # 50 up to max_dim, inc by 50
                for iterPt in range(0, max_dim):  # 0 up to max_dim, every pixel on line
                    # Looking at 2 pixels every iteration of inner loop (vertical and horiz grid line)
                    # Calculate pixel's avg value across 3 channels (RGB), increment tissue pixel
                    # count if average is less than white pixel threshold
                    if grid_line < TILE_HEIGHT and iterPt < TILE_WIDTH:
                        total_pixels += 1
                        # Pixel on horizontal grid line; First one at 50 height, 0 right. (height, width, bands)
                        if (((tile[grid_line, iterPt, 0] < c0_cut_top) and (tile[grid_line, iterPt, 0] > c0_cut_bot)) and ((tile[grid_line, iterPt, 1] < c1_cut_top) and (tile[grid_line, iterPt, 1] > c1_cut_bot)) and ((tile[grid_line, iterPt, 2] < c2_cut_top) and (tile[grid_line, iterPt, 2] > c2_cut_bot))):
                            tissue_pixels += 1

                    if iterPt < TILE_HEIGHT and grid_line < TILE_WIDTH:
                        total_pixels += 1
                        # Pixel on vertical grid line; first is at 0 height, 50 right
                        if (((tile[iterPt, grid_line, 0] < c0_cut_top) and (tile[iterPt, grid_line, 0] > c0_cut_bot)) and ((tile[iterPt, grid_line, 1] < c1_cut_top) and (tile[iterPt, grid_line, 1] > c1_cut_bot)) and ((tile[iterPt, grid_line, 2] < c2_cut_top) and (tile[iterPt, grid_line, 2] > c2_cut_bot))):
                            tissue_pixels += 1

            # Calculate percentage of pixels that have tissue
            previous_percentage = percentage # Save previous
            percentage = tissue_pixels / total_pixels # New percentage
            # print('Percentage of tissue pixels in tile_x{}_y{}:'.format(i, k), percentage)

            # See if percentage has increased or decreased
            if previous_percentage != -1.0 and percentage < previous_percentage:
                decreased = True
            else:
                decreased = False

            if percentage > TISSUE_THRESHOLD_PERCENTAGE and decreased:
                # Tile meets threshold and % has dec, indicating this is best tile at this height in image
                # Best tile for this height will be 50 spaces back, save this tile

                # Changing tile from 50 spaces back to contiguous memory to be saved by matplotlib
                tile1_contig = np.ascontiguousarray(np_img[k:k+TILE_HEIGHT, i-50:i+TILE_WIDTH-50, :])

                # Saving image as png
                print('tile_x{}_y{} has enough tissue ({}%), saving to png.\n'.format(i, k, round(100*percentage, 2)))
                plt.imsave('/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/TI_crops_set_a_v4/{}_x{}_y{}_{}.png'.format(filename, i, k, round(100*percentage, 2)), tile1_contig)
                draw.rectangle([math.floor(N*(i-50)), math.floor(N*k),
                                math.floor(N*(i+TILE_WIDTH-50)), math.floor(N*(k+TILE_HEIGHT))], outline="red", width=3)

                #  Syed testing on noisy samples, real save line above
                #  plt.imsave('/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/TI_noise_debugging_Syed/{}_x{}_y{}_{}.png'.format(filename, i, k, round(100 * percentage, 2)), tile1_contig)
                num_saved_tiles += 1
                percentage = -1.0

                # Increment i by TILE_WIDTH so that there is no overlap in image. Just saved 1 tile
                i += TILE_WIDTH
            elif percentage > TISSUE_THRESHOLD_PERCENTAGE and not decreased:
                # Percentage is over threshold but % is still growing, keep looking 50 px right
                i += 50
            elif percentage < 0.2:
                i += TILE_WIDTH  # If less than 20% pixels had tissue, skip ahead by tile width to save computations
            else:
                i += 50  # Check next tile frame 50 to the right to look for enough tissue

    if (num_saved_tiles < 8):  # One WSI has only 8 possible tiles - minimum
        print("Extracted less than 5 tiles from image. Rerunning script with lower threshold")
        # Delete generated images. ***Important: Change directory to wherever tiles are being saved, otherwise
        # error might be thrown
        delete_files('/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/TI_crops_set_a_v4/', filename)
        #  delete_files('/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/TI_noise_debugging_Syed/', filename)
        print()

        # Rerun script with threshold 5% lower
        load_and_extract_tiles('/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/AC_set_a/' + filename + '.tif', TISSUE_THRESHOLD_PERCENTAGE - .05)
    else:
        # Save thumbnail image with boxes indicating saved tiles to folder
        thumbnailImg.save('/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/TI_crops_set_a_v4/thumbnails/{}_thumbnail.png'.format(filename))
        print("Saved", num_saved_tiles, "tiles from", filename, ". Done, thumbnail photo in subfolder.\n")


if __name__ == "__main__":
    # Noisy samples: Unknown(10.7.19)_2214_100720190115.tif and Unknown(10.7.19)_2246_100720190313.tif
    # and Unknown(10.7.19)_2197_100720191217.tif
    # Samples needing cropping: MJK_2229_100720190202.tif and MJK_2234_100720190211.tif
    # and Unknown(10.7.19)_2197_100720191217.tif
    wsi_filename = '/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/AC_set_a/MJK_2234_100720190211.tif'
    load_and_extract_tiles(wsi_filename, .75)  # Pass tissue threshold percentage as a decimal #** Should be 85
