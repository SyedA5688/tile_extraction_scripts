import pyvips
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from skimage.transform import resize
from numpy.random import RandomState
import cv2
from scipy import stats
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


def load_and_extract_tiles(filename, TISSUE_THRESHOLD_PERCENTAGE):
    # Loading large tiff using pyvips library
    img = pyvips.Image.new_from_file(filename)

    # Isolating image filenames now from filepath passed to function
    filename = filename.split("/")[-1]
    filename = filename.split(".tif")[0]
    mask_filename = filename.split("/")[-1]
    mask_filename = filename.split(".tif")[0]

    # Converting image to numpy array
    print("Script running on image", filename, "now.")
    print("Converting image to numpy array...")
    np_img = np.ndarray(buffer=img.write_to_memory(), dtype=format_to_dtype[img.format],
                        shape=[img.height, img.width, img.bands])[:, :, 0:3]  # Channels needs to be last dimension

    # Creating mask
    mask = np.zeros(np_img.shape[0:2], dtype=int)
    for mask_file in os.listdir('/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/whole_slide_masks/masks/'):
        if mask_file.find(filename) != -1:
            print("Loading annotation mask", mask_file, "to put onto mask for ", filename)
            sub_mask_img = pyvips.Image.new_from_file(mask_file)
            np_sub_mask = np.ndarray(buffer=sub_mask_img.write_to_memory(), dtype=format_to_dtype[sub_mask_img.format],
                                     shape=[sub_mask_img.height, sub_mask_img.width, sub_mask_img.bands])[:, :, 0:3]
            # Get x/y coord and width/height of image
            second_half_filename = mask_file.split("_")[-1]
            mask_spatial_info = second_half_filename.split("-mask")[0]
            mask_spatial_info = mask_spatial_info.split("(")[-1]
            mask_spatial_info = mask_spatial_info.split(")")[0]
            info_list = list(map(int, mask_spatial_info.split(",")))
            # info_list has [downsample_amt, xcoord, ycoord, width, height] of annotation mask
            # in overall mask, axis are (height, width, channels)
            # mask[ycoord:ycoord+height, xcoord:xcoord+width] = np_sub_mask
            mask[info_list[2]:info_list[2]+info_list[4], info_list[1]:info_list[1]+info_list[3]] = np_sub_mask
    print("Done building mask, dimensions are", mask.shape)


    print("Shape of whole image numpy array:", np_img.shape, '\n')
    print("Generating thumbnail...")
    idx = [1, 0]  # cv2 takes height and width in reverse order
    np_img_mini = cv2.resize(np_img, dsize=tuple((N * np.array(np_img.shape)[0:2][idx]).astype(int)))
    # Create PIL image from thumbnail to draw boxes for indicating where we saved tiles from
    thumbnail_img = Image.fromarray(np_img_mini)
    draw = ImageDraw.Draw(thumbnail_img)

    print('Entering loop, calculating tissue percentage and saving tiles > {}%'.format(100*TISSUE_THRESHOLD_PERCENTAGE))
    num_saved_tiles = 0
    # Start from disqualify border, go to WSI height minus 1 tile width (saving tiles) minus disqualify border again.
    # Increment by tile size
    for k in range(0, np_img.shape[0] - TILE_HEIGHT, TILE_HEIGHT):
        print("Looking on height:", k)
        i = 0
        # Percentage variable
        percentage = -1.0

        while i < (np_img.shape[1] - TILE_WIDTH):  # From 0 to width - tile_width
            # Look at 1 tile
            # tile = np_img[k:k+TILE_HEIGHT, i:i+TILE_WIDTH, :]
            tile_mask = mask[k:k + TILE_HEIGHT, i:i + TILE_WIDTH, :]
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
                        if tile_mask[grid_line, iterPt] == 1:
                            tissue_pixels += 1

                    if iterPt < TILE_HEIGHT and grid_line < TILE_WIDTH:
                        total_pixels += 1
                        # Pixel on vertical grid line; first is at 0 height, 50 right
                        if tile_mask[iterPt, grid_line] == 1:
                            tissue_pixels += 1

            # Calculate percentage of pixels that have tissue (in mask)
            previous_percentage = percentage # Save previous
            percentage = tissue_pixels / total_pixels # New percentage
            # print('Percentage of tissue pixels in tile_x{}_y{}:'.format(i, k), percentage)

            # See if percentage has increased or decreased
            if previous_percentage != -1.0 and percentage < previous_percentage:
                decreased = True
            else:
                decreased = False

            if percentage > TISSUE_THRESHOLD_PERCENTAGE and decreased:
                # Tile meets threshold and % has dec, indicating this is best tile at this height will be 50 spaces back
                # Changing tile from 50 spaces back to contiguous memory to be saved by matplotlib
                tile1_contig = np.ascontiguousarray(np_img[k:k+TILE_HEIGHT, i-50:i+TILE_WIDTH-50, :])

                # Saving image as png
                print('tile_x{}_y{} has enough tissue ({}%), saving tile.\n'.format(i-50, k, round(100*percentage, 2)))
                #plt.imsave('/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/TI_crops_set_a_v5/{}_x{}_y{}_{}.png'
                #           .format(filename, i, k, round(100*percentage, 2)), tile1_contig)
                draw.rectangle([math.floor(N*(i-50)), math.floor(N*k),
                                math.floor(N*(i+TILE_WIDTH-50)), math.floor(N*(k+TILE_HEIGHT))], outline="red", width=3)

                # Syed testing on noisy samples, real save line above
                plt.imsave('/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/TI_noise_debugging_Syed/{}_x{}_y{}_{}.png'
                           .format(filename, i, k, round(100 * percentage, 2)), tile1_contig)
                num_saved_tiles += 1
                percentage = -1.0
                i += TILE_WIDTH
            elif percentage > TISSUE_THRESHOLD_PERCENTAGE and not decreased:
                # Percentage is over threshold but % is still growing, keep looking 50 px right
                i += 50
            elif percentage < 0.2:
                i += TILE_WIDTH  # If less than 20% pixels had tissue, skip ahead by tile width to save computations
            else:
                i += 50  # Check next tile frame 50 to the right to look for enough tissue

    if num_saved_tiles < 8:  # One WSI has only 8 possible tiles - minimum
        print("Extracted less than 5 tiles from image. Rerunning script with lower threshold")
        # Delete generated images.
        #  delete_files('/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/TI_crops_set_a_v5/', filename)
        delete_files('/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/TI_noise_debugging_Syed/', filename)
        print()

        # Rerun script with threshold 5% lower
        load_and_extract_tiles('/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/AC_set_a/' + filename +
                               '.tif', TISSUE_THRESHOLD_PERCENTAGE - .05)
    else:
        # Save thumbnail image with boxes indicating saved tiles to folder
        thumbnail_img.save('/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/TI_crops_set_a_v5/thumbnails/{}_thumbnail.png'
                           .format(filename))
        print("Saved", num_saved_tiles, "tiles from", filename, ". Done, thumbnail photo in subfolder.\n")


if __name__ == "__main__":
    # Noisy samples: Unknown(10.7.19)_2214_100720190115.tif and Unknown(10.7.19)_2246_100720190313.tif
    # and Unknown(10.7.19)_2197_100720191217.tif
    # Samples needing cropping: MJK_2229_100720190202.tif and MJK_2234_100720190211.tif
    # and Unknown(10.7.19)_2197_100720191217.tif
    wsi_filename = '/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/AC_set_a/MJK_2234_100720190211.tif'
    load_and_extract_tiles(wsi_filename, .75)  # Pass tissue threshold percentage as a decimal #** Should be 85
