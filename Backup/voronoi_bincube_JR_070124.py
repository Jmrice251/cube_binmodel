from mpdaf.obj import Cube
import numpy as np
import matplotlib.pyplot as plt

from regions import Regions, PixCoord
# import astropy.units as u
import astropy.io.fits as fits
from astropy.wcs import WCS

import sys
import json
from tqdm import tqdm
# from joblib import Parallel, delayed

import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning

from scipy import spatial

# ----- from cmd line ----------------------
# arguments from cmd line
nargs = 2
if (len(sys.argv) < nargs):
    print(f'Usage: python3 {sys.argv[0]} <bin_info_file>')
    exit()

# open fit info json file
bin_info_fn = sys.argv[1]
bin_info_f = open(bin_info_fn, 'r')
bin_info_obj = json.load(bin_info_f)
bin_info_f.close()

# extract bin information
cube_fn = bin_info_obj['cube_filename']
regions_fn = bin_info_obj['regions_filename']
redshift = bin_info_obj['redshift']
center = bin_info_obj['center_wave']
margin = bin_info_obj['margin']
target_sn = bin_info_obj['target_sn']

# optional fields
# output average bin cube (very heavy)
output_cube_fn = bin_info_obj['output_cube_fn'] if 'output_cube_fn' in bin_info_obj else None
# output bin map for piexels
output_binmap_fn = bin_info_obj['output_binmap_fn'] if 'output_binmap_fn' in bin_info_obj else None

# ----------- READ CUBE & REGION ---------------------------------------------------------------------------------------------
print("Reading cube...")
cube = Cube(cube_fn).copy()      # do a copy here to fully load into memory, important for later
w = WCS(cube.get_wcs_header())
N_w, N_y, N_x = cube.shape

# read region list
sky_reg_list = Regions.read(regions_fn)             # list of regions
if (len(sky_reg_list) == 0):
    print("ERROR: no regions found")
    exit()
# form union of all regions
sky_reg = sky_reg_list[0]
for i in range(1, len(sky_reg_list)):
    sky_reg = sky_reg.union(sky_reg_list[i])
# convert to pixel region, using celestial WCS
pix_reg = sky_reg.to_pixel(w.celestial)

# list of (y,x) pixel coordinates lying inside the region
img_pix_xcoords = np.repeat(range(N_x), N_y)
img_pix_ycoords = np.tile(np.array(range(N_y)), N_x)
pix_coords = PixCoord(img_pix_xcoords, img_pix_ycoords)
inside = pix_reg.contains(pix_coords)
x_inside, y_inside = img_pix_xcoords[inside], img_pix_ycoords[inside]
inside_pix_coords = list(zip(y_inside, x_inside))
print("Done!")

# ------------------ AGGREGATE OVER WAVELENGTH RANGE ----------------------------
print("Aggregating data over wavelength range...")
# get range
range_min, range_max = center*(1+redshift) - margin, center*(1+redshift) + margin
range_min, range_max = max(range_min, cube.wave.get_start()), min(range_max, cube.wave.get_end())

# get metadata about the wave spectrum
CRVAL = cube.wave.get_crval()
CDELT = cube.wave.get_step()
lo_pix = int(np.floor((range_min - CRVAL) / CDELT))
hi_pix = int(np.ceil((range_max - CRVAL) / CDELT))
# x0=np.arange(hi_pix + 1 - lo_pix)
# wave=((x0+lo_pix)*CDELT + CRVAL)

# sum over range, along the wave axis
sum_image = cube[lo_pix:(hi_pix+1), :, :].sum(axis=0)

# corresponding signals and noises for each pixel inside region
signals = np.array([sum_image.data.data[coord] for coord in inside_pix_coords])
noises = np.sqrt(np.array([sum_image.var.data[coord] for coord in inside_pix_coords]))
print("Done!")

#--------------- VORONOI BINNING ------------------------------
print("Voronoi binning...")
# scale of pixel (i.e. distance between 2 closest pixels)
# pixscale = np.min(spatial.distance.pdist(np.column_stack([img_pix_xcoords[inside], img_pix_ycoords[inside]])))
# Voronoi binning
bin_number, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(
    x_inside, y_inside, signals, noises, target_sn, cvt=True, pixelsize=1, plot=False,
    quiet=True, sn_func=None, wvt=False)
print("Done!")

# -------------- CREATE THE AVERAGED BIN CUBE --------------------------
print("Create and write cube to file...")
n_bins = len(nPixels)    # number of bins
# map of the bins, np.nan is not in any bin / outside region
bin_array = np.full((N_y,N_x), -1, dtype=float)

# put pixels into bin
for i in range(len(inside_pix_coords)):
        bin_index, coord = bin_number[i], inside_pix_coords[i]
        bin_array[coord] = bin_index

if output_binmap_fn:
   np.savetxt(output_binmap_fn, bin_array, fmt="%d")

# if outputting a full average bin cube
if output_cube_fn:

    # array of the averaged signals and variances
    avg_signal_array = np.zeros((n_bins, N_w), dtype=np.float64)
    avg_variance_array = np.zeros((n_bins, N_w), dtype=np.float64)

    # add the spectrum from each pixel into their respective bin
    for i in tqdm(range(len(inside_pix_coords))):
        bin_index, coord = bin_number[i], inside_pix_coords[i]
        # bin_array[coord] = bin_index
        avg_signal_array[bin_index] += cube[:, coord[0], coord[1]].data.data
        avg_variance_array[bin_index] += cube[:, coord[0], coord[1]].var.data
    
    reshaped_nPixels = nPixels.reshape((n_bins,1))
    avg_signal_array = avg_signal_array / (reshaped_nPixels)             # signal for each bin
    avg_variance_array = avg_variance_array / (reshaped_nPixels**2)      # variance for each bin 

    # spectrum and variance map
    binned_specmap, binned_varmap = np.full((N_w, N_y, N_x), np.nan, dtype=np.float64), np.full((N_w, N_y, N_x), np.nan, dtype=np.float64)
    
    for y in tqdm(range(N_y)):
        for x in range(N_x):
            bin_id = int(bin_array[y,x])
            if bin_id > 0:           # if pixel is in a bin
                binned_specmap[:, y, x], binned_varmap[:, y, x] = avg_signal_array[bin_id], avg_variance_array[bin_id]
    
    # create and write to file
    avg_binned_cube = Cube(ext=('DATA', 'STAT'), wcs=cube.wcs, wave=cube.wave, unit=cube.unit, copy=False,
                          data=binned_specmap, var=binned_varmap)
    avg_binned_cube.write(output_cube_fn)
print("Done!")