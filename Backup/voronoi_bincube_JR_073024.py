from mpdaf.obj import Cube, WCS, WaveCoord, Spectrum
import mpdaf
import numpy as np
import matplotlib.pyplot as plt

from regions import Regions, PixCoord
import astropy.units as u
import astropy.io.fits as fits
from astropy.wcs import WCS

import gc
import sys
import json
from tqdm import tqdm
from joblib import Parallel, delayed

import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning

#from scipy import spatial

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
# output bin map for pixels
output_binmap_fn = bin_info_obj['output_binmap_fn'] if 'output_binmap_fn' in bin_info_obj else None
# output bins as fits file
output_fits_fn = bin_info_obj['output_fits_fn'] if 'output_fits_fn' in bin_info_obj else None

output_spec_fn = bin_info_obj['output_spec_fn'] if 'output_spec_fn' in bin_info_obj else None


# ----------- READ CUBE & REGION ---------------------------------------------------------------------------------------------
print("Reading cube...")
cube = Cube(cube_fn).copy()      # do a copy here to fully load into memory, important for later
wcs_header = cube.get_wcs_header()
w = WCS(wcs_header)
N_w, N_y, N_x = cube.shape

# get range
range_min, range_max = center*(1+redshift) - margin, center*(1+redshift) + margin
range_min, range_max = max(range_min, cube.wave.get_start()), min(range_max, cube.wave.get_end())

# get metadata about the wave spectrum
CRVAL = cube.wave.get_crval()
CDELT = cube.wave.get_step()
lo_pix = int(np.floor((range_min - CRVAL) / CDELT))
hi_pix = int(np.ceil((range_max - CRVAL) / CDELT))

# sum over range, along the wave axis
sum_image = cube[lo_pix:(hi_pix+1), :, :].sum(axis=0)

if output_cube_fn:
    # get cube data necessary to write binned cube
    cube_data = cube.data.data
    cube_var = cube.var.data
    cube_wcs = cube.wcs
    cube_wave = cube.wave
    cube_unit = cube.unit

if output_spec_fn:
    cube_center = cube.wcs.get_center(unit=u.Unit("deg"))
    cube_range = cube.wcs.get_range(unit=u.Unit("arcsec"))
    cube_radius = np.sqrt(((cube_range[2]-cube_range[0])**2)+((cube_range[3]-cube_range[1])**2))/2
# delete the cube object to free up memory
del cube
gc.collect()

# list of all pixel coordinates in the cube
img_pix_xcoords = np.repeat(range(N_x), N_y)
img_pix_ycoords = np.tile(np.array(range(N_y)), N_x)
pix_coords = PixCoord(img_pix_xcoords, img_pix_ycoords)
# defines inside the region if binning the whole cube
if regions_fn == "None":
    inside = [True for i in range(len(pix_coords))]
# read in the region list
else: 
    sky_reg_list = Regions.read(regions_fn)
    if (len(sky_reg_list) == 0):
        print("ERROR: no regions found")
        exit()  
# form union of all regions
    sky_reg = sky_reg_list[0]
    for i in range(1, len(sky_reg_list)):
        sky_reg = sky_reg.union(sky_reg_list[i])
# convert to pixel region, using celestial WCS
    pix_reg = sky_reg.to_pixel(w.celestial)
# defines inside if regions are masked
    inside = pix_reg.contains(pix_coords)
# list of (y,x) pixel coordinates lying inside the region    
x_inside, y_inside = img_pix_xcoords[inside], img_pix_ycoords[inside]
inside_pix_coords = list(zip(y_inside, x_inside))

print("Done!")

# ------------------ AGGREGATE OVER WAVELENGTH RANGE ----------------------------
print("Aggregating data over wavelength range...")

# corresponding signals and noises for each pixel inside region
signals = np.array([sum_image.data.data[coord] for coord in inside_pix_coords])
noises = np.sqrt(np.abs(np.array([sum_image.var.data[coord] for coord in inside_pix_coords])))

# remove pixels where there is no data if no region was given
if regions_fn == "None":
    noises, signals, x_inside, y_inside, i = noises.tolist(), signals.tolist(), x_inside.tolist(), y_inside.tolist(), 0
    while i < len(noises):
        if noises[i] == 0:
            noises.pop(i)
            signals.pop(i)
            x_inside.pop(i)
            y_inside.pop(i)
            inside_pix_coords.pop(i)
        else:
            i += 1
    noises, signals, x_inside, y_inside = np.array(noises), np.array(signals), np.array(x_inside), np.array(y_inside)

print("Done!")

#--------------- VORONOI BINNING ------------------------------
print("Voronoi binning...")

# Voronoi binning
bin_number, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(
    x_inside, y_inside, signals, noises, target_sn, cvt=True, pixelsize=1.0, plot=False,
    quiet=True, sn_func=None, wvt=False)
print("Done!")

# -------------- CREATE THE AVERAGED BIN CUBE --------------------------
print("Create and write output files...")
n_bins = len(nPixels)    # number of bins
# map of the bins, np.nan is not in any bin / outside region
bin_array = np.full((N_y,N_x), -1, dtype=float)

# put pixels into bin
for i in range(len(inside_pix_coords)):
        bin_index, coord = bin_number[i], inside_pix_coords[i]
        bin_array[coord] = bin_index

# output binmap to a csv file
if output_binmap_fn:
    print("    Csv file...")
    np.savetxt(output_binmap_fn, bin_array, fmt="%d")
    print("    Done!")

# if outputting a full average bin cube
if output_cube_fn:
    print("    Binned cube...")

    # array of the averaged signals and variances
    avg_sig_array = np.zeros((n_bins, N_w), dtype=np.float64)
    avg_var_array = np.zeros((n_bins, N_w), dtype=np.float64)

    # add the spectrum from each pixel into their respective bin
    for i in tqdm(range(len(inside_pix_coords))):
        bin_index, coord = bin_number[i], inside_pix_coords[i]
        avg_sig_array[bin_index] += cube_data[:, coord[0], coord[1]]
        avg_var_array[bin_index] += cube_var[:, coord[0], coord[1]]
    
    reshaped_nPixels = nPixels.reshape((n_bins,1))
    avg_sig_array = avg_sig_array / (reshaped_nPixels)             # signal for each bin
    avg_var_array = avg_var_array / (reshaped_nPixels**2)      # variance for each bin 

    # spectrum and variance map
    binned_specmap, binned_varmap = np.full((N_w, N_y, N_x), np.nan, dtype=np.float64), np.full((N_w, N_y, N_x), np.nan, dtype=np.float64)
    
    for y in tqdm(range(N_y)):
        for x in range(N_x):
            bin_id = int(bin_array[y,x])
            if bin_id >= 0:
                binned_specmap[:, y, x], binned_varmap[:, y, x] = avg_sig_array[bin_id], avg_var_array[bin_id]
    
    # create and write to file
    avg_binned_cube = Cube(ext=('DATA', 'STAT'), wcs=cube_wcs, wave=cube_wave, unit=cube_unit, copy=False,
                          data=binned_specmap, var=binned_varmap)
    avg_binned_cube.write(output_cube_fn)

    del avg_binned_cube, binned_specmap, binned_varmap, avg_sig_array, avg_var_array
    gc.collect()

    print("    Done!")

if output_spec_fn:
    print("    Spectrum...")

    # array of the averaged signals and variances
    avg_sig_array = np.zeros((n_bins, N_w), dtype=np.float64)
    avg_var_array = np.zeros((n_bins, N_w), dtype=np.float64)
    spec_sig = np.zeros(N_w, dtype=np.float64)
    spec_var = np.zeros(N_w, dtype=np.float64)

    # add the spectrum from each pixel into their respective bin if the sn is greater than the target sn
    for i in tqdm(range(len(inside_pix_coords))):
        bin_index, coord = bin_number[i], inside_pix_coords[i]
        if sn[bin_index] >= target_sn:
            avg_sig_array[bin_index] += cube_data[:, coord[0], coord[1]]
            avg_var_array[bin_index] += cube_var[:, coord[0], coord[1]]
    
    reshaped_nPixels = nPixels.reshape((n_bins,1))
    avg_sig_array = avg_sig_array / (reshaped_nPixels)             # signal for each bin
    avg_var_array = avg_var_array / (reshaped_nPixels**2)      # variance for each bin 
    
    spec_sig = np.sum(avg_sig_array, axis=0) / n_bins
    spec_var = np.sum(avg_var_array, axis=0) / (n_bins**2)

    spec = Spectrum(wave=cube_wave, data=spec_sig, var=spec_var, unit=cube_unit)
    
    spec.plot(title="Average Spectrum For High S/N bins", noise=True)
    plt.savefig(output_spec_fn)

    print("    Done!")
    
if output_fits_fn:
    print("    Bin map...")
    # assign each bin number a value that is its signal to noise ratio
    bin_sn_val = bin_number
    for i in range(len(inside_pix_coords)):
        bin_sn_val[i] = sn[bin_number[i]]
    
    # create a new bin array with the randomized bin numbers
    sn_bin_array = np.full((N_y,N_x), np.nan, dtype=float)
    for i in range(len(inside_pix_coords)):
        sn_bin_index, coord = bin_sn_val[i], inside_pix_coords[i]
        sn_bin_array[coord] = sn_bin_index

    # write the high contrast bin array to a fits file
    bins_fits = fits.PrimaryHDU(data=sn_bin_array, header=fits.Header(wcs_header))
    bins_fits.writeto(output_fits_fn, overwrite=True)

    del bins_fits
    gc.collect()

    print("    Done!")

print("Done!")
