from mpdaf.obj import Cube
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import features

from regions import Regions
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord


import json
from tqdm import tqdm
from joblib import Parallel, delayed

np.seterr(invalid='ignore')

#Changes from npichette: Line 153: Added an if statement so that bad bins are not subscripted. Bad bins are those that are ignored in features.py because of the wave[okvals] vector being empty.
#------------------------------------------------------------------------------
# arguments from cmd line
nargs = 2
if (len(sys.argv) < nargs):
    print(f'Usage: python3 {sys.argv[0]} <fit_info_file>')
    exit()

# open fit info json file
fit_info_fn = sys.argv[1]
fit_info_f = open(fit_info_fn, 'r')
fit_info_obj = json.load(fit_info_f)
fit_info_f.close()

# extract fit information
cube_fn = fit_info_obj['cube_filename']
mask_fn = fit_info_obj['mask_filename']
result_dir_name = fit_info_obj['result_dir_name']
model_fn = fit_info_obj['model_filename']
method = fit_info_obj['method']
numprocs = fit_info_obj['numprocs']
redshift = fit_info_obj['redshift']
contpoly_order = fit_info_obj['contpoly_order']
binmap_fn = fit_info_obj['binmap_fn']
    
# fitting window (extend by +/- 50A at endpoints)
margin = 50
wave_min = min(fit_info_obj['wave_fit_window'])*(1+redshift) - margin
wave_max = max(fit_info_obj['wave_fit_window'])*(1+redshift) + margin 
print(f'Fitting from {wave_min} A to {wave_max} A...')
    
# create directory if necessary
if not os.path.exists(result_dir_name):
        os.mkdir(result_dir_name)
        
#--------------------------------------------------------------------------------
# Process information from cube
cube = Cube(cube_fn)
cube_wcs = WCS(cube.get_wcs_header())
N_w, N_y, N_x = cube.shape

# read region list
sky_mask_list = Regions.read(mask_fn)             # list of regions
if (len(sky_mask_list) == 0):
    print("ERROR: no mask regions found")
    exit()
# form union of all regions
sky_mask = sky_mask_list[0]
for i in range(1, len(sky_mask_list)):
    sky_mask = sky_mask.union(sky_mask_list[i])
# convert to pixel region, using celestial WCS
pix_mask = 1 - sky_mask.to_pixel(cube_wcs.celestial).to_mask('center').to_image((N_y, N_x)).astype('int')

# read in bin file
old_bin_array = np.loadtxt(binmap_fn, dtype=int)      # read the map of bin in as integer array
old_bin_array = np.where(pix_mask == 0, -1, old_bin_array)        # filter out mask

# map the old bin ids to new bin ids
old_bin_ids = np.unique(old_bin_array)
old_bin_ids = np.delete(old_bin_ids, np.where(old_bin_ids == -1))
new_bin_array = result_map = np.full((N_y, N_x), -1, dtype='int')
for i in range(len(old_bin_ids)):
    new_bin_array = np.where(old_bin_array == old_bin_ids[i], i, new_bin_array)

# get information about bin
n_bins = np.max(new_bin_array) + 1
pixels_per_bin = np.array([np.count_nonzero(new_bin_array == bin_id) for bin_id in range(n_bins)], dtype=int)     # how many pixels in each bin

# get the coordinates that are inside valid bins
(y_inside, x_inside) = (new_bin_array >= 0).nonzero()
inside_pix_coords = list(zip(y_inside, x_inside))

#-------------------------------------------------------------------------------------
# EXTRACT THE SPECTRA
print("Extracting the spectra from pixels:...")
fit_min, fit_max = max(wave_min, cube.wave.get_start()), min(wave_max, cube.wave.get_end())

# get metadata about the wave spectrum
CRVAL = cube.wave.get_crval()
CDELT = cube.wave.get_step()
lo_pix = int(np.floor((fit_min - CRVAL) / CDELT))
hi_pix = int(np.ceil((fit_max - CRVAL) / CDELT))
x0=np.arange(hi_pix + 1 - lo_pix)
wave_A=((x0+lo_pix)*CDELT + CRVAL)

# wave for plotting, contain more data points than the fitting wave array
# points_multiplier = 1000
# wave_plot = np.linspace(fit_min, fit_max, num=points_multiplier*(hi_pix - lo_pix + 1))

# cube of data within wavelength range, for pixels inside the region
cube2 = cube[lo_pix:(hi_pix+1), :, :]

# get the average spectrum (data + noise) for each bin
# array of the averaged signals and variances
avg_signal_array = np.zeros((n_bins, len(wave_A)), dtype=np.float64)
avg_variance_array = np.zeros((n_bins, len(wave_A)), dtype=np.float64)

# add the spectrum from each pixel into their respective bin
for coord in tqdm(inside_pix_coords):
    bin_index = new_bin_array[coord]
    avg_signal_array[bin_index] += cube2[:, coord[0], coord[1]].data.data
    avg_variance_array[bin_index] += cube2[:, coord[0], coord[1]].var.data

reshaped_nPixels = pixels_per_bin.reshape((n_bins,1))
avg_signal_array = avg_signal_array / (reshaped_nPixels)             # signal for each bin
avg_variance_array = avg_variance_array / (reshaped_nPixels**2)      # variance for each bin
avg_noise_array = np.sqrt(avg_variance_array)

#-----------------------------------------------------------
# parsed model
Model = features.parseModelFunction(model_fn, delta=CDELT)

# Do work in parallel
print('Fitting...')
results = Parallel(n_jobs=numprocs, backend="loky")(delayed(features.fit_lm)(bin, Model, wave_A, avg_signal_array[bin], avg_noise_array[bin], method, contpoly_order=contpoly_order) for bin in tqdm(range(n_bins)))

#----------------------------------------------------------------------------------------------
# MAP RESULT
print("Mapping results...")
fitresult_dict = {}
velo_map = np.full((N_y, N_x), np.nan)
AHb_map = np.full((N_y, N_x), np.nan)

for res in tqdm(results):
    bin_id = res[0]
    
    fitresult_dict[bin_id] = {
        'final_params': res[1],
        'contpoly_coefs': res[2]
    }
print("Velo map...")
for coord in tqdm(inside_pix_coords):
    bin_id = new_bin_array[coord]
    if fitresult_dict[bin_id]['final_params'] is not None:
    	velo_map[coord] = fitresult_dict[bin_id]['final_params']['z'].value
        
print("Hβ map...")

for coord in tqdm(inside_pix_coords):
    bin_id = new_bin_array[coord]
    if fitresult_dict[bin_id]['final_params'] is not None:
    	AHb_map[coord] = fitresult_dict[bin_id]['final_params']['A_Hb'].value
    
header = WCS.to_header(cube_wcs.celestial)

# write to files
fits.writeto(filename=f'{result_dir_name}/velo_map.fits', data=velo_map, header=header, overwrite=True)
fits.writeto(filename=f'{result_dir_name}/AHb_map.fits', data=AHb_map, header=header, overwrite=True)


# --------------------------------------------------------------------------
# interactive
while True:
    print("\nEnter x-coordinate and y-coordinate of spaxel, separated by whitespace. Enter \'q\' to quit:")
    user_input = input()
    
    # exit
    if user_input == 'q' or user_input == 'Q':
        exit()
    
    # check valid input
    split_list = user_input.split()
    if len(split_list) != 2:
        print("Invalid input. Try again...\n")
        continue
        
    # get pixel coordinate
    pix_coord = None
    try:
        ycoord, xcoord = int(split_list[1]), int(split_list[0])
        pix_coord = (ycoord, xcoord)
    except:
        print("Invalid input. Try again...\n")
        continue
    
    # check that pixel is in bin
    pix_bin_num = new_bin_array[pix_coord]
    if pix_bin_num < 0:
        print("Spaxel not in any valid bin. Try again...\n")
        continue
        
    # get info
    final_params = fitresult_dict[pix_bin_num]['final_params']
    contpoly_coefs = fitresult_dict[pix_bin_num]['contpoly_coefs']
    contpoly_vals = np.polynomial.polynomial.polyval(x=wave_A, c=contpoly_coefs)
    contpoly_vals_plot = np.polynomial.polynomial.polyval(x=wave_A, c=contpoly_coefs)
    [spec, noise] = avg_signal_array[pix_bin_num], avg_variance_array[pix_bin_num]
    
    # display spaxel information
    print('Spaxel fitting result:')
    sky_coord_str = pixel_to_skycoord(pix_coord[1], pix_coord[0], cube_wcs).to_string()
    print(f'RA DEC (in degrees): {sky_coord_str}')
    print(f'Bin number: {pix_bin_num}')
    print(f'Continuum polynomial coefficients (lowest degree term to highest):')
    print(contpoly_coefs)
    print('Best-fit results:')
    final_params.pretty_print()
    for line in Model.lines_dict:
        features.print_line_info(line, final_params)
    
    # Show plot
    plt.figure(figsize=(25, 5))
    plt.errorbar(wave_A, spec, yerr=noise, ls='none')
    plt.step(wave_A, spec, where='mid', linewidth=1.5, color='y')
    plt.step(wave_A, Model.eval(final_params, x=wave_A) + contpoly_vals_plot, where='mid', linewidth=1.5, color='r')
    plt.step(wave_A, contpoly_vals, where='mid', linewidth=3, color='g')
    plt.show()
