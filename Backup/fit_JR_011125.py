from mpdaf.obj import Cube, Spectrum
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import gc
import features

from regions import Regions
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord


import json
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import date

np.seterr(invalid='ignore')

# CHANGE 07/24/24: add condition to check that bin is fitted, to match with Nathanael's features.py code 

#------------------------------------------------------------------------------
# arguments from cmd line
nargs = 2
if (len(sys.argv) < nargs):
    print(f'Usage: python3 {sys.argv[0]} <fit_options_file>')
    exit()

# open fit options json file
fit_options_fn = sys.argv[1]
fit_options_f = open(fit_options_fn, 'r')
fit_options_obj = json.load(fit_options_f)
fit_options_f.close()

# extract fit options
cube_fn = fit_options_obj['cube_filename']
mask_fn = fit_options_obj['mask_filename']
result_dir_name = fit_options_obj['result_dir_name']
model_fn = fit_options_obj['model_filename']
method = fit_options_obj['method']
numprocs = fit_options_obj['numprocs']
redshift = fit_options_obj['redshift']
contpoly_order = fit_options_obj['contpoly_order']
binmap_fn = fit_options_obj['binmap_fn']
output_spec_fn = fit_options_obj['output_spec_fn'] if 'output_spec_fn' in fit_options_obj else None
sn_cutoff = fit_options_obj['sn_cutoff']
output_maps = fit_options_obj['output_maps']
fit_result_fn = fit_options_obj['fit_result_fn'] if 'fit_result_fn' in fit_options_obj else None
    
# fitting window (extend by margin at endpoints)
margin = fit_options_obj['margin']
wave_min = min(fit_options_obj['wave_fit_window'])*(1+redshift) - margin
wave_max = max(fit_options_obj['wave_fit_window'])*(1+redshift) + margin 

if not fit_result_fn:
    print(f'Fitting from {wave_min} A to {wave_max} A...')
    
    # create directory if necessary
    if not os.path.exists(result_dir_name):
        os.mkdir(result_dir_name)
    
    today = date.today()
    fit_result_dir = f'{result_dir_name}/fit_result_{today}.json'
    
        
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
# 0 = out, 1 = in
    pix_mask = 1 - sky_mask.to_pixel(cube_wcs.celestial).to_mask('center').to_image((N_y, N_x)).astype('int')

# read in bin file
    old_bin_array = np.loadtxt(binmap_fn, dtype=int)      # read the map of bin in as integer array
    old_bin_array = np.where(pix_mask == 0, -1, old_bin_array)        # filter out mask (out pixels => in bin -1)

# map the old bin ids to new bin ids
    old_bin_ids = np.unique(old_bin_array)
    old_bin_ids = np.delete(old_bin_ids, np.where(old_bin_ids == -1))
    new_bin_array = np.full((N_y, N_x), -1, dtype='int')
    for i in range(len(old_bin_ids)):
        new_bin_array = np.where(old_bin_array == old_bin_ids[i], i, new_bin_array)

# get information about bin
    n_bins = np.max(new_bin_array) + 1
    pixels_per_bin = np.array([np.count_nonzero(new_bin_array == bin_id) for bin_id in range(n_bins)], dtype=int)     # how many pixels in each bin

# get the coordinates that are inside valid bins
    (y_inside, x_inside) = np.nonzero(new_bin_array >= 0)
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

    # cube of data within wavelength range
    cube2 = cube[lo_pix:(hi_pix+1), :, :]
    cube_data = cube2.data.data
    cube_data[np.isnan(cube_data)] = 0
    cube_var = cube2.var.data
    cube_var[np.isnan(cube_var)] = 0.000001
    cube_unit = cube2.unit
    cube_wave = cube2.wave

    # IMPORTANT: FREE MEMORY FOR CUBE AND CUBE2
    del cube
    del cube2
    gc.collect()

    # get the average spectrum (data + noise) for each bin
    # array of the averaged signals and variances
    avg_signal_array = np.zeros((n_bins, len(wave_A)), dtype=np.float64)
    avg_variance_array = np.zeros((n_bins, len(wave_A)), dtype=np.float64)

    # add the spectrum from each pixel into their respective bin
    for coord in tqdm(inside_pix_coords):
        bin_index = new_bin_array[coord]
        avg_signal_array[bin_index] += cube_data[:, coord[0], coord[1]]
        avg_variance_array[bin_index] += cube_var[:, coord[0], coord[1]]

    reshaped_nPixels = pixels_per_bin.reshape((n_bins,1))
    avg_signal_array = avg_signal_array / (reshaped_nPixels)             # signal for each bin
    avg_variance_array = avg_variance_array / (reshaped_nPixels**2)      # variance for each bin

    np.savetxt(f'{result_dir_name}/avg_sig_array.csv', avg_signal_array, fmt="%f")
    np.savetxt(f'{result_dir_name}/avg_var_array.csv', avg_variance_array, fmt="%f")

    avg_noise_array = np.sqrt(avg_variance_array)
#-----------------------------------------------------------
# parsed model
    Model = features.parseModelFunction(model_fn, delta=CDELT)

# Do work in parallel
    print('Fitting...')
    results = Parallel(n_jobs=numprocs, backend="loky")(delayed(features.fit_lm)(bin, Model, wave_A, avg_signal_array[bin], avg_noise_array[bin], method, contpoly_order=contpoly_order) for bin in tqdm(range(n_bins)))

#----------------------------------------------------------------------------------------------
# RETRIEVE RESULT
    fitresult_dict = {}
    fitresult_dict['cube_info'] = {
        'wave_A': wave_A,
        'cube_unit': cube_unit,
        'cube_wcs': cube_wcs,
        'N_w': N_w,
        'N_y': N_y,
        'N_x': N_x,
        'CDELT': CDELT
    }    
    print("Retrieving fit results...")
    for res in tqdm(results):
        bin_id = res[0]  # Logic to determine whether or not fit was sucessful, overwrites bad fits with 0 vals
        if res[1] and (res[1]['A2'].stderr and res[1]['A1'].stderr and res[1]['A4'].stderr and res[1]['A5'].stderr):
            amps = [res[1]['A2'].value, 3*res[1]['A1'].value, res[1]['A4'].value, res[1]['A5'].value]
            errs = [res[1]['A2'].stderr, 3*res[1]['A1'].stderr, res[1]['A4'].stderr,res[1]['A5'].stderr]
            if (max(amps) != amps[0] and max(amps) != amps[1]) or \
                (min([amps[x] for x in [0,1]]) < 0) or\
                (sum([amps[i]/errs[i] < sn_cutoff for i in [0,1]]) > 0) or \
                (2*amps[1] < amps[2]) or (2*amps[1] < amps[3]) or (res[1]['A2'].min == amps[0]) or\
                (amps[0] - 0.1 < res[1]['A2'].min) or (res[1]['z'].value < redshift/3)\
                or (amps[0] > 10*amps[1]):
                    res[1]['A2'].min = 0
                    res[1]['A1'].value, res[1]['A2'].value, res[1]['A4'].value, res[1]['A5'].value, res[1]['z'].value, res[1]['sigma'].value = 0, 0, 0, 0, -100000, 0
        else:
            res[1]['A1'].value, res[1]['A2'].value, res[1]['A4'].value, res[1]['A5'].value, res[1]['z'].value, res[1]['sigma'].value = 0, 0, 0, 0, -100000, 0
        fitresult_dict[bin_id] = {
            'final_params': res[1],
            'contpoly_coefs': res[2]
        }

    # OUTPUT MAPS
    for map_name in output_maps:
        print(f"Creating map: {map_name}.fits")
        header = WCS.to_header(cube_wcs.celestial)
        try:
            # expression
            map_expr = output_maps[map_name] 

            # get value for each pixel
            result_map = np.full((N_y, N_x), np.nan)
            for coord in tqdm(inside_pix_coords):
                bin_id = new_bin_array[coord]
                if fitresult_dict[bin_id]['final_params'] is not None:       # check that bin is fitted
                    # create custom scope for model variables
                    custom_scope = {}
                    for var_name in Model.param_names:
                        custom_scope[var_name] = fitresult_dict[bin_id]['final_params'][var_name].value
                    # eval
                    result_map[coord] = eval(map_expr, custom_scope)
                    unit = cube_unit
                    if map_name == 'z_map':
                        result_map[coord] = (result_map[coord]-redshift)*299792.458
                        header['bunit'] = 'km/s'
                    elif map_name == 'sigma_map':
                        result_map[coord] = (result_map[coord]/6562.80)*299792.458
                        header['bunit'] = 'km/s'
                    else:
                        header['bunit'] = str(cube_unit)
            
            # save to disk
            fits.writeto(filename=f'{result_dir_name}/{map_name}.fits', data=result_map, header=header, overwrite=True)

            # free mem
            del result_map
            gc.collect()

        except:
            print(f"Error when creating map {map_name}.fits!\n")

    json_fitresult_dict = features.convert_to_json_serializable(fitresult_dict)
    with open(fit_result_dir, "w") as fit:
        json.dump(json_fitresult_dict, fit)
        
# --------------------------------------------------------------------------
# fit retrieval
else:
    print('Loading fit result...')
    fit_result_dir = f'{result_dir_name}/{fit_result_fn}'
    with open(fit_result_dir, "r") as fit:
        json_fitresult_dict = json.load(fit)

    fitresult_dict = features.convert_to_interactive(json_fitresult_dict)

    cube_wcs = fitresult_dict['cube_info']['cube_wcs']
    N_w, N_y, N_x = fitresult_dict['cube_info']['N_w'], fitresult_dict['cube_info']['N_y'], fitresult_dict['cube_info']['N_x']
    wave_A = fitresult_dict['cube_info']['wave_A']
    CDELT = fitresult_dict['cube_info']['CDELT']
    cube_unit = fitresult_dict['cube_info']['cube_unit']

    print('Reading Binmap...')
    sky_mask_list = Regions.read(mask_fn)             # list of regions
    if (len(sky_mask_list) == 0):
        print("ERROR: no mask regions found")
        exit()
# form union of all regions
    sky_mask = sky_mask_list[0]
    for i in range(1, len(sky_mask_list)):
        sky_mask = sky_mask.union(sky_mask_list[i])
# convert to pixel region, using celestial WCS
# 0 = out, 1 = in
    pix_mask = 1 - sky_mask.to_pixel(cube_wcs.celestial).to_mask('center').to_image((N_y, N_x)).astype('int')

# read in bin file
    old_bin_array = np.loadtxt(binmap_fn, dtype=int)      # read the map of bin in as integer array
    old_bin_array = np.where(pix_mask == 0, -1, old_bin_array)        # filter out mask (out pixels => in bin -1)

# map the old bin ids to new bin ids
    old_bin_ids = np.unique(old_bin_array)
    old_bin_ids = np.delete(old_bin_ids, np.where(old_bin_ids == -1))
    new_bin_array = np.full((N_y, N_x), -1, dtype='int')
    for i in range(len(old_bin_ids)):
        new_bin_array = np.where(old_bin_array == old_bin_ids[i], i, new_bin_array)

# get information about bin
    n_bins = np.max(new_bin_array) + 1
    pixels_per_bin = np.array([np.count_nonzero(new_bin_array == bin_id) for bin_id in range(n_bins)], dtype=int)     # how many pixels in each bin

    print('Extracting data from average arrays...')
    avg_signal_array = np.array(np.loadtxt(f'{result_dir_name}/avg_sig_array.csv', dtype=float))
    avg_variance_array = np.array(np.loadtxt(f'{result_dir_name}/avg_var_array.csv', dtype=float))
    avg_noise_array = np.sqrt(avg_variance_array)

    Model = features.parseModelFunction(model_fn, delta=CDELT)

# --------------------------------------------------------------------------
# interactive mode
while True:
    print("\nEnter x-coordinate and y-coordinate of spaxel, separated by whitespace OR bin id. Enter \'q\' to quit:")
    user_input = input()
    
    # exit
    if user_input == 'q' or user_input == 'Q':
        if output_spec_fn:
            while True:
                print("\nDo you want to output the Average Spectrum? (y/n):")
                user_input = input()
                if user_input == 'y' or user_input == 'Y':
                    # Total Spectrum
                    print("Creating Average Spectrum")
                    zeroed_signal_array = avg_signal_array
                    zeroed_signal_array[np.isnan(avg_signal_array)] = 0 #change nan values to zero to not mess up the spec graph
                    zeroed_variance_array = avg_variance_array
                    zeroed_variance_array[np.isnan(avg_variance_array)] = 0

                    spec_sig = np.zeros(len(wave_A), dtype=np.float64)
                    spec_var = np.zeros(len(wave_A), dtype=np.float64)

                    bins_used = 0
                    for bin_id in tqdm(range(n_bins)):  # add the signal and variance of high sigma bins
                        if fitresult_dict[bin_id]['final_params'] and fitresult_dict[bin_id]['final_params']['A2'].stderr:
                            if fitresult_dict[bin_id]['final_params']['A2'].value/fitresult_dict[bin_id]['final_params']['A2'].stderr > sn_cutoff:
                                spec_sig += zeroed_signal_array[bin_id]
                                spec_var += zeroed_variance_array[bin_id]
                                bins_used += 1

                    spec_sig = spec_sig / bins_used  # average the signal and variances
                    spec_var = spec_var / (bins_used**2)
                    spec_noise = np.sqrt(spec_var)
                    
                    plt.figure(figsize=(17, 5))
                    plt.title(f'Average Spectrum For Bins with s/n > {sn_cutoff}')  # plot and save the spectrum
                    plt.ylabel(f'Spectral Flux Density [{cube_unit}]')
                    plt.xlabel('Wavelength [Angstrom]')
                    plt.errorbar(wave_A, spec_sig, yerr=spec_noise, ls='none', label='Error')
                    plt.step(wave_A, spec_sig, where='mid', linewidth=2, color='y', label='Data')
                    plt.grid(which='both',axis='both',color='0.95')
                    plt.minorticks_on()
                    plt.legend()
                    plt.subplots_adjust(left=0.055, right=0.994, top=0.85, bottom=0.117)
                    plt.margins(0.01,0.1)
                    plt.savefig(f'{result_dir_name}/{output_spec_fn}.png', bbox_inches='tight')
                    exit()
                elif user_input == 'n' or user_input == 'N':
                    exit()
                else:
                    print("Invalid input. Try again...\n")                
                    continue
        else:
            exit()
    
    # check valid input
    split_list = user_input.split()
    if len(split_list) != 2 and len(split_list) != 1:
        print("Invalid input. Try again...\n")
        continue
        
    # get pixel coordinate
    pix_coord = None
    if len(split_list) == 2:
        try:
            ycoord, xcoord = int(split_list[1]), int(split_list[0])
            pix_coord = (ycoord, xcoord)
        except:
            print("Invalid coordinate. Try again...\n")
            continue
    else:
        try:
            bin_id = int(split_list[0])
            (ypixel, xpixel) = np.where(new_bin_array == bin_id)
            if isinstance(xpixel,int):
                pix_coord = (ypixel, xpixel)
            else:
                pix_coord = (ypixel[0], xpixel[0])
        except:
            print("Invalid bin id. Try again...\n")
            continue
    
    # check that pixel is in bin
    try:
        pix_bin_num = new_bin_array[pix_coord]
        if pix_bin_num < 0:
            print("Spaxel not in any valid bin. Try again...\n")
            continue
    except:
        print("Pixel out of range of Cube. Try again...\n")
        continue
        
    # get info
    final_params = fitresult_dict[pix_bin_num]['final_params']
    contpoly_coefs = fitresult_dict[pix_bin_num]['contpoly_coefs']
    
    # display spaxel information
    if fit_result_fn:
            print('-------- RESULTS RETRIEVED FROM PREVIOUS FIT --------')
    print('Spaxel fitting result:')
    sky_coord_str = pixel_to_skycoord(pix_coord[1], pix_coord[0], cube_wcs).to_string()
    print(f'RA DEC (in degrees): {sky_coord_str}')
    print(f'Bin number: {old_bin_array[pix_coord]}')
    
    if final_params is None:               # bin is not fitted
        print(f'Bin {old_bin_array[pix_coord]} is not fitted')
    else:                                  # if bin is fitted, i.e. both final_params & contpoly_coefs not None
        contpoly_vals = np.polynomial.polynomial.polyval(x=wave_A, c=contpoly_coefs)
        [spec, noise] = avg_signal_array[pix_bin_num], avg_noise_array[pix_bin_num]
        print(f'Continuum polynomial coefficients (lowest degree term to highest):')
        print(contpoly_coefs)
        print('Best-fit results:')
        final_params.pretty_print()
        for line in Model.lines_dict:
            features.print_line_info(line, final_params)
        
        # Show plot
        plt.figure(figsize=(17, 5))
        plt.title(f'Bin number: {old_bin_array[pix_coord]}\n pixel: x = {pix_coord[1]}, y = {pix_coord[0]}\n RA DEC: {sky_coord_str}')
        plt.ylabel(f'Spectral Flux Density [{cube_unit}]')
        plt.xlabel('Wavelength [Angstrom]')
        error = plt.errorbar(wave_A, spec, yerr=noise, ls='none', label='Error')
        data = plt.step(wave_A, spec, where='mid', linewidth=1.5, color='y', label='Data')
        gaussians = plt.step(wave_A, Model.eval(final_params, x=wave_A) + contpoly_vals, where='mid', linewidth=1.5, color='r', label='Gaussians')
        continuum = plt.step(wave_A, contpoly_vals, where='mid', linewidth=3, color='g', label='Continuum')
        plt.grid(which='both',axis='both',color='0.95')
        plt.minorticks_on()
        plt.legend()
        plt.subplots_adjust(left=0.055, right=0.994, top=0.85, bottom=0.117)
        plt.show()

