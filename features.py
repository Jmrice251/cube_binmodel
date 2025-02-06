import numpy as np
import json
from lmfit import Model, Parameters, parameter
import copy
import astropy.units as u
import astropy.wcs as w
import astropy.io.fits as fits
import tqdm

# Changes from npichette: Line 158: Added an if statement so that only bins that have a non-empty wave ok-value array are fitted. An else is added here as well to keep bin-numbers throughout. Line 128: Added an if statement so that bins that were ignored as part of the line 158 change are not fit to avoid NaN errors. 

# Changes from Huan (07/24/24): use try/except blocks for Nathanael's changes

# Changes from John: Add parameter guessing function (lines 178-218) for each spaxel and function to convert results dict to all .json projectable datatypes, and then back again when saving and loading fits (lines 19-46) and (lines 49-71).

# Changes from Huan (11/15/2024): add a covariance calculated flag to fit result (Lines 237-244)

#----------------------------------------------------------------------------------------

# process python dictionary to a json serializable format
def convert_to_json_serializable(data):
    if isinstance(data, np.ndarray): #Arrays are converted to lists
        return data.tolist()
    elif isinstance(data, dict): #Dictionaries keys are fed through again to convert those to proper classes
        return {k: convert_to_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list): #List elements are fed through again
        return [convert_to_json_serializable(v) for v in data]
    elif isinstance(data, u.CompositeUnit): #Astropy unit classes are converted to strings
        return str(data) 
    elif isinstance(data, w.WCS): #WCS objects are converted to headers first and then fed through again
        data = data.to_header()
        return convert_to_json_serializable(data)
    elif isinstance(data, fits.Header): #Header objects are converted to dictionaries
        return {k: convert_to_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, parameter.Parameter): #Parameter objects are converted to dictionaries manually
        data = {
            'name': data.name,
            'value': data.value,
            'vary': data.vary,
            'min': data.min,
            'max': data.max,
            'expr': data.expr,
            'brute_step': data.brute_step,
            'stderr': data.stderr
        }
        return convert_to_json_serializable(data)
    else: #If the type was already json serializable, then return that data to stop iteration
        return data

# process json formated dictionary back to iterable 
def convert_to_interactive(data):
    for key in list(data.keys()): 
        if key == 'cube_info': #Change cube wcs back to WCS class and change wave_A back to an array
            data[key]['cube_wcs'] = w.WCS(fits.Header(data[key]['cube_wcs']))
            data[key]['wave_A'] = np.array(data[key]['wave_A'])
        elif isinstance(int(key), int): 
            data[int(key)] = data[key]
            final_params = Parameters() #initialize a parameters object to put back fit parameters into
            for k in list(data[key]['final_params'].keys()): 
                name = data[key]['final_params'][k]['name']
                value = data[key]['final_params'][k]['value']
                vary = data[key]['final_params'][k]['vary']
                min = data[key]['final_params'][k]['min']
                max = data[key]['final_params'][k]['max']
                expr = data[key]['final_params'][k]['expr']
                brute_step = data[key]['final_params'][k]['brute_step']
                stderr = data[key]['final_params'][k]['stderr']
                final_params.add(name, value=value, vary=vary, min=min, max=max, expr=expr, brute_step=brute_step) #add dictionary values to parameter objects
                final_params[name].stderr = stderr
            data[int(key)]['final_params'] = final_params #replace dictionary with parameters and then delete dictionary
            del data[key]
            
    return data
    

# process gaussian component
def parseGaussian(component):
    # form expression
    center, flux, velo, disp = component['center'], component['flux'], component['velo'], component['disp'] 
    expression = f'({flux})*exp(-(x-({center})*(1.0+({velo})))**2/(2*(({disp})**2)))'
    used_vars = {center, flux, velo, disp}

    # expression for antiderivative
    antiderivative_expression = f'({flux})*(sqrt(2.*pi)*({disp})/2.)*erf((x-({center})*(1.0+({velo})))/(sqrt(2.)*({disp})))'

    return expression, used_vars, component, antiderivative_expression

# parse model
def parseModelFunction(filename, delta=0.0, consumer='lmfit'):
    
    f = open(filename, 'r')
    obj = json.load(f)     # load json object
    f.close()
    
    # indicator to guess variable initial value or not
    guess_params = obj['guess_params']

    # get variable list
    variables = obj['variables']
    var_list = [var['name'] for var in variables]
    if not var_list:
        raise Exception('Variable list cannot be empty')

    # set of unused variables
    unused_var_set = set(var_list)
    if len(var_list) > len(unused_var_set):
        raise Exception('Duplicates found in variable list')
        
    # line dictionaries, containing basic info about the lines
    line_dicts_list = []

    # build function expression
    func_expression_list = []
    antiderivative_expression_list = []
    for component in obj['components']:
        component_expression, used_vars, antiderivative_expression = None, None, None
        if component['type'] == 'gaussian':
            # read in component, add to line dicts list
            component_expression, used_vars, line_dict, antiderivative_expression = parseGaussian(component)
            line_dicts_list.append(line_dict)
        else:
            raise Exception('Unrecognized component type. Must be \'gaussian\'')

        # add to expression
        func_expression_list.append(component_expression)
        antiderivative_expression_list.append(antiderivative_expression)
        unused_var_set = unused_var_set.difference(used_vars)

    # check if there's any remaining unused variables
    if len(unused_var_set) > 0:
        raise Exception(f'The following variables are not used: {unused_var_set}')

    # define the model functions, store into custom scope
    func_expression = ' + '.join(func_expression_list)
    antiderivative_expression = '+'.join(antiderivative_expression_list)
    var_list_str = ', '.join(var_list)
    script1 = f'from numpy import exp \ndef unbinned_model_func(x,{var_list_str}): \n\treturn {func_expression}'             # script to define the unbinned model function
    # script2 = f'from scipy.integrate import quad \ndef binned_model_func_unvectorized(x,{var_list_str}): \n\treturn ' \
    #             + f'quad(lambda x,{var_list_str}: {func_expression}, args=({var_list_str}), a=x-({delta}/2.), b=x+({delta}/2.))[0] / {delta} \n\n'           # script to define the binned model function
    # script3 = f'from numpy import vectorize \nbinned_model_func_vectorized = vectorize(binned_model_func_unvectorized); binned_model_func = lambda x,{var_list_str}: binned_model_func_vectorized(x,{var_list_str})'
    script2 = f'from scipy.special import erf \nfrom numpy import pi, sqrt\nantiderivative_func = lambda x,{var_list_str}: {antiderivative_expression} \n'
    script3 = f'binned_model_func = lambda x,{var_list_str}: (antiderivative_func(x+{delta}/2.,{var_list_str}) - antiderivative_func(x-{delta}/2.,{var_list_str})) / {delta}'
    scope = {}
    exec(script1, scope)
    exec(script2 + script3, scope)
    unbinned_model_func = scope['unbinned_model_func']
    binned_model_func = scope['binned_model_func']
    
    # if expect model to be consumed by LMFIT
    if consumer == 'lmfit':
        class ParsedModelLMFIT(Model):       # define a LMFIT model class
            # initializer
            def __init__(self, *args, **kwargs):
                # define expression
                super(ParsedModelLMFIT, self).__init__(binned_model_func if delta > 0.0 else unbinned_model_func)
                self.expression = func_expression
                self.lines_dict = line_dicts_list
                self.isBinned = True if delta > 0.0 else False           # delta > 0.0 means using binned model, else using unbinned
                self.unbinned_model_func = unbinned_model_func
                self.binned_model_func = binned_model_func
                self.ant_exp = antiderivative_expression
                self.guess_params = guess_params
                
                # set parameter hints
                self.parsed_init_params = None
                for var in variables:
                    var_name = var['name']
                    self.set_param_hint(name=var_name, vary=True)
                    if 'lb' in var:                # set lower bound
                        self.set_param_hint(name=var_name, min=var['lb'])
                    if 'ub' in var: # set upper bound
                        self.set_param_hint(name=var_name, max=var['ub'])

                init_dict = { var['name'] : var['init'] for var in variables }
                self.parsed_init_params = self.make_params(**init_dict)
                    
        return ParsedModelLMFIT()


def guess_model(fit_model, X, Y):
    class ParsedModelLMFIT(Model):       # define a LMFIT model class
        # initializer
        def __init__(self, *args, **kwargs):
            # define expression
            super(ParsedModelLMFIT, self).__init__(fit_model.binned_model_func if fit_model.isBinned else fit_model.unbinned_model_func)
            self.expression = fit_model.expression
            self.lines_dict = fit_model.lines_dict
            self.isBinned = fit_model.isBinned
            self.unbinned_model_func = fit_model.unbinned_model_func
            self.binned_model_func = fit_model.binned_model_func
            self.ant_exp = fit_model.ant_exp
            self.guess_params = fit_model.guess_params

            width = estimatewidth(X, Y, 6562.819, fit_model.parsed_init_params['z'].value)
                
            # set parameter hints, i.e. guessed constraints
            self.parsed_init_params = None
            for param in fit_model.parsed_init_params:
                self.set_param_hint(name=param, vary=True)
                if param == 'A2':
                    self.set_param_hint(name=param, min=0.4*max(Y), max=1.2*max(Y))
                elif param == 'A1':
                    self.set_param_hint(name=param, min=0, max=1.2/3*max(Y))
                elif param == 'A4' or param == 'A5':
                    self.set_param_hint(name=param, min=0, max=0.7*max(Y))
                elif param == 'z':
                    self.set_param_hint(name=param, min=fit_model.parsed_init_params[param].value-3e-3, max=fit_model.parsed_init_params[param].value+3e-3)
                else:
                    self.set_param_hint(name=param, min=fit_model.parsed_init_params[param].min, max=fit_model.parsed_init_params[param].max)
            init_vals = {}    
            for param in fit_model.parsed_init_params: # guess initial values
                if param == 'z':
                    init_vals[param] = fit_model.parsed_init_params[param].value
                elif param == 'sigma':
                    init_vals[param] = width
                elif param == 'A2':
                    init_vals[param] = max(Y)
                elif param == 'A1':
                    init_vals[param] = max(Y)/3
                else:
                    init_vals[param] = 0.2*max(Y)
            init_dict = { param : init_vals[param] for param in fit_model.parsed_init_params }
            self.parsed_init_params = self.make_params(**init_dict)

    return ParsedModelLMFIT()


# define fit function
def fit_lm(label, fit_model, X, Y, dY, method_name, contpoly_order):

    try:
        # weights from standard error
        err_weights = 1. / dY
        
        # continuum fitting
        contpoly_coefs, contsub, sigcont, _ = continuum_fit(X, Y, dY, contpoly_order)
        
        # fit spectrum
        if not fit_model.guess_params:     # if there are initial param values
            result = fit_model.fit(Y - contsub, params=fit_model.parsed_init_params, weights=err_weights, method=method_name, x=X)
        	
        # if not then guess
        else:
            new_model = guess_model(fit_model, X, Y - contsub)       
            result = new_model.fit(Y - contsub, params=new_model.parsed_init_params, weights=err_weights, method=method_name, x=X)

        # if result stay None, throw error
        if (result is None):
            raise Exception("FitResult still None after fitting")
        
        # is covariance calculated
        isCovarCalculated = (result.covar is not None)

        return (label, result.params, contpoly_coefs, isCovarCalculated, result.covar)
            
    except Exception as err:
        print("Failed to fit model on spectrum:", err)
        return (label, None, None)
    
# ------------------------------------------------------------------------
# Features from KubeViz by Dr. Matteo Fossati (https://github.com/matteofox/kubeviz)

# continuum fitting (lines  3047-3070)
def continuum_fit(wave, spec, noise, polynomial_fit_order):
    if polynomial_fit_order is None:
        return None, np.zeros_like(spec), 0., None
                                                            
    # weights from standard error
    sig_weights = 1. / noise
    
    # get values on the wave axis to be included for continuum fitting. Criteria:
    #     1. weight is finite
    #     2. |noise - median(noise)| leq (max_cutoff)%ile - (min_cutoff)%ile
    #     3. |spec - median(spec)| leq (max_cutoff)%ile - (min_cutoff)%ile
    max_cutoff, min_cutoff = 90, 10
    okvals = np.isfinite(sig_weights) \
        * np.asarray(np.abs(noise - np.median(noise)) <= np.percentile(noise, max_cutoff)-np.percentile(noise, min_cutoff)) \
        * np.asarray(np.abs(spec - np.median(spec)) <= np.percentile(spec, max_cutoff)-np.percentile(spec, min_cutoff))
    
    # fit to polynomial of specified order
    try:	
    	contpoly_coef = np.polynomial.polynomial.polyfit(x=wave[okvals], y=spec[okvals], 
                                                     w=sig_weights[okvals], deg=polynomial_fit_order)
    	contpoly_vals = np.polynomial.polynomial.polyval(x=wave, c=contpoly_coef)
    	residual = spec[okvals] - contpoly_vals[okvals]
    	sigcont = 0.5 * (np.percentile(residual, 86) + np.abs(np.percentile(residual, 14)))
    
    	return contpoly_coef, contpoly_vals, sigcont, okvals
    except:
        raise Exception("Failed to fit continuum")


# get instrumentation resolution (line 1332-1333)
def get_corrected_width(observed_xcen, observed_width):
    # evaluate polynomial
    MUSE_instrrespoly_coefs = np.array([3.32338754e+03, -1.46784920, 3.12869916e-04, -1.64479615e-08])
    instrres_R = np.polynomial.polynomial.polyval(x=observed_xcen, c=MUSE_instrrespoly_coefs)
    
    # convert to angstrom
    sigtofwhm = 2.35482
    instrres_A = observed_xcen / (sigtofwhm * instrres_R)
    
    if observed_width < instrres_A:    # if observed width is smaller than instr. resolution, return 0
        return 0
    else:
        return np.sqrt(observed_width**2 - instrres_A**2)    # return subtracted width

def estimatewidth(x, y, wavelength, z):
    #estimatewidth is kubeviz_linefit_estimatekinematics (line 1216-1281)
    glambda = wavelength*(1+z)
    
    checkrange = 40/(1+z)
    nearpeak = np.asarray(np.abs(x-glambda) <= checkrange).nonzero()
    nx = np.shape(nearpeak)[1]

    if nx < 5:
        raise Exception("Not enough data points near peak")

    x = x[nearpeak]
    y = y[nearpeak]

    i_s = np.argsort(x)
    xs, ys = x[i_s], y[i_s]

    xmin, xmax, ymin, ymax = np.min(xs), np.max(xs), np.nanmin(ys), np.nanmax(ys)
    dx = 0.5 * np.concatenate((np.array([xs[1] - xs[0]]), xs[2:] - xs[:nx-2], np.array([xs[nx-1] - xs[nx-2]])))
    totalarea = np.sum(np.multiply(ys,dx))
    av = totalarea/(xmax-xmin)

    wh1 = np.asarray(ys >= av).nonzero()
    ct1 = np.shape(wh1)[1]
    wh2 = np.asarray(ys <= av).nonzero()
    ct2 = np.shape(wh2)[1]
    if ct1 == 0 or ct1 == 0:
        raise Exception("Average should fall between range of Y values but doesn't")

    cent = x[np.asarray(y == ymax).nonzero()]
    cent = cent[0]
    peak = ymax - av
    peakarea = totalarea - np.sum(np.multiply(dx[wh2],ys[wh2]))
    if peak == 0:
        peak = 0.5*peakarea
    width = peakarea / (2*abs(peak))

    return width


# display line information
def print_line_info(line_dict, bestfit_params):
    
    # extract param_names, eval them
    # the variables exec'd this will be in the function scope, so no conflict with other global variables
    param_names = list(bestfit_params.keys())
    for name in param_names:
        # store these as python vars inside local scope
        exec(name + ' = ' + str(bestfit_params[name].value))
        
    # extract variables from line information
    observed_center = eval(str(line_dict['center'])) * ( 1.0 + eval(str(line_dict['velo'])) )
    observed_width = eval(str(line_dict['disp']))
    
    # correct width
    corrected_width = get_corrected_width(observed_center, observed_width)
    
    print(line_dict['name'] + ' info:')
    print(f'Observed center: {observed_center} A')
    print(f'Corrected width: {corrected_width} A')
    
    # delete the variables defined by exec'd
    for name in param_names:
        # store these as python vars for usage, just inside this function
        exec(f'del {name}') 
    
