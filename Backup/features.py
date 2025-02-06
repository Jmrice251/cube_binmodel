import numpy as np
import json
from lmfit import Model, Parameters
import copy

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
                
                # set parameter hints
                self.parsed_init_params = None
                for var in variables:
                    var_name = var['name']
                    self.set_param_hint(name=var_name, vary=True)
                    if 'lb' in var:                # set lower bound
                        self.set_param_hint(name=var_name, min=var['lb'])
                    if 'ub' in var:                # set 
                        self.set_param_hint(name=var_name, max=var['ub'])
                # if not guessing initial values
                if not guess_params:      
                    init_dict = { var['name'] : var['init'] for var in variables }
                    self.parsed_init_params = self.make_params(**init_dict)
                    
        return ParsedModelLMFIT()




# define fit function
def fit_lm(label, fit_model, X, Y, dY, method, contpoly_order):
    # weights from standard error
    err_weights = 1. / dY
    
    # continuum fitting:
    contpoly_coefs, contsub, sigcont, _ = continuum_fit(X, Y, dY, contpoly_order)
    
    # if mixed method
    method_name = method
    
    
    # if there are initial parameters
    if fit_model.parsed_init_params:
        result = fit_model.fit(Y - contsub, params=fit_model.parsed_init_params, weights=err_weights, method=method_name, x=X)
        return (label, result.params, contpoly_coefs)
    # if not then guess, haven't implemented this yet #
    else:  
        return None
    
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
    contpoly_coef = np.polynomial.polynomial.polyfit(x=wave[okvals], y=spec[okvals], 
                                                     w=sig_weights[okvals], deg=polynomial_fit_order)
    contpoly_vals = np.polynomial.polynomial.polyval(x=wave, c=contpoly_coef)
    residual = spec[okvals] - contpoly_vals[okvals]
    sigcont = 0.5 * (np.percentile(residual, 86) + np.abs(np.percentile(residual, 14)))
    
    return contpoly_coef, contpoly_vals, sigcont, okvals


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
    