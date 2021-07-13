from prospect.models.sedmodel import SedModel
from prospect.models.templates import TemplateLibrary
from prospect.models import priors
from prospect.models import transforms

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------


### Leja Functions for model_params ###
def transform_logmass_to_mass(mass=None, logmass=None, **extras):
    return 10**logmass

def load_gp(**extras):
    return None, None

def tie_gas_logz(logzsol=None, **extras):
    return logzsol

def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2

def transform_zfraction_to_sfrfraction(sfr_fraction=None, z_fraction=None, **extras):
    if sfr_fraction.size != 0:
        sfr_fraction[0] = 1-z_fraction[0]
        for i in list(range(1,sfr_fraction.shape[0])): 
            sfr_fraction[i] = np.prod(z_fraction[:i])*(1-z_fraction[i])
    return sfr_fraction

# ----------------------------------------------------------------------------------------------------------------------


### Build a prospect.models.SedModel object ###

def build_model(**run_params):    

    ### Get (a copy of) the prepackaged model set dict (dict of dicts, keyed by parameter name) ###
    model_params = TemplateLibrary["alpha"]

    # From Leja's parameter files ----------------------------
    model_params['add_igm_absorption']= {'N': 1, 'isfree': False, 'init': 1,
                    'units': None,
                    'prior_function': None,
                    'prior_args': None}
                    # Absorbtion from intergalactic medium ~not that important
                    # Switch to include IGM absorption via Madau (1995). The zred parameter 
                    #   must be non-zero for this switch to have any effect. The optical 
                    #   depth can be scaled using the igm_factor parameter (FSPS)

    model_params['add_agb_dust_model'] = { 'N': 1, 'isfree': False, 'init': True,
                    'units': None,
                    'prior_function': None,
                    'prior_args': None}
                    # AGN pulsate and emit dust (adding to gal dust)
                    # Switch to turn on/off the AGB circumstellar dust model presented in 
                    #   Villaume (2014). NB: The AGB dust emission is scaled by the 
                    #   parameter agb_dust. (FSPS)

    # model_params['pmetals'] = { 'N': 1, 'isfree': False, 'init': -99,
    #                 'units': '',
    #                 'prior_function': None,
    #                 'prior_args': {'mini':-3, 'maxi':-1}}
                    # only needs if zcontinuous is 2: convolve with a metallicity 
                    #   distribution function at each age.The MDF is controlled by the 
                    #   parameter "pmetals"

    model_params['agebins'] = { 'N': run_params['ncomp'], 'isfree': False, 'init': run_params['agebins_init'],
                    'units': 'log(yr)',
                    'prior': None}
    
    model_params['sfr_fraction'] = {'N': run_params['ncomp'] - 1, 'isfree': False, 
                    'init': np.zeros(run_params['ncomp'] - 1)+1./run_params['ncomp'],
                    'depends_on': transform_zfraction_to_sfrfraction,
                    'units': '',
                    'prior': priors.TopHat(mini=np.full(run_params['ncomp'] - 1, 0.0), 
                        maxi=np.full(run_params['ncomp'] - 1, 1.0))}
    
    model_params['z_fraction'] = {'N': run_params['ncomp'] - 1, 'isfree': True, 
                    'init': run_params['z_fraction_init'],
                    'units': '',
                    'init_disp': 0.02,
                    'prior': priors.Beta(alpha=run_params['tilde_alpha'], 
                        beta=np.ones_like(run_params['tilde_alpha']),mini=0.0,maxi=1.0)}
    if run_params['total_mass_switch'] == False:
        print('Total_mass is off')
        model_params['total_mass'] = {'N': 1, 'isfree': False,
                        'init': 10000000000}
    elif run_params['total_mass_switch'] != True:
        # This should be empty
        empty_var_tmass = 1

    model_params['imf_type'] = {'N': 1, 'isfree': False, 'init': 1, 
                    'units': None,
                    'prior_function_name': None,
                    'prior_args': None}
                    # 1  chabrier (2003) (FSPS)

    model_params['dust_type'] = {'N': 1, 'isfree': False, 'init': 4,
                    'units': 'index',
                    'prior_function_name': None,
                    'prior_args': None}
                    # Common variable deﬁning the extinction curve for dust around old stars
                    #   4: Kriek & Conroy (2013) attenuation curve (FSPS)

    model_params['dust1'] = {'N': 1, 'isfree': False, 'depends_on': to_dust1,
                    'init': 1.0,
                    'units': '',
                    'prior': priors.TopHat(mini=0.0, maxi=6.0)}
                    # Dust parameter describing the attenuation of young stellar light (FSPS)

    model_params['dust1_fraction'] = {'N': 1, 'isfree': True, 'init': 1.0,
                    'init_disp': 0.8,
                    'disp_floor': 0.8,
                    'units': '',
                    'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    # model_params['dust_index'] = {'N': 1, 'isfree': True, 'init': 0.0,
    #                 'init_disp': 0.25,
    #                 'disp_floor': 0.15,
    #                 'units': '',
    #                 'prior': priors.TopHat(mini=-2.2, maxi=0.4)}
                    # Not needed because dust_type is 4
                    # Power law index of attenuation curve. Only used when dust_type=0 (FSPS)

    model_params['dust1_index'] = {'N': 1, 'isfree': False, 'init': -1.0,
                    'units': '',
                    'prior': priors.TopHat(mini=-1.5, maxi=-0.5)}
                    # Try init = -0.2?
                    # Power law index of the attenuation curve affecting stars younger than 
                    #   dust_tesc corresponding to dust1 (FSPS)

    model_params['dust_tesc'] = {'N': 1, 'isfree': False, 'init': 7.0,
                    'units': 'log(Gyr)',
                    'prior_function_name': None,
                    'prior_args': None }
                    # Stars younger than dust_tesc are attenuated by both dust1 and dust2, 
                    #   while stars older are attenuated by dust2 only (FSPS)
    
    model_params['add_dust_emission']= {'N': 1, 'isfree': False, 'init': 1,
                    'units': None,
                    'prior_function': None,
                    'prior_args': None}
                    # Switch to turn on/off the Draine & Li 2007 dust emission model (FSPS)

    model_params['duste_gamma'] = {'N': 1, 'isfree': True, 'init': 0.01,
                    'init_disp': 0.4,
                    'disp_floor': 0.3,
                    'units': None,
                    'prior': priors.TopHat(mini=0.0, maxi=1.0)}
                    # Parameter of the Draine & Li (2007) dust emission model. Specifies 
                    #   the relative contribution of dust heated at a radiation field 
                    #   strength of Umin and dust heated at Umin < U <= Umax (FSPS)

    model_params['duste_umin'] = {'N': 1, 'isfree': True, 'init': 1.0,
                    'init_disp': 10.0,
                    'disp_floor': 5.0,
                    'units': None,
                    'prior': priors.TopHat(mini=0.1, maxi=25.0)}
                    # Parameter of the Draine & Li (2007) dust emission model. Specifies 
                    #   the minimum radiation field strength in units of the MW value (FSPS)

    model_params['duste_qpah']= {'N': 1, 'isfree': True, 'init': 3.0,
                    'init_disp': 3.0,
                    'disp_floor': 3.0,
                    'units': 'percent',
                    'prior': priors.TopHat(mini=0.0, maxi=10.0)}
                    # Make bumps in mid
                    # Parameter of Draine & Li (2007) dust emission model. Specifies grain
                    #   size distribution through the fraction of grain mass in PAHs (FSPS)

    model_params['add_neb_emission'] = {'N': 1, 'isfree': False, 'init': True,
                    'units': r'log Z/Z_\odot',
                    'prior_function_name': None,
                    'prior_args': None}
                    # Switch to turn on/off a nebular emission model (both continuum and
                    #   line emission), based on Cloudy models from Nell Byler. Contrary 
                    #   to FSPS, this option is turned off by default (FSPS)
    
    # model_params['add_neb_continuum']= {'N': 1, 'isfree': False, 'init': True,
    #                 'units': r'log Z/Z_\odot',
    #                 'prior_function_name': None,
    #                 'prior_args': None}
                    # Not needed: Turns off automatically because add_neb_emission is on
                    # Switch to turn on/off the nebular continuum component (FSPS)

    model_params['nebemlineinspec'] = {'N': 1, 'isfree': False, 'init': False,
                    'prior_function_name': None,
                    'prior_args': None}
                    # Flag to include the emission line fluxes in the spectrum. Turning 
                    #   this off is a significant speedup in model calculation time. If 
                    #   not set, the line luminosities are still computed (FSPS)
    
    model_params['gas_logz'] = {'N': 1, 'isfree': False, 'init': 0.0,
                    'depends_on': tie_gas_logz,
                    'units': r'log Z/Z_\odot',
                    'prior': priors.TopHat(mini=-2.0, maxi=0.5)}
                    # Sharp lines above optical
                    # Log of the gas-phase metallicity; relevant only for the 
                    #   nebular emission model (FSPS)
    
    model_params['gas_logu'] = {'N': 1, 'isfree': False, 'init': -2.0,
                    'units': '',
                    'prior': priors.TopHat(mini=-4.0, maxi=-1.0)}
                    # Log of the gas ionization parameter; relevant only 
                    #   for the nebular emission model (FSPS)
    

    ######### AGN PARAMETERS ###############################################
    if run_params['AGN_switch'] == True:
        model_params['add_agn_dust'] = {'N': 1, 'isfree': False, 'init': True,
                        'units': '',
                        'prior_function_name': None,
                        'prior_args': None}
        
        model_params['fagn'] = {'N': 1, 'isfree': True, 'init': 0.05,
                        'init_disp': 0.05,      
                        'disp_floor': 0.01,
                        'units': '',
                        'prior': priors.LogUniform(mini=0.000316228, maxi=3.0)}
        
        model_params['agn_tau'] = {'N': 1, 'isfree': True, 'init': 10.0,
                        'init_disp': 10,
                        'disp_floor': 2,
                        'units': '',
                        'prior': priors.LogUniform(mini=5.0, maxi=150.0)}
    else:
        print('AGN parameters are off')
        model_params['fagn'] = {'N': 1, 'isfree': False, 'init': 0, 'units': ''}
        model_params['add_agn_dust'] = {'N': 1, 'isfree': False, 'init': True,
                        'units': '',
                        'prior_function_name': None,
                        'prior_args': None}
        model_params['agn_tau'] = {'N': 1, 'isfree': False, 'init': 10.0,
                        'init_disp': 10,
                        'disp_floor': 2,
                        'units': '',
                        'prior': priors.LogUniform(mini=5.0, maxi=150.0)}
        
    ########################################################################


    # ----------------------------------------------------------------
    # Unit Parameters (Leja)
    model_params['peraa'] = {'N': 1, 'isfree': False, 
                    'init': False}
                    # True: return the spectrum in L_sun/A. Else, return 
                    #   the spectrum in the FSPS standard L_sun/Hz (FSPS)
    model_params['mass_units'] = {'N': 1, 'isfree': False, 'init': 'mformed'}
    # ----------------------------------------------------------------

    # Original Parameters #

    model_params['logzsol'] =  { 'N': 1, 'isfree': True, 'init': -0.5,
                    'init_disp': 0.25,
                    'disp_floor': 0.2,
                    'units': r'$\log (Z/Z_\odot)$',
                    'prior': priors.TopHat(mini=-1.98, maxi=0.19)}
                    # Parameter describing the metallicity (FSPS)

    model_params['sfh'] = {'init': 4 }
                    # Defines the type of star formation history, normalized such 
                    #   that one solar mass of stars is formed over the full SFH (FSPS)
    model_params["zred"] = {'init': run_params['object_redshift'] }
                    # Redshift
                    #  If this value is non-zero and if redshift_colors=1, the 
                    #   magnitudes will be computed for the spectrum placed at redshift 
                    #   zred (FSPS)

    model_params["lumdist"] = {"N": 1, "isfree": False, "init": run_params['ldist'], 
                    "units":"Mpc"} 

    model_params['logmass'] = {'N': 1, 'isfree': True, 'init': 10.0,
                    'units': 'Msun',
                    'prior': priors.TopHat(mini=5.0, maxi=13.0)}

    model_params['mass'] = { 'N': 1, 'isfree': False, 'init': 1e10, 
                    'units': 'Msun',
                    'depends_on' : transform_logmass_to_mass,
                    'prior': priors.TopHat(mini=1e5, maxi=1e13)}    

    model_params['dust2'] = {'N': 1, 'isfree': True, 'init': 0.3,
                    'init_disp': 0.25,
                    'disp_floor': 0.15,
                    'units': '',
                    'prior': priors.TopHat(mini=0, maxi=4.0)}
                    # Dust parameter describing the attenuation of old stellar light, 
                    #   i.e. where t > dust_tesc (see Conroy et al. 2009a) (FSPS)

    # Create a fit order in order to fit in higest priority first                # From Leja
    fit_order = ['logmass', 'z_fraction', 'dust2', 'logzsol', 'dust_index', 
                    'dust1_fraction', 'duste_qpah', 'duste_gamma', 'duste_umin'] # add AGN to end

    parnames = {k: model_params[k] for k in fit_order}
    tparams = parnames

    for param in model_params:
        if param not in fit_order:
            tparams[param] = model_params[param]
    
    model_params = tparams

    model = SedModel(model_params)

    return model