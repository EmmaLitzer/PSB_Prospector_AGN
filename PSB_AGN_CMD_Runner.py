
# File is intended to run on the UIUC Campus Cluster (CAPs).
# You can specify the galaxies you want to run

# In terminal:
#     To run all galaxies:        ~$ python3 PSB_AGN_CMD_Runner.py all
#     To run galaxies 1, 6, 12:   ~$ python3 PSB_AGN_CMD_Runner.py 1 6 12


# 1:    ### Import packages ### ----------------------------------------------------------------------------
import time, sys, os
import numpy as np
import astropy as ap
from matplotlib.pyplot import *
import seaborn as sns

# Import packages to open fits and calculate cosmology
from astropy.io import fits
from astropy.cosmology import WMAP9
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.sdss import SDSS
from scipy.io import readsav

# Import prospector 
# sys.path.insert(0, '/mnt/c/Users/emma_d/ASTR_Research/lib/python3.8/site-packages/repo/prospector/')
import prospect     
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.plotting import corner
# import corner

# Import my build functions
from build_model_funct import build_model
from build_obs_funct import build_obs
from FracSFH_ import FracSFH

# re-defining plotting defaults for matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
my_cmap = sns.color_palette("tab10")
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'xtick.color': 'k'})
rcParams.update({'ytick.color': 'k'})
rcParams.update({'font.size': 30})
rcParams.update({'axes.facecolor':'w'})
rcParams.update({'savefig.facecolor':'w'})
rcParams.update({'lines.linewidth':'0.7'})
rcParams.update({'axes.prop_cycle': cycler(color = sns.color_palette("tab10"))})
rcParams.update({'figure.dpi': 300})
rcParams.update({'savefig.dpi': 300})

# Turn off warnings to clear up terminal when running
import warnings
warnings.filterwarnings('ignore', message='.*encountered in log.*')
warnings.filterwarnings('ignore', message='.*Reading unicode strings without specifying.*')
warnings.filterwarnings('ignore', message='.*Invalid keyword for column 66.*')
warnings.filterwarnings('ignore', message='.*Could not store paramfile text.*')
warnings.filterwarnings('ignore', message='.*Could not JSON serialize.*')

# package versions 
# vers = (np.__version__, scipy.__version__, h5py.__version__, fsps.__version__, prospect.__version__)
# print("numpy: {}\nscipy: {}\nh5py: {}\nfsps: {}\nprospect: {}".format(*vers))


# 2:    ### Define constants and functions ### ------------------------------------------------------------- 
# Constants:
lsun = 3.846e33
pc = 3.085677581467192e18                   # in cm
lightspeed = 2.998e18                       # A/s
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2)
jansky_mks = 1e-26

# My Unit Functions:
def mJy_to_maggies(mJy):
    """ Converts mJy to maggies
    """
    conversion_mJy_Jy = .001
    conversion_mJy_maggie = 3631
    return mJy * conversion_mJy_Jy/conversion_mJy_maggie

def f_nu_to_f_lambda_maggie(lam, f_nu):
    """ Converts f_nu (ergs) to f_lambda (maggies)
    """
    f_lambda_ergs = (10**-17 * f_nu) * (lam**2)/lightspeed
    f_lambda_mJy = (f_lambda_ergs / (1E-23)) * 1000
    f_lambda_maggie = mJy_to_maggies(f_lambda_mJy)
    return f_lambda_maggie

# From GitHub:
def get_best_v2(res, **kwargs):
    """ Get the maximum a posteriori parameters.
    """
    imax = np.argmax(res['lnprobability'])
    try:
        i, j = np.unravel_index(imax, res['lnprobability'].shape)
        theta = res['chain'][i, j, :].copy()
    except(ValueError):
        theta = res['chain'][imax, :].copy()
        
    return  theta

def build_sps(zcontinuous=1, **extras):
    """ Build sps object using FracSFH basis
    """
    sps = FracSFH(zcontinuous = 1, **extras)        
    return sps

def zfrac_to_masses(total_mass=None, z_fraction=None, agebins=None, **extras):
    """This transforms from independent dimensionless `z` variables to sfr
    fractions and then to bin mass fractions. The transformation is such that
    sfr fractions are drawn from a Dirichlet prior.  See Betancourt et al. 2010
    and Leja et al. 2017
    :param total_mass:
        The total mass formed over all bins in the SFH.
    :param z_fraction:
        latent variables drawn form a specific set of Beta distributions. (see
        Betancourt 2010)
    :returns masses:
        The stellar mass formed in each age bin.
    """
    # sfr fractions
    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)):
        sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])

    # convert to mass fractions
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    mass_fraction = sfr_fraction * np.array(time_per_bin)
    mass_fraction /= mass_fraction.sum()

    masses = total_mass * mass_fraction
    return masses

def zfrac_to_sfr(total_mass=None, z_fraction=None, agebins=None, **extras):
    """This transforms from independent dimensionless `z` variables to SFRs.
    :returns sfrs:
        The SFR in each age bin (msun/yr).
    """
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    masses = zfrac_to_masses(total_mass, z_fraction, agebins)
    return masses / time_per_bin

def best_vals(result_list):
    q_16, q_50, q_84 = corner.quantile(result_list, [0.16, 0.5, 0.84]) 
    dx_down, dx_up = q_50-q_16, q_84-q_50
    # return (np.around(q_50,2), np.around(dx_up,2), np.around(dx_down,2))
    return q_50, dx_up, dx_down

def PSB_AGN_CAPS_Funct(galaxy_num, Template_Type):
    # 3:    ### Start Timer ### ---------------------------------------------------------------------------------
    start_time = time.time()


    # 4:    ### Import full galaxy file for all 58 galaxies ### --------------------------------------------------
    AGN_file = fits.open('/home/elitzer3/scratch/PSB_AGN_Scratch/Data/asu.fit') # '/mnt/c/Users/emma_d/ASTR_Research/Data/asu.fit'
    AGN_data = AGN_file[1].data


    # 5:    ### Choose a galaxy (0 to 57) ### -------------------------------------------------------------------
    galaxy_num = galaxy_num
    Template_Type = Template_Type

    # Create galaxy file to store plots and hdf5 data file
    
    if not os.path.exists('/home/elitzer3/scratch/PSB_AGN_Scratch//Galaxy_output/G{}/'.format(galaxy_num)): #'Galaxy_output/G{}/'
        os.mkdir('/home/elitzer3/scratch/PSB_AGN_Scratch//Galaxy_output/G{}/'.format(galaxy_num))
    Galaxy_Path = '/home/elitzer3/scratch/PSB_AGN_Scratch//Galaxy_output/G{}/'.format(galaxy_num)

    print('{0}: This is for Galaxy {1}'.format(time.strftime("%H:%M:%S", time.localtime()), galaxy_num))

    ts = time.strftime("%y%b%d", time.localtime())
    print('{0}: The Date is {1}'.format(time.strftime("%H:%M:%S", time.localtime()), ts))
    print('{0}: The template type is {1}'.format(time.strftime("%H:%M:%S", time.localtime()), Template_Type))


    # 6: ### Pull RA, DEC from data file and query SDSS ### -----------------------------------------------------
    Gal_RA, Gal_DEC = AGN_data[galaxy_num][2], AGN_data[galaxy_num][3]
    pos = coord.SkyCoord(Gal_RA, Gal_DEC, unit='deg',frame='icrs')
    xid = SDSS.query_region(pos, spectro=True)


    # 7: ### Redefine RA, DEC ### -------------------------------------------------------------------------------
    Gal_RA = xid['ra'][0]
    Gal_DEC = xid['dec'][0]

    G_Redshift = xid['z'][0]                                    # Use redshift from SDSS query
    cosmo = ap.cosmology.FlatLambdaCDM(H0=70 , Om0=0.3)         # Cosmological redshift object   
    ldist_Mpc_units = cosmo.comoving_distance(G_Redshift)       # Cosmological redshift 

    ldist_Mpc = ldist_Mpc_units.value           
    tage_of_univ = WMAP9.age(G_Redshift).value                  # Gyr


    # 8:    ### Get optical image from the url using galaxy RA and DEC ### -------------------------------------
    SDSS_scale = 0.396127                   # arcsec/pix
    SDSS_width = 128                        # SDSS_arcsec/SDSS_scale
    SDSS_arcsec = SDSS_width * SDSS_scale   # arcsec 
    SDSS_deg = SDSS_arcsec/3600             # deg

    SDSS_center_coords = SkyCoord(ra=Gal_RA*u.degree, dec=Gal_DEC*u.degree)
    SDSS_left_coords = SkyCoord(ra=Gal_RA*u.degree - (SDSS_deg/2) * u.deg, dec=Gal_DEC*u.degree - (SDSS_deg/2) * u.deg)
    SDSS_right_coords = SkyCoord(ra=Gal_RA*u.degree + (SDSS_deg/2) * u.deg, dec=Gal_DEC*u.degree + (SDSS_deg/2) * u.deg)

    ra_center_str = '{0:.0f}h{1:.0f}m{2:.0f}s'.format(SDSS_center_coords.ra.hms[0], SDSS_center_coords.ra.hms[1], SDSS_center_coords.ra.hms[2])
    ra_left_str = '{0:.0f}h{1:.0f}m{2:.0f}s'.format(SDSS_left_coords.ra.hms[0], SDSS_left_coords.ra.hms[1], SDSS_left_coords.ra.hms[2])
    ra_right_str = '{0:.0f}h{1:.0f}m{2:.0f}s'.format(SDSS_right_coords.ra.hms[0], SDSS_right_coords.ra.hms[1], SDSS_right_coords.ra.hms[2])

    dec_center_str = '{0:.0f}h{1:.0f}m{2:.0f}s'.format(SDSS_center_coords.dec.hms[0], SDSS_center_coords.dec.hms[1], SDSS_center_coords.dec.hms[2])
    dec_left_str = '{0:.0f}h{1:.0f}m{2:.0f}s'.format(SDSS_left_coords.dec.hms[0], SDSS_left_coords.dec.hms[1], SDSS_left_coords.dec.hms[2])
    dec_right_str = '{0:.0f}h{1:.0f}m{2:.0f}s'.format(SDSS_right_coords.dec.hms[0], SDSS_right_coords.dec.hms[1], SDSS_right_coords.dec.hms[2])


    # 9:    ### Retreive IRS Data and indicate if it exists ### -------------------------------------------------
    ea_struct = readsav('/home/elitzer3/scratch/PSB_AGN_Scratch/Data/ea_struct_v9.sav')['ea_struct'] #'Data/ea_struct_v9.sav'

    gal_desig = AGN_data[galaxy_num][1]
    gal_EA_Desig = gal_desig[2:]
    s1, = np.where(ea_struct.source.astype(str) == 'Spitzer')       # Pulls only spitzer from sources
    ea_struct = ea_struct[s1]                                       # Masks ea_struct to get spitzer

    # If the IRS data contains the galaxy, graph the IRS data. If not print there is no IRS data.
    if gal_EA_Desig in ea_struct['EA_DESIG'].astype(str):
        IRS_indicator = 1                                           # Indicates there is IRS data
        print('{0}: There is Spitzer IRS spectrum data for this galaxy'.format(time.strftime("%H:%M:%S", time.localtime())))
    else:
        IRS_indicator = 0                                           # Indicates there is no IRS data
        print('{0}: There is no Spitzer IRS spectrum data for this galaxy'.format(time.strftime("%H:%M:%S", time.localtime())))


    # 10:   ### Create run_params dictionary ### -----------------------------------------------------------------
    run_params = { 'ldist': ldist_Mpc,
                    'agelims': [0.0,8.0,8.5,9.0,9.5,9.8,10.0],
                    'object_redshift': G_Redshift,
                    'zcontinuous': 1,                           # Leja: 2
                    'verbose': False,                           # Leja: True
                    'dynesty': False,                           
                    'emcee': False,                             
                    'optimize': False,                          
                    'optimization': False,
                    'min_method': 'lm',
                    'nmin': 5,                                 # initially 2, can try 1 # 5
                    'nwalkers': 128,                            # Leja: 620 #128 #300
                    'niter': 300,                               # Leja: 7500 # 512
                    'nburn': [16, 32, 64],                      # Leja: [150, 200, 200] # [16, 32, 64]
                    'optimization': False,
                    'nested_method': 'rwalk',
                    'nlive_init': 400,
                    'nlive_batch': 200,
                    'nested_dlogz_init': 0.05,
                    'nested_posterior_thresh': 0.05,
                    'nested_maxcall': int(1e7),

                    'objname': 'G{0}_{1}'.format(galaxy_num, gal_desig),
                    'initial_disp': 0.1,
                    'AGN_switch': True,
                    'tage_of_univ': tage_of_univ,
                    'Template_Type': Template_Type,
                    'galaxy_num': galaxy_num,
                    'IRS_indicator': IRS_indicator,
                    'SDSS_Query': xid,
                    'total_mass_switch': True,
                    'gal_desig': gal_desig,
                    'ts': ts,
                    'ID': 'G{0}_{1}_{2}_{3}'.format(galaxy_num, gal_desig, Template_Type, ts),
                    'ra_labels': [ra_left_str, ra_center_str, ra_right_str],
                    'dec_labels': [dec_left_str, dec_center_str, dec_right_str]
                    
                    # 'nofork': True,
                    # 'ftol': 0.5e-5
                    # 'maxfev': 5000,
                    # 'interval': 0.2,
                    # 'convergence_check_interval': 50,
                    # 'convergence_chunks': 325,
                    # 'convergence_kl_threshold': 0.016,
                    # 'convergence_stable_points_criteria': 8, 
                    # 'convergence_nhist': 50,
                    # 'compute_vega_mags': False,
                    # 'interp_type': 'logarithmic'
                    }


    # 11:   ### Built obs dict using meta-params ### ------------------------------------------------------------
    obs = build_obs(**run_params)


    # 12:   ### Define agelimits, tilde_alpha, and z_fraction_init for build_model ### --------------------------
    # Determine agelimits for SFR
    agelims = run_params['agelims']
    agelims[-1] = np.log10(tage_of_univ * 1e9)
    agebins = np.array([agelims[:-1], agelims[1:]])
    ncomp = agebins.shape[1]

    tilde_alpha = np.array([ncomp - i for i in list(range(1, ncomp))])
    z_fraction_init = np.array([(i-1)/float(i) for i in range(ncomp,1,-1)])

    # Save prospector compatible arrays to run_params
    run_params['ncomp'] = ncomp
    run_params['agebins_init'] = agebins.T
    run_params['tilde_alpha'] = tilde_alpha
    run_params['z_fraction_init'] = z_fraction_init


    # 13:   ### Build and view model ### ------------------------------------------------------------------
    model = build_model(**run_params)

    # Save number of free parameters and number of data points to run_params
    run_params['num_free_params'] = len(model.free_params)
    run_params['free_params'] = model.free_params 
    run_params['num_data_points'] = len(obs['maggies'][obs["phot_mask"]])


    # 14:   ### Build sps object using FracSFH basis ### -------------------------------------------------------
    sps = build_sps(**run_params)               # Charlie Konroy


    # 15:   ### View Model ### ---------------------------------------------------------------------------------
    ### prediction for the data from any set of model params ###
    ### Generate the model SED with the 'init' parameter values in model
    theta = model.theta.copy()
    initial_spec, initial_phot, initial_mfrac = model.sed(theta, obs=obs, sps=sps)


    # 16:   ### Likelyhood Function ### -----------------------------------------------------------------------
    # Turn off AGN parameters so that all galaxies can have a Xi2 prediction
    run_params['AGN_switch'] = False
    run_params['total_mass_switch'] = False
    run_params["emcee"] = False
    run_params["optimize"] = True

    ### Running Prospector ###
    # Fit the model using chi squared minimization and Ensemble MCMC sampler around best location from the minimization 

    # Run all building functions
    obs = build_obs(**run_params)
    sps = build_sps(**run_params)
    model = build_model(**run_params)

    # For fsps based sources it is useful to know which stellar isochrone and spectral library
    # print(sps.ssp.libraries)
    # help(fit_model)


    # 17:   ### Minimization ### ------------------------------------------------------------------------------
    # Uses Levenberg-Marquardt, needed parameters in run_params requires a likelihood function that returns a vector of chi values
    # --- start minimization ---- # 
    print('{0}: Begin optimization for {1}'.format(time.strftime("%H:%M:%S", time.localtime()), run_params['ID']))
    output = fit_model(obs, model, sps, lnprobfn=prospect.fitting.lnprobfn, **run_params)
    print("{0}: Done optmization in {1}m for {2}".format(time.strftime("%H:%M:%S", time.localtime()), output["optimization"][1]/60, run_params['ID']))


    # 18:   ### Plot prediction SED with initial SED ### -------------------------------------------------------
    (results, topt) = output["optimization"]
    ind_best = np.argmin([r.cost for r in results]) # Find which of the minimizations gave the best result and use the parameter vector for that minimization
    print('{0}: ind_best = {1}'.format(time.strftime("%H:%M:%S", time.localtime()), ind_best))
    theta_prediction = results[ind_best].x.copy()
    print("{0} theta_prediction = {1}".format(time.strftime("%H:%M:%S", time.localtime()), theta_prediction))


    # 19:   ### Rebuild model and obs for MCMC ### ---------------------------------------------------------------
    # Turn AGN parameters back on, turn on emcee
    run_params['AGN_switch'] = True
    run_params['total_mass_switch'] = True
    run_params["optimize"] = False
    run_params["emcee"] = True

    # Re-run build functions with AGN
    obs = build_obs(**run_params)
    sps = build_sps(**run_params)
    model = build_model(**run_params)


    # 20:   ### Run MCMC with emcee ### ------------------------------------------------------------------------
    # Print run_params
    # print('run_params:')
    # for i, item1 in enumerate(run_params):
    #     print(item1, ':\t', run_params[item1])

    # --- Run MCMC --- #
    print('{1}: Start emcee for {0}'.format(run_params['ID'], time.strftime("%H:%M:%S", time.localtime())))
    output = fit_model(obs, model, sps, lnprobfn=prospect.fitting.lnprobfn, **run_params)
    print('{2}: done emcee in {0:.2f}m for {1}'.format(output["sampling"][1]/60, run_params['ID'], time.strftime("%H:%M:%S", time.localtime())))


    # 21:   ### Create file path and re-run build functions ### ------------------------------------------------
    hfile = Galaxy_Path + '{}_mcmc.h5'.format(run_params['ID'])
    obs, model, sps = build_obs(**run_params),  build_model(**run_params), build_sps(**run_params)


    # 22:   ### Save results to h5 File ### -------------------------------------------------------------------
    writer.write_hdf5(hfile, run_params, model, obs, 
                    output["sampling"][0], output["optimization"][0],
                    tsample=output["sampling"][1],
                    toptimize=output["optimization"][1],
                    sps=sps)
    print('{0}: Finished writing h5 file'.format(time.strftime("%H:%M:%S", time.localtime())))


    # 23: ### Print time it takes to run ### -------------------------------------------------------------------
    end_time = time.time()
    print('{0}: This program takes:\n\t {1:.2f} \tsecs\n\t {2:.2f} \tmins\n\t {3:.2f} \thours'.format(time.strftime("%H:%M:%S", time.localtime()), (end_time - start_time), (end_time - start_time)/60, (end_time - start_time)/60/60))
    print('{0}: Finished'.format(time.strftime("%H:%M:%S", time.localtime())))

print('{0}: Packages and Functions Loaded'.format(time.strftime("%H:%M:%S", time.localtime())))


Galaxy_list = sys.argv[1:]
print('{0}: Running PSB_AGN_CMD_Runner.py for Galaxies: \t {1}'.format(time.strftime("%H:%M:%S", time.localtime()), Galaxy_list))

if str(Galaxy_list[0]) == 'all':
    for i in range(0, 58):
        PSB_AGN_CAPS_Funct(galaxy_num = i, Template_Type = 'Test_CAPS_All_Galaxy_Run')

else:
    for i in range(0, len(Galaxy_list)):
        PSB_AGN_CAPS_Funct(galaxy_num = int(Galaxy_list[i]), Template_Type = 'Test_CAPS_Limited_Galaxy_Run' )