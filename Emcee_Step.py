# File is intended to run on the UIUC Campus Cluster (CAPs).
# You can specify the galaxies you want to run

# In terminal:
#     To run all galaxies:        ~$ python3 PSB_AGN_CMD_Runner.py all
#     To run galaxies 1, 6, 12:   ~$ python3 PSB_AGN_CMD_Runner.py 1 6 12

# 1:    ### Import packages ### ----------------------------------------------------------------------------
# import standard python packages
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
from functools import partial
import prospect.io.read_results as reader


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


CAPS = True

if CAPS == False:
    AGN_FITS_PATH = '/mnt/c/Users/emma_d/ASTR_Research/Data/asu.fit'
    G_PATH = 'Galaxy_output'
    IRS_PATH = 'Data/ea_struct_v9.sav'
elif CAPS == True:
    AGN_FITS_PATH = '/home/elitzer3/scratch/PSB_AGN_Scratch/Data/asu.fit'
    G_PATH = '/home/elitzer3/scratch/PSB_AGN_Scratch/Galaxy_output'
    IRS_PATH = '/home/elitzer3/scratch/PSB_AGN_Scratch/Data/ea_struct_v9.sav'
else:
    print('Please set CAPS to True for running on CAPS or False for running on personal machine.')

print('{0}: Packages and Functions Loaded'.format(time.strftime("%H:%M:%S", time.localtime())))

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

def initialize_theta(galaxy_num, Template_Type, ts, AGN_data, gal_desig, Galaxy_Path, Run_Num, Num_Iters): #, input_hfile 
    if Run_Num == 0:
        # 5: ### Pull RA, DEC from data file and query SDSS ### -----------------------------------------------------
        Gal_RA, Gal_DEC = AGN_data[galaxy_num][2], AGN_data[galaxy_num][3]
        pos = coord.SkyCoord(Gal_RA, Gal_DEC, unit='deg',frame='icrs')
        xid = SDSS.query_region(pos, spectro=True)


        # 6: ### Redefine RA, DEC ### -------------------------------------------------------------------------------
        Gal_RA = xid['ra'][0]
        Gal_DEC = xid['dec'][0]

        G_Redshift = xid['z'][0]                                    # Use redshift from SDSS query
        cosmo = ap.cosmology.FlatLambdaCDM(H0=70 , Om0=0.3)         # Cosmological redshift object   
        ldist_Mpc_units = cosmo.comoving_distance(G_Redshift)       # Cosmological redshift 

        ldist_Mpc = ldist_Mpc_units.value           
        tage_of_univ = WMAP9.age(G_Redshift).value                  # Gyr


        # 7:    ### Get optical image from the url using galaxy RA and DEC ### -------------------------------------
        SDSS_scale = 0.396127                                       # arcsec/pix
        SDSS_width = 128                                            # SDSS_arcsec/SDSS_scale
        SDSS_arcsec = SDSS_width * SDSS_scale                       # arcsec 
        SDSS_deg = SDSS_arcsec/3600                                 # deg

        SDSS_center_coords = SkyCoord(ra=Gal_RA*u.degree, dec=Gal_DEC*u.degree)
        SDSS_left_coords = SkyCoord(ra=Gal_RA*u.degree - (SDSS_deg/2) * u.deg, dec=Gal_DEC*u.degree - (SDSS_deg/2) * u.deg)
        SDSS_right_coords = SkyCoord(ra=Gal_RA*u.degree + (SDSS_deg/2) * u.deg, dec=Gal_DEC*u.degree + (SDSS_deg/2) * u.deg)

        ra_center_str = '{0:.0f}h{1:.0f}m{2:.0f}s'.format(SDSS_center_coords.ra.hms[0], SDSS_center_coords.ra.hms[1], SDSS_center_coords.ra.hms[2])
        ra_left_str = '{0:.0f}h{1:.0f}m{2:.0f}s'.format(SDSS_left_coords.ra.hms[0], SDSS_left_coords.ra.hms[1], SDSS_left_coords.ra.hms[2])
        ra_right_str = '{0:.0f}h{1:.0f}m{2:.0f}s'.format(SDSS_right_coords.ra.hms[0], SDSS_right_coords.ra.hms[1], SDSS_right_coords.ra.hms[2])

        dec_center_str = '{0:.0f}h{1:.0f}m{2:.0f}s'.format(SDSS_center_coords.dec.hms[0], SDSS_center_coords.dec.hms[1], SDSS_center_coords.dec.hms[2])
        dec_left_str = '{0:.0f}h{1:.0f}m{2:.0f}s'.format(SDSS_left_coords.dec.hms[0], SDSS_left_coords.dec.hms[1], SDSS_left_coords.dec.hms[2])
        dec_right_str = '{0:.0f}h{1:.0f}m{2:.0f}s'.format(SDSS_right_coords.dec.hms[0], SDSS_right_coords.dec.hms[1], SDSS_right_coords.dec.hms[2])


        # 8:    ### Retreive IRS Data and indicate if it exists ### -------------------------------------------------
        ea_struct = readsav(IRS_PATH)['ea_struct'] 

        # gal_desig = AGN_data[galaxy_num][1]
        gal_EA_Desig = gal_desig[2:]
        s1, = np.where(ea_struct.source.astype(str) == 'Spitzer')       # Pulls only spitzer from sources
        ea_struct = ea_struct[s1]                                       # Masks ea_struct to get spitzer

        # If the IRS data contains the galaxy, graph the IRS data. If not print there is no IRS data
        if gal_EA_Desig in ea_struct['EA_DESIG'].astype(str):
            IRS_indicator = 1                                           # Indicates there is IRS data
            print('{0}: There is Spitzer IRS spectrum data for this galaxy'.format(time.strftime("%H:%M:%S", time.localtime())))
        else:
            IRS_indicator = 0                                           # Indicates there is no IRS data
            print('{0}: There is no Spitzer IRS spectrum data for this galaxy'.format(time.strftime("%H:%M:%S", time.localtime())))


        # 9:   ### Create run_params dictionary ### -----------------------------------------------------------------
        run_params = { 'ldist': ldist_Mpc,
                        'agelims': [0.0,8.0,8.5,9.0,9.5,9.8,10.0],
                        'object_redshift': G_Redshift,
                        'zcontinuous': 1,                           # Leja: 2
                        'verbose': True,                            # Leja: True, OG: False
                        'dynesty': False,                           
                        'emcee': False,                             
                        'optimize': False,                          
                        'optimization': False,
                        'min_method': 'lm',
                        'nmin': 1,                                  
                        'nwalkers': 620,                            # Leja: 620 #128 #300
                        'niter': 1024,                               # Leja: 7500 # 512
                        'nburn': [16, 32, 64],                      # Leja: [150, 200, 200] # [16, 32, 64] <-- takes ~ 30m
                        'optimization': False,

                        # 'nested_method': 'rwalk',
                        # 'nlive_init': 400,
                        # 'nlive_batch': 200,
                        # 'nested_dlogz_init': 0.05,
                        # 'nested_posterior_thresh': 0.05,
                        # 'nested_maxcall': int(1e7),

                        'interval': 0.2,
                        'nofork': True,

                        # 'convergence_check_interval': 50,
                        'convergence_chunks': 325,
                        'convergence_kl_threshold': 0.016,
                        'convergence_stable_points_criteria': 8, 
                        # 'convergence_nhist': 50,

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
                        'dec_labels': [dec_left_str, dec_center_str, dec_right_str],
                        'Run_Num': Run_Num
                        }


        # 10:   ### Built obs dict using meta-params ### ------------------------------------------------------------
        obs = build_obs(**run_params)


        # 11:   ### Define agelimits, tilde_alpha, and z_fraction_init for build_model ### --------------------------
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


        # 12:   ### Build model and sps ### -----------------------------------------------------------------------
        model = build_model(**run_params)
        sps = build_sps(**run_params)                   # Build sps object using FracSFH basis: From Charlie Konroy

        # Save number of free parameters and number of data points to run_params
        run_params['num_free_params'] = len(model.free_params)
        run_params['free_params'] = model.free_params 
        run_params['num_data_points'] = len(obs['maggies'][obs["phot_mask"]])

            
        # 13:   ### Likelyhood Function ### -----------------------------------------------------------------------
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


        # 14:   ### Minimization ### ------------------------------------------------------------------------------
        # Uses Levenberg-Marquardt, needed parameters in run_params requires a likelihood function that returns a vector of chi values
        # --- start minimization ---- # 
        print('{0}: Begin optimization for {1}'.format(time.strftime("%H:%M:%S", time.localtime()), run_params['ID']))
        output = fit_model(obs, model, sps, lnprobfn=prospect.fitting.lnprobfn, **run_params)
        print("{0}: Done optmization in {1}m for {2}".format(time.strftime("%H:%M:%S", time.localtime()), output["optimization"][1]/60, run_params['ID']))


        # 15:   ### Plot prediction SED with initial SED ### -------------------------------------------------------
        (results, topt) = output["optimization"]
        ind_best = np.argmin([r.cost for r in results]) # Find which of the minimizations gave the best result and use the parameter vector for that minimization
        print('{0}: ind_best = {1}'.format(time.strftime("%H:%M:%S", time.localtime()), ind_best))
        theta_prediction = results[ind_best].x.copy()


        # 16:   ### Rebuild model and obs for MCMC ### ---------------------------------------------------------------
        # Turn AGN parameters back on, turn on emcee
        run_params['AGN_switch'] = True
        run_params['total_mass_switch'] = True
        run_params["optimize"] = False
        run_params["emcee"] = True


    else: 
        # hfile = Galaxy_Path + 'G{0}_{1}_{2}_{3}'.format(galaxy_num, gal_desig, ts, Run_Num)
        # hfile = Galaxy_Path + 'G11_EAH03_21Aug05_0.h5'
        # hfile = Galaxy_Path + input_hfile
        hfile = Galaxy_Path + 'G{0}_{1}_{2}.h5'.format(galaxy_num, gal_desig, Run_Num -1)
        result = reader.results_from(hfile, dangerous = False)[0]
        run_params = result['run_params']
        run_params['Run_Num'] = Run_Num
        run_params['nburn'] = [2, 2, 4]
        run_params['niter'] = Num_Iters

        theta_prediction = result['bestfit']['parameter']

    return theta_prediction, run_params

### -------------------------------------------------------------------------------------------------------------------
### -------------------------------------------------------------------------------------------------------------------
### -------------------------------------------------------------------------------------------------------------------

def PSB_AGN_CAPS_Funct(galaxy_num, Run_Num, Template_Type, Num_Iters): #input_hfile
    # 3:    ### Start Timer and Import full galaxy file for all 58 galaxies ### ---------------------------------
    start_time = time.time()

    AGN_file = fits.open(AGN_FITS_PATH)
    AGN_data = AGN_file[1].data
    gal_desig = AGN_data[galaxy_num][1]


    # 4:    ### Choose a galaxy (0 to 57) ### -------------------------------------------------------------------
    galaxy_num = galaxy_num
    Template_Type = Template_Type

    # Create galaxy file to store plots and hdf5 data file
    if not os.path.exists('{0}/G{1}/'.format(G_PATH, galaxy_num)): #'Galaxy_output/G{}/'
        os.mkdir('{0}/G{1}/'.format(G_PATH, galaxy_num))
    Galaxy_Path = '{0}/G{1}/'.format(G_PATH, galaxy_num)

    print('{0}: This is for Galaxy {1} \t\t {2}'.format(time.strftime("%H:%M:%S", time.localtime()), galaxy_num, gal_desig))

    ts = time.strftime("%y%b%d", time.localtime())
    print('{0}: The Date is {1}'.format(time.strftime("%H:%M:%S", time.localtime()), ts))
    print('{0}: The template type is {1}'.format(time.strftime("%H:%M:%S", time.localtime()), Template_Type))


    theta_prediction, run_params = initialize_theta(galaxy_num, Template_Type, ts, AGN_data, gal_desig, Galaxy_Path, Run_Num, Num_Iters) #, input_hfile
    
    print("{0} theta_prediction = {1}".format(time.strftime("%H:%M:%S", time.localtime()), theta_prediction))


    # Re-run build functions with AGN
    obs = build_obs(**run_params)
    sps = build_sps(**run_params)
    model = build_model(**run_params)


    # 17:   ### Run MCMC with emcee ### ------------------------------------------------------------------------
    lnprobfn_fixed = partial(prospect.fitting.lnprobfn, sps=sps)

    # --- Run MCMC --- #
    print('{1}: Start emcee for {0}'.format(run_params['ID'], time.strftime("%H:%M:%S", time.localtime())))
    print('\tniter:', run_params['niter'])
    print('\tnwalkers:', run_params['nwalkers'])

    output = fit_model(obs, model, sps, lnprobfn=lnprobfn_fixed, **run_params)
    print('{2}: Finished emcee in {0:.2f}m for {1}'.format(output["sampling"][1]/60, run_params['ID'], time.strftime("%H:%M:%S", time.localtime())))


    # 18:   ### Create file path and re-run build functions ### ------------------------------------------------
    # if input_hfile == None:
    #     hfile = Galaxy_Path + 'G{0}_{1}_{2}.h5'.format(galaxy_num, gal_desig, Run_Num)
    # else:
    #     temp_hfile = Galaxy_Path + 'G{0}_{1}_{2}.h5'.format(galaxy_num, gal_desig, Run_Num)
    #     hfile = hfile = Galaxy_Path + 'G{0}_{1}_{2}.h5'.format(galaxy_num, gal_desig, Run_Num)
    hfile = Galaxy_Path + 'G{0}_{1}_{2}.h5'.format(galaxy_num, gal_desig, Run_Num)
    obs, model, sps = build_obs(**run_params),  build_model(**run_params), build_sps(**run_params)


    # 19:   ### Save results to h5 File ### -------------------------------------------------------------------
    writer.write_hdf5(hfile, run_params, model, obs, 
                    output["sampling"][0], output["optimization"][0],
                    tsample=output["sampling"][1],
                    toptimize=output["optimization"][1],
                    sps=sps)
    print('{0}: Finished writing {1} file'.format(time.strftime("%H:%M:%S", time.localtime()), hfile))


    # 20: ### Print time it takes to run ### -------------------------------------------------------------------
    end_time = time.time()
    print('{0}: This program takes:\n\t {1:.2f} \tsecs\n\t {2:.2f} \tmins\n\t {3:.2f} \thours'.format(time.strftime("%H:%M:%S", time.localtime()), (end_time - start_time), (end_time - start_time)/60, (end_time - start_time)/60/60))


Galaxy_list = sys.argv[1]
Run_Num = sys.argv[2]
Num_Iters = 1200  #sys.argv[3:]

# try:
#     sys.argv[3:]
#     [input_hfile] = sys.argv[3:]
# except NameError:
#     input_hfile = None

print('{0}: Running PSB_AGN_CMD_Runner.py for Galaxy: \t {1}'.format(time.strftime("%H:%M:%S", time.localtime()), Galaxy_list))
# print('{0}: Input h5 file: \t {1}'.format(time.strftime("%H:%M:%S", time.localtime()))) #, input_hfile

PSB_AGN_CAPS_Funct(galaxy_num = int(Galaxy_list), Run_Num=int(Run_Num), Template_Type = 'Test_step' , Num_Iters=Num_Iters) #, input_hfile=input_hfile
