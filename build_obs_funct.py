import sedpy    # contains some routines for computing projecting spectra onto filter bandpasses
from sedpy import observate
from astropy.io import fits
from prospect.utils.obsutils import fix_obs
import numpy as np

def mJy_to_maggies(mJy):
    conversion_mJy_Jy = .001
    conversion_mJy_maggie = 3631
    return mJy * conversion_mJy_Jy/conversion_mJy_maggie

# -----------------------------------------------------------------------------

CAPS = False

if CAPS == True:
    AGN_FITS_PATH = '/home/elitzer3/scratch/PSB_AGN_Scratch/Data/asu.fit' 
else:
    AGN_FITS_PATH = '/mnt/c/Users/emma_d/ASTR_Research/Data/asu.fit'

AGN_file = fits.open(AGN_FITS_PATH)
AGN_data = AGN_file[1].data

# -----------------------------------------------------------------------------

### Flux data ###
FFUV = mJy_to_maggies(AGN_data.field(4))
FNUV = mJy_to_maggies(AGN_data.field(6))
Fu = mJy_to_maggies(AGN_data.field(8))
Fg = mJy_to_maggies(AGN_data.field(10))
Fr = mJy_to_maggies(AGN_data.field(12))
Fi = mJy_to_maggies(AGN_data.field(14))
Fz = mJy_to_maggies(AGN_data.field(16))
FJ = mJy_to_maggies(AGN_data.field(18))
FH = mJy_to_maggies(AGN_data.field(20))
FKs = mJy_to_maggies(AGN_data.field(22))
F3_4 = mJy_to_maggies(AGN_data.field(24))
F4_6 = mJy_to_maggies(AGN_data.field(26))
F12 = mJy_to_maggies(AGN_data.field(28))
F22 = mJy_to_maggies(AGN_data.field(31))
F70 = mJy_to_maggies(AGN_data.field(33))
F100 = mJy_to_maggies(AGN_data.field(35))
F160 = mJy_to_maggies(AGN_data.field(37))
F250 = mJy_to_maggies(AGN_data.field(39))
F350 = mJy_to_maggies(AGN_data.field(42))
F500 = mJy_to_maggies(AGN_data.field(45))

Flux_Data = np.column_stack((FFUV, FNUV, Fu, Fg, Fr, Fi, Fz, F3_4, F4_6, F12, F22, F70, F100, F160, F250, F350, F500, FJ, FH, FKs))

### Filter uncertainty in mJy ###
FFUVe = mJy_to_maggies(AGN_data.field(5))
FNUVe = mJy_to_maggies(AGN_data.field(7))
Fue = mJy_to_maggies(AGN_data.field(9))
Fge = mJy_to_maggies(AGN_data.field(11))
Fre = mJy_to_maggies(AGN_data.field(13))
Fie = mJy_to_maggies(AGN_data.field(15))
Fze = mJy_to_maggies(AGN_data.field(17))
FJe = mJy_to_maggies(AGN_data.field(19))
FHe = mJy_to_maggies(AGN_data.field(21))
FKse = mJy_to_maggies(AGN_data.field(23))
F3_4e = mJy_to_maggies(AGN_data.field(25))
F4_6e = mJy_to_maggies(AGN_data.field(27))
F12e = mJy_to_maggies(AGN_data.field(29))
F22e = mJy_to_maggies(AGN_data.field(32))
F70e = mJy_to_maggies(AGN_data.field(34))
F100e = mJy_to_maggies(AGN_data.field(36))
F160e = mJy_to_maggies(AGN_data.field(38))
F250e = mJy_to_maggies(AGN_data.field(40))
F350e = mJy_to_maggies(AGN_data.field(43))
F500e = mJy_to_maggies(AGN_data.field(46))

# -----------------------------------------------------------------------------



### Build a dictionary of observational data to use in fit ###
def build_obs( **run_params):
    obs = {}

    # Filters (same order as photometric data) ###
    galex = ['galex_FUV', 'galex_NUV']
    galex_unc = FFUVe, FNUVe

    sdss = ['sdss_{0}0'.format(b) for b in ['u','g','r','i','z']]
    sdss_unc = Fue, Fge, Fre, Fie, Fze

    allWise = ['wise_w{0}'.format(b) for b in ['1','2','3','4']]
    allWise_unc = F3_4e, F4_6e, F12e, F22e

    herschel_pacs = ['herschel_pacs_{0}'.format(b) for b in ['70','100','160']]
    herschel_pacs_unc = F70e, F100e, F160e

    herschel_spire = ['herschel_spire_{0}'.format(b) for b in ['250','350','500']]
    herschel_spire_unc = F250e, F350e, F500e

    twomass = ['twomass_{0}'.format(b) for b in ['H','J','Ks']]          #Turned off 2mass
    twomass_unc = FHe, FJe, FKse

    # Put all filters into an array
    filternames = galex + sdss + allWise + herschel_pacs + herschel_spire + twomass
    obs["filters"] = sedpy.observate.load_filters(filternames)

    # Flux data (in same order as filters) in maggies 
    obs["maggies"] = Flux_Data[run_params['galaxy_num']]

    # Uncertainties (same order as filters) in maggies 
    uncertainty_cols = galex_unc, sdss_unc, allWise_unc, herschel_pacs_unc, herschel_spire_unc , twomass_unc

    maggies_uncertainy = np.vstack(uncertainty_cols).T      #transposed to get the row data not the column data
    obs["maggies_unc"] = maggies_uncertainy[run_params['galaxy_num']]

    obs["phot_mask"] = np.full((len(Flux_Data[run_params['galaxy_num']])), True, dtype=bool)  # use all photomectric data
    # obs["phot_mask"] = np.ones((len(Flux_Data[galaxy_num])), dtype=bool)
    # obs["phot_mask"] = obs['maggies'] != 0

    for i in range(0, len(obs["maggies"])):
        if obs["maggies"][i] != 0 and obs["maggies_unc"][i] == 0:
            """ If a flux value is a upper limit instead of a proper flux, 
                redefine flux as 0 and uncertainty as the 1σ value.
            """
            obs["maggies_unc"][i] = obs["maggies"][i]/5            # to recieve 1σ from 5σ (Li limit flags!)
            obs["maggies"][i] = 0




    ### Array of effective wavelengths for each filter (not necessary, but can be useful for plotting) ###
    obs["phot_wave"] = np.array([f.wave_effective for f in obs["filters"]])

    obs["wavelength"] = None    # vector of vaccum λs in angstroms
    obs["spectrum"] = None      # in maggies
    obs['unc'] = None           # spectral uncertainties
    obs['mask'] = None

    obs = fix_obs(obs)      # Ensures all required keys are present in the obs dictionary

    return obs