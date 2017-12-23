# ---------------------------------------------------------#
#   astroNN.datasets.h5: compile h5 files for NN
# ---------------------------------------------------------#

import os
import time
from functools import reduce

import h5py
import numpy as np
from astropy.io import fits

import astroNN.apogee.downloader
import astroNN.datasets.xmatch
from astroNN.gaia.downloader import tgas_load
from astroNN.apogee.chips import gap_delete, continuum
from astroNN.apogee.apogee_shared import apogee_env, apogee_default_dr
from astroNN.apogee.downloader import combined_spectra, visit_spectra
from astroNN.gaia.gaia_shared import gaia_env, mag_to_absmag
from astroNN.shared.nn_tools import h5name_check

currentdir = os.getcwd()
_APOGEE_DATA = apogee_env()
_GAIA_DATA = gaia_env()


class H5Compiler():
    """
    A class for compiling h5 dataset for Keras to use
    """

    def __init__(self):
        self.apogee_dr = None
        self.gaia_dr = None
        self.starflagcut = True
        self.aspcapflagcut = True
        self.vscattercut = 1
        self.teff_high = 5500
        self.teff_low = 4000
        self.SNR_low = 200
        self.SNR_high = 99999
        self.ironlow = -3
        self.h5_filename = None
        self.reduce_size = False  # True to filter out all -9999
        self.cont_mask = None  # Continuum Mask
        self.use_apogee = True
        self.use_esa_gaia = False
        self.use_anderson = True
        self.use_all = False
        self.target = 'all'
        self.err_info = True  # Whether to include error information in h5 dataset
        self.continuum = True # True to do continuum normalization, False to use aspcap normalized spectra

    def load_allstar(self):
        allstarpath = astroNN.apogee.downloader.allstar(dr=self.apogee_dr)
        hdulist = fits.open(allstarpath)
        print('Loaded allStar DR{} catalog'.format(self.apogee_dr))
        return hdulist

    def filter_apogeeid_list(self, hdulist):
        vscatter = hdulist[1].data['VSCATTER']
        SNR = hdulist[1].data['SNR']
        location_id = hdulist[1].data['LOCATION_ID']
        teff = hdulist[1].data['PARAM'][:, 0]
        logg = hdulist[1].data['PARAM'][:, 1]
        Fe = hdulist[1].data['X_H'][:, 17]
        K = hdulist[1].data['K']

        total = range(len(SNR))

        if self.starflagcut is True:
            starflag = hdulist[1].data['STARFLAG']
            fitlered_starflag = np.where(starflag == 0)[0]
        else:
            fitlered_starflag = total

        if self.aspcapflagcut is True:
            aspcapflag = hdulist[1].data['ASPCAPFLAG']
            fitlered_aspcapflag = np.where(aspcapflag == 0)[0]
        else:
            fitlered_aspcapflag = total

        fitlered_temp_lower = np.where((self.teff_low <= teff))[0]
        fitlered_temp_upper = np.where((self.teff_high >= teff))[0]
        fitlered_vscatter = np.where(vscatter < self.vscattercut)[0]
        fitlered_Fe = np.where(Fe > self.ironlow)[0]
        fitlered_logg = np.where(logg != -9999)[0]
        fitlered_snrlow = np.where(SNR > self.SNR_low)[0]
        fitlered_snrhigh = np.where(SNR < self.SNR_high)[0]
        fitlered_K = np.where(K != -9999)[0]
        fitlered_location = np.where(location_id > 1)[0]

        filtered_index = reduce(np.intersect1d,
                                (fitlered_starflag, fitlered_aspcapflag, fitlered_temp_lower, fitlered_vscatter,
                                 fitlered_Fe, fitlered_logg, fitlered_snrlow, fitlered_snrhigh, fitlered_location,
                                 fitlered_temp_upper, fitlered_K))

        print('Total Combined Spectra after filtering: ', filtered_index.shape[0])
        print('Total Individual Visit Spectra there: ', np.sum(hdulist[1].data['NVISITS'][filtered_index]))
        return filtered_index

    def apstar_normalization(self, spectra, spectra_err):
        return continuum(spectra=spectra, spectra_vars=spectra_err, cont_mask=self.cont_mask, deg=2, dr=self.apogee_dr)

    def compile(self):
        hdulist = self.load_allstar()
        indices = self.filter_apogeeid_list(hdulist)
        start_time = time.time()
        for counter, index in enumerate(indices):
            apogee_id = hdulist[1].data['APOGEE_ID'][index]
            location_id = hdulist[1].data['LOCATION_ID'][index]
            if counter % 100 == 0:
                print('Completed {} of {}, {:.03f} seconds elapsed'.format(counter, indices.shape[0],
                                                                           time.time() - start_time))
            warningflag, path = combined_spectra(dr=self.apogee_dr, location=location_id, apogee=apogee_id, verbose=0)
            if warningflag is None:
                combined_file = fits.open(path)
                _spec = combined_file[1].data  # Pseudo-continuum normalized flux
                _spec_err = combined_file[2].data  # Spectrum error array
                _spec = gap_delete(_spec, dr=self.apogee_dr)  # Delete the gap between sensors
                _spec_err = gap_delete(_spec_err, dr=self.apogee_dr)
                combined_file.close()

                warningflag, apstar_path = visit_spectra(dr=self.apogee_dr, location=location_id, apogee=apogee_id,
                                                         verbose=0)
                apstar_file = fits.open(apstar_path)
                nvisits = apstar_file[0].header['NVISITS']
                if nvisits == 1:
                    ap_spec = apstar_file[1].data
                    ap_err = apstar_file[2].data
                else:
                    ap_spec = apstar_file[1].data[1:]
                    ap_err = apstar_file[2].data[1:]
                ap_spec = gap_delete(ap_spec, dr=self.apogee_dr)
                ap_err = gap_delete(ap_err, dr=self.apogee_dr)
                cont_arr = self.apstar_normalization(ap_spec, ap_err)

                spec = []
                spec_err = []
                SNR = []
                RA = []
                DEC = []
                teff = []
                logg = []
                MH = []
                alpha_M = []
                C = []
                Cl = []
                N = []
                O = []
                Na = []
                Mg = []
                Al = []
                Si = []
                P = []
                S = []
                K = []
                Ca = []
                Ti = []
                Ti2 = []
                V = []
                Cr = []
                Mn = []
                Fe = []
                Ni = []
                Cu = []
                Ge = []
                Rb = []
                Y = []
                Nd = []
                absmag = []

                spec_continuum.extend([cont_arr])
                spec_continuum_err.extend([ap_err])
                apstar_file.close()

                spec.extend([_spec])
                spec_err.extend([_spec_err])
                SNR.extend([hdulist[1].data['SNR'][index]])
                RA.extend([hdulist[1].data['RA'][index]])
                DEC.extend([hdulist[1].data['DEC'][index]])
                teff.extend([hdulist[1].data['PARAM'][index, 0]])
                logg.extend([hdulist[1].data['PARAM'][index, 1]])
                MH.extend([hdulist[1].data['PARAM'][index, 3]])
                alpha_M.extend([hdulist[1].data['PARAM'][index, 6]])
                C.extend([hdulist[1].data['X_H'][index, 0]])
                Cl.extend([hdulist[1].data['X_H'][index, 1]])
                N.extend([hdulist[1].data['X_H'][index, 2]])
                O.extend([hdulist[1].data['X_H'][index, 3]])
                Na.extend([hdulist[1].data['X_H'][index, 4]])
                Mg.extend([hdulist[1].data['X_H'][index, 5]])
                Al.extend([hdulist[1].data['X_H'][index, 6]])
                Si.extend([hdulist[1].data['X_H'][index, 7]])
                P.extend([hdulist[1].data['X_H'][index, 8]])
                S.extend([hdulist[1].data['X_H'][index, 9]])
                K.extend([hdulist[1].data['X_H'][index, 10]])
                Ca.extend([hdulist[1].data['X_H'][index, 11]])
                Ti.extend([hdulist[1].data['X_H'][index, 12]])
                Ti2.extend([hdulist[1].data['X_H'][index, 13]])
                V.extend([hdulist[1].data['X_H'][index, 14]])
                Cr.extend([hdulist[1].data['X_H'][index, 15]])
                Mn.extend([hdulist[1].data['X_H'][index, 16]])
                Fe.extend([hdulist[1].data['X_H'][index, 17]])
                Ni.extend([hdulist[1].data['X_H'][index, 19]])
                Cu.extend([hdulist[1].data['X_H'][index, 20]])
                Ge.extend([hdulist[1].data['X_H'][index, 21]])
                Rb.extend([hdulist[1].data['X_H'][index, 22]])
                Y.extend([hdulist[1].data['X_H'][index, 23]])
                Nd.extend([hdulist[1].data['X_H'][index, 24]])
                absmag.extend([np.float32(-9999.)])


class H5Loader():
    def __init__(self, filename):
        self.h5name = filename

    def output(self):
        x, y = 0, 0
        return x, y
