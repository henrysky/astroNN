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
from astroNN.gaia.downloader import tgas_load, anderson_2017_parallax
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
        self.continuum = True  # True to do continuum normalization, False to use aspcap normalized spectra

    def load_allstar(self):
        self.apogee_dr = apogee_default_dr(dr=self.apogee_dr)
        allstarpath = astroNN.apogee.downloader.allstar(dr=self.apogee_dr)
        hdulist = fits.open(allstarpath)
        print('Loading allStar DR{} catalog'.format(self.apogee_dr))
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

        # Error arrays
        teff_err = np.array([])
        logg_err = np.array([])
        MH_err = np.array([])
        alpha_M_err = np.array([])
        C_err = np.array([])
        Cl_err = np.array([])
        N_err = np.array([])
        O_err = np.array([])
        Na_err = np.array([])
        Mg_err = np.array([])
        Al_err = np.array([])
        Si_err = np.array([])
        P_err = np.array([])
        S_err = np.array([])
        K_err = np.array([])
        Ca_err = np.array([])
        Ti_err = np.array([])
        Ti2_err = np.array([])
        V_err = np.array([])
        Cr_err = np.array([])
        Mn_err = np.array([])
        Fe_err = np.array([])
        Ni_err = np.array([])
        Cu_err = np.array([])
        Ge_err = np.array([])
        Rb_err = np.array([])
        Y_err = np.array([])
        Nd_err = np.array([])
        absmag_err = np.array([])

        for counter, index in enumerate(indices[1:100]):
            nvisits = 1
            apogee_id = hdulist[1].data['APOGEE_ID'][index]
            location_id = hdulist[1].data['LOCATION_ID'][index]
            if counter % 100 == 0:
                print('Completed {} of {}, {:.03f} seconds elapsed'.format(counter, indices.shape[0],
                                                                           time.time() - start_time))
            if self.continuum is False:
                warningflag, path = combined_spectra(dr=self.apogee_dr, location=location_id, apogee=apogee_id, verbose=0)
                if warningflag is None:
                    combined_file = fits.open(path)
                    _spec = combined_file[1].data  # Pseudo-continuum normalized flux
                    _spec_err = combined_file[2].data  # Spectrum error array
                    _spec = gap_delete(_spec, dr=self.apogee_dr)  # Delete the gap between sensors
                    _spec_err = gap_delete(_spec_err, dr=self.apogee_dr)
                    combined_file.close()
            else:
                warningflag, apstar_path = visit_spectra(dr=self.apogee_dr, location=location_id, apogee=apogee_id,
                                                         verbose=0)
                apstar_file = fits.open(apstar_path)
                nvisits = apstar_file[0].header['NVISITS']
                if nvisits == 1:
                    _spec = apstar_file[1].data
                    _spec_err = apstar_file[2].data
                else:
                    _spec = apstar_file[1].data[1:]
                    _spec_err = apstar_file[2].data[1:]
                    nvisits += 1
                _spec = gap_delete(_spec, dr=self.apogee_dr)
                _spec_err = gap_delete(_spec_err, dr=self.apogee_dr)
                _spec, _spec_err = self.apstar_normalization(_spec, _spec_err)
                apstar_file.close()

            if counter == 0:
                spec = np.array(_spec)
                spec_err = np.array(_spec_err)
                SNR = np.array([])
                RA = np.tile(hdulist[1].data['RA'][index], (nvisits, 1))
                DEC = np.array([])
                # Data array
                teff = np.array([])
                logg = np.array([])
                MH = np.array([])
                alpha_M = np.array([])
                C = np.array([])
                Cl = np.array([])
                N = np.array([])
                O = np.array([])
                Na = np.array([])
                Mg = np.array([])
                Al = np.array([])
                Si = np.array([])
                P = np.array([])
                S = np.array([])
                K = np.array([])
                Ca = np.array([])
                Ti = np.array([])
                Ti2 = np.array([])
                V = np.array([])
                Cr = np.array([])
                Mn = np.array([])
                Fe = np.array([])
                Ni = np.array([])
                Cu = np.array([])
                Ge = np.array([])
                Rb = np.array([])
                Y = np.array([])
                Nd = np.array([])
                absmag = np.array([])

            if warningflag is None:
                print(apogee_id)
                print(_spec)
                spec = np.vstack((spec, _spec))
                spec_err = np.vstack((spec_err, _spec_err))
                # SNR.extend([hdulist[1].data['SNR'][index]])
                RA = np.vstack((RA, np.tile(hdulist[1].data['RA'][index], (nvisits, 1))))
                # DEC.extend([hdulist[1].data['DEC'][index]])
                # teff.extend([hdulist[1].data['PARAM'][index, 0]])
                # logg.extend([hdulist[1].data['PARAM'][index, 1]])
                # MH.extend([hdulist[1].data['PARAM'][index, 3]])
                # alpha_M.extend([hdulist[1].data['PARAM'][index, 6]])
                # C.extend([hdulist[1].data['X_H'][index, 0]])
                # Cl.extend([hdulist[1].data['X_H'][index, 1]])
                # N.extend([hdulist[1].data['X_H'][index, 2]])
                # O.extend([hdulist[1].data['X_H'][index, 3]])
                # Na.extend([hdulist[1].data['X_H'][index, 4]])
                # Mg.extend([hdulist[1].data['X_H'][index, 5]])
                # Al.extend([hdulist[1].data['X_H'][index, 6]])
                # Si.extend([hdulist[1].data['X_H'][index, 7]])
                # P.extend([hdulist[1].data['X_H'][index, 8]])
                # S.extend([hdulist[1].data['X_H'][index, 9]])
                # K.extend([hdulist[1].data['X_H'][index, 10]])
                # Ca.extend([hdulist[1].data['X_H'][index, 11]])
                # Ti.extend([hdulist[1].data['X_H'][index, 12]])
                # Ti2.extend([hdulist[1].data['X_H'][index, 13]])
                # V.extend([hdulist[1].data['X_H'][index, 14]])
                # Cr.extend([hdulist[1].data['X_H'][index, 15]])
                # Mn.extend([hdulist[1].data['X_H'][index, 16]])
                # Fe.extend([hdulist[1].data['X_H'][index, 17]])
                # Ni.extend([hdulist[1].data['X_H'][index, 19]])
                # Cu.extend([hdulist[1].data['X_H'][index, 20]])
                # Ge.extend([hdulist[1].data['X_H'][index, 21]])
                # Rb.extend([hdulist[1].data['X_H'][index, 22]])
                # Y.extend([hdulist[1].data['X_H'][index, 23]])
                # Nd.extend([hdulist[1].data['X_H'][index, 24]])
                # absmag.extend([np.float32(-9999.)])
                #
                # teff_err.extend([hdulist[1].data['PARAM'][index, 0]])
                # logg_err.extend([hdulist[1].data['PARAM'][index, 1]])
                # MH.extend([hdulist[1].data['PARAM'][index, 3]])
                # alpha_M.extend([hdulist[1].data['PARAM'][index, 6]])
                # C.extend([hdulist[1].data['X_H'][index, 0]])
                # Cl.extend([hdulist[1].data['X_H'][index, 1]])
                # N.extend([hdulist[1].data['X_H'][index, 2]])
                # O.extend([hdulist[1].data['X_H'][index, 3]])
                # Na.extend([hdulist[1].data['X_H'][index, 4]])
                # Mg.extend([hdulist[1].data['X_H'][index, 5]])
                # Al.extend([hdulist[1].data['X_H'][index, 6]])
                # Si.extend([hdulist[1].data['X_H'][index, 7]])
                # P.extend([hdulist[1].data['X_H'][index, 8]])
                # S.extend([hdulist[1].data['X_H'][index, 9]])
                # K.extend([hdulist[1].data['X_H'][index, 10]])
                # Ca.extend([hdulist[1].data['X_H'][index, 11]])
                # Ti.extend([hdulist[1].data['X_H'][index, 12]])
                # Ti2.extend([hdulist[1].data['X_H'][index, 13]])
                # V.extend([hdulist[1].data['X_H'][index, 14]])
                # Cr.extend([hdulist[1].data['X_H'][index, 15]])
                # Mn.extend([hdulist[1].data['X_H'][index, 16]])
                # Fe.extend([hdulist[1].data['X_H'][index, 17]])
                # Ni.extend([hdulist[1].data['X_H'][index, 19]])
                # Cu.extend([hdulist[1].data['X_H'][index, 20]])
                # Ge.extend([hdulist[1].data['X_H'][index, 21]])
                # Rb.extend([hdulist[1].data['X_H'][index, 22]])
                # Y.extend([hdulist[1].data['X_H'][index, 23]])
                # Nd.extend([hdulist[1].data['X_H'][index, 24]])
                # absmag.extend([np.float32(-9999.)])

        print('Creating {}.h5'.format(self.h5_filename))
        h5f = h5py.File('{}.h5'.format(self.h5_filename), 'w')
        h5f.create_dataset('spectra', data=spec)
        h5f.create_dataset('spectra_err', data=spec_err)
        h5f.create_dataset('index', data=indices)
        # h5f.create_dataset('SNR', data=SNR)
        h5f.create_dataset('RA', data=RA)
        # h5f.create_dataset('DEC', data=DEC)
        # h5f.create_dataset('teff', data=teff)
        # h5f.create_dataset('logg', data=logg)
        # h5f.create_dataset('M', data=MH)
        # h5f.create_dataset('alpha', data=alpha_M)
        # h5f.create_dataset('C', data=C)
        # h5f.create_dataset('Cl', data=Cl)
        # h5f.create_dataset('N', data=N)
        # h5f.create_dataset('O', data=O)
        # h5f.create_dataset('Na', data=Na)
        # h5f.create_dataset('Mg', data=Mg)
        # h5f.create_dataset('Al', data=Al)
        # h5f.create_dataset('Si', data=Si)
        # h5f.create_dataset('P', data=P)
        # h5f.create_dataset('S', data=S)
        # h5f.create_dataset('K', data=K)
        # h5f.create_dataset('Ca', data=Ca)
        # h5f.create_dataset('Ti', data=Ti)
        # h5f.create_dataset('Ti2', data=Ti2)
        # h5f.create_dataset('V', data=V)
        # h5f.create_dataset('Cr', data=Cr)
        # h5f.create_dataset('Mn', data=Mn)
        # h5f.create_dataset('Fe', data=Fe)
        # h5f.create_dataset('Ni', data=Ni)
        # h5f.create_dataset('Cu', data=Cu)
        # h5f.create_dataset('Ge', data=Ge)
        # h5f.create_dataset('Rb', data=Rb)
        # h5f.create_dataset('Y', data=Y)
        # h5f.create_dataset('Nd', data=Nd)
        # h5f.create_dataset('absmag', data=absmag)
        h5f.close()
        print('Successfully created {}.h5 in {}'.format(self.h5_filename, currentdir))


class H5Loader():
    def __init__(self, filename):
        self.h5name = filename

    def output(self):
        x, y = 0, 0
        return x, y
