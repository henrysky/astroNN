# ---------------------------------------------------------#
#   astroNN.datasets.h5_compiler: compile h5 files for NN
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

    # def compile(self, continuum_mask=None):
    #     self.cont_mask = continuum_mask
    #
    #     self.apogee_dr = apogee_default_dr(dr=self.apogee_dr)
    #     h5name_check(self.h5_filename)
    #
    #     hdulist = self.load_allstar()
    #     indices = self.filter_apogeeid_list(hdulist)
    #
    #     spec = []
    #     spec_continuum = []
    #     spec_continuum_err = []
    #     spec_err = []
    #     SNR = []
    #     RA = []
    #     DEC = []
    #     teff = []
    #     logg = []
    #     MH = []
    #     alpha_M = []
    #     C = []
    #     Cl = []
    #     N = []
    #     O = []
    #     Na = []
    #     Mg = []
    #     Al = []
    #     Si = []
    #     P = []
    #     S = []
    #     K = []
    #     Ca = []
    #     Ti = []
    #     Ti2 = []
    #     V = []
    #     Cr = []
    #     Mn = []
    #     Fe = []
    #     Ni = []
    #     Cu = []
    #     Ge = []
    #     Rb = []
    #     Y = []
    #     Nd = []
    #     absmag = []
    #
    #     ra_gaia, dec_gaia, pmra_gaia, pmdec_gaia, parallax_gaia, parallax_error_gaia, g_band_gaia = \
    #         tgas_load(dr=self.gaia_dr)
    #
    #     index_1 = np.where(parallax_gaia > 0)
    #     index_2 = np.where((parallax_error_gaia / parallax_gaia) < 0.15)
    #     good_index = reduce(np.intersect1d, (index_1, index_2))
    #     ra_gaia = ra_gaia[good_index]
    #     dec_gaia = dec_gaia[good_index]
    #     pmra_gaia = pmra_gaia[good_index]
    #     pmdec_gaia = pmdec_gaia[good_index]
    #     parallax_gaia = parallax_gaia[good_index]
    #
    #     ra_apogee = (hdulist[1].data['RA'])[indices]
    #     dec_apogee = (hdulist[1].data['DEC'])[indices]
    #     k_mag_apogee = (hdulist[1].data['K'])[indices]
    #
    #     m1, m2, sep = astroNN.datasets.xmatch.xmatch(ra_apogee, ra_gaia, maxdist=2, colRA1=ra_apogee, colDec1=dec_apogee
    #                                                  , epoch1=2000., colRA2=ra_gaia, colDec2=dec_gaia, epoch2=2015.,
    #                                                  colpmRA2=pmra_gaia, colpmDec2=pmdec_gaia, swap=True)
    #
    #     absmag_temp = mag_to_absmag(k_mag_apogee[m1], (parallax_gaia[m2] / 1000))
    #
    #     start_time = time.time()
    #     for counter, index in enumerate(indices):
    #         apogee_id = hdulist[1].data['APOGEE_ID'][index]
    #         location_id = hdulist[1].data['LOCATION_ID'][index]
    #         if counter % 100 == 0:
    #             print('Completed {} of {}, {:.03f} seconds elapsed'.format(counter, indices.shape[0],
    #                                                                        time.time() - start_time))
    #         warningflag, path = combined_spectra(dr=self.apogee_dr, location=location_id, apogee=apogee_id, verbose=0)
    #         if warningflag is None:
    #             combined_file = fits.open(path)
    #             _spec = combined_file[1].data  # Pseudo-comtinumm normalized flux
    #             _spec_err = combined_file[2].data  # Spectrum error array
    #             _spec = gap_delete(_spec, dr=self.apogee_dr)  # Delete the gap between sensors
    #             _spec_err = gap_delete(_spec_err, dr=self.apogee_dr)
    #             combined_file.close()
    #
    #             warningflag, apstar_path = visit_spectra(dr=self.apogee_dr, location=location_id, apogee=apogee_id,
    #                                                      verbose=0)
    #             apstar_file = fits.open(apstar_path)
    #             nvisits = apstar_file[0].header['NVISITS']
    #             if nvisits == 1:
    #                 ap_spec = apstar_file[1].data
    #                 ap_err = apstar_file[2].data
    #             else:
    #                 ap_spec = apstar_file[1].data[1:]
    #                 ap_err = apstar_file[2].data[1:]
    #             ap_spec = gap_delete(ap_spec, dr=self.apogee_dr)
    #             ap_err = gap_delete(ap_err, dr=self.apogee_dr)
    #             cont_arr = self.apstar_normalization(ap_spec, ap_err)
    #
    #             spec_continuum.extend([cont_arr])
    #             spec_continuum_err.extend([ap_err])
    #             apstar_file.close()
    #
    #             spec.extend([_spec])
    #             spec_err.extend([_spec_err])
    #             SNR.extend([hdulist[1].data['SNR'][index]])
    #             RA.extend([hdulist[1].data['RA'][index]])
    #             DEC.extend([hdulist[1].data['DEC'][index]])
    #             teff.extend([hdulist[1].data['PARAM'][index, 0]])
    #             logg.extend([hdulist[1].data['PARAM'][index, 1]])
    #             MH.extend([hdulist[1].data['PARAM'][index, 3]])
    #             alpha_M.extend([hdulist[1].data['PARAM'][index, 6]])
    #             C.extend([hdulist[1].data['X_H'][index, 0]])
    #             Cl.extend([hdulist[1].data['X_H'][index, 1]])
    #             N.extend([hdulist[1].data['X_H'][index, 2]])
    #             O.extend([hdulist[1].data['X_H'][index, 3]])
    #             Na.extend([hdulist[1].data['X_H'][index, 4]])
    #             Mg.extend([hdulist[1].data['X_H'][index, 5]])
    #             Al.extend([hdulist[1].data['X_H'][index, 6]])
    #             Si.extend([hdulist[1].data['X_H'][index, 7]])
    #             P.extend([hdulist[1].data['X_H'][index, 8]])
    #             S.extend([hdulist[1].data['X_H'][index, 9]])
    #             K.extend([hdulist[1].data['X_H'][index, 10]])
    #             Ca.extend([hdulist[1].data['X_H'][index, 11]])
    #             Ti.extend([hdulist[1].data['X_H'][index, 12]])
    #             Ti2.extend([hdulist[1].data['X_H'][index, 13]])
    #             V.extend([hdulist[1].data['X_H'][index, 14]])
    #             Cr.extend([hdulist[1].data['X_H'][index, 15]])
    #             Mn.extend([hdulist[1].data['X_H'][index, 16]])
    #             Fe.extend([hdulist[1].data['X_H'][index, 17]])
    #             Ni.extend([hdulist[1].data['X_H'][index, 19]])
    #             Cu.extend([hdulist[1].data['X_H'][index, 20]])
    #             Ge.extend([hdulist[1].data['X_H'][index, 21]])
    #             Rb.extend([hdulist[1].data['X_H'][index, 22]])
    #             Y.extend([hdulist[1].data['X_H'][index, 23]])
    #             Nd.extend([hdulist[1].data['X_H'][index, 24]])
    #             absmag.extend([np.float32(-9999.)])
    #
    #     absmag = np.array(absmag)
    #     absmag[m1] = absmag_temp
    #
    #     print('Creating {}.h5'.format(self.h5_filename))
    #     h5f = h5py.File('{}.h5'.format(self.h5_filename), 'w')
    #     h5f.create_dataset('spectra', data=spec)
    #     h5f.create_dataset('spec_continuum', data=spec_continuum)
    #     h5f.create_dataset('spectra_err', data=spec_err)
    #     h5f.create_dataset('index', data=indices)
    #     h5f.create_dataset('SNR', data=SNR)
    #     h5f.create_dataset('RA', data=RA)
    #     h5f.create_dataset('DEC', data=DEC)
    #     h5f.create_dataset('teff', data=teff)
    #     h5f.create_dataset('logg', data=logg)
    #     h5f.create_dataset('M', data=MH)
    #     h5f.create_dataset('alpha', data=alpha_M)
    #     h5f.create_dataset('C', data=C)
    #     h5f.create_dataset('Cl', data=Cl)
    #     h5f.create_dataset('N', data=N)
    #     h5f.create_dataset('O', data=O)
    #     h5f.create_dataset('Na', data=Na)
    #     h5f.create_dataset('Mg', data=Mg)
    #     h5f.create_dataset('Al', data=Al)
    #     h5f.create_dataset('Si', data=Si)
    #     h5f.create_dataset('P', data=P)
    #     h5f.create_dataset('S', data=S)
    #     h5f.create_dataset('K', data=K)
    #     h5f.create_dataset('Ca', data=Ca)
    #     h5f.create_dataset('Ti', data=Ti)
    #     h5f.create_dataset('Ti2', data=Ti2)
    #     h5f.create_dataset('V', data=V)
    #     h5f.create_dataset('Cr', data=Cr)
    #     h5f.create_dataset('Mn', data=Mn)
    #     h5f.create_dataset('Fe', data=Fe)
    #     h5f.create_dataset('Ni', data=Ni)
    #     h5f.create_dataset('Cu', data=Cu)
    #     h5f.create_dataset('Ge', data=Ge)
    #     h5f.create_dataset('Rb', data=Rb)
    #     h5f.create_dataset('Y', data=Y)
    #     h5f.create_dataset('Nd', data=Nd)
    #     h5f.create_dataset('absmag', data=absmag)
    #     h5f.close()
    #     print('Successfully created {}.h5 in {}'.format(self.h5_filename, currentdir))

#
# def compile_gaia(h5name=None, gaia_dr=None, apogee_dr=None, SNR_low=100, vscattercut=1):
#     """
#     NAME: compile_gaia
#     PURPOSE: compile gaia data to a h5 file
#     INPUT:
#         gaia_dr= 1
#         apogee_dr=14
#     OUTPUT: (just operations)
#     HISTORY:
#         2017-Nov-08 Henry Leung
#     """
#     h5name_check(h5name)
#     apogee_dr = apogee_default_dr(dr=apogee_dr)
#     gaia_dr = gaia_default_dr(dr=gaia_dr)
#
#     allstarpath = astroNN.apogee.downloader.allstar(dr=apogee_dr)
#     tgas_list = astroNN.gaia.downloader.tgas(dr=gaia_dr)
#
#     tefflow = 4000
#     teffhigh = 5500
#
#     hdulist = fits.open(allstarpath)
#     SNR = hdulist[1].data['SNR']
#     starflag = hdulist[1].data['STARFLAG']
#     aspcapflag = hdulist[1].data['ASPCAPFLAG']
#     vscatter = hdulist[1].data['VSCATTER']
#     location_id = hdulist[1].data['LOCATION_ID']
#     teff = hdulist[1].data['PARAM'][:, 0]
#
#     DR_fitlered_SNR_low = np.where(SNR > SNR_low)[0]
#     DR_fitlered_starflag = np.where(starflag == 0)[0]
#     DR_fitlered_aspcapflag = np.where(aspcapflag == 0)[0]
#     DR_fitlered_vscatter = np.where(vscatter < vscattercut)[0]
#     DR_fitlered_temp_lower = np.where((tefflow <= teff))[0]
#     DR_fitlered_temp_upper = np.where((teffhigh >= teff))[0]
#     # There are some location_id=1 to avoid
#     DR14_fitlered_location = np.where(location_id > 1)[0]
#
#     # Here we found the common indices that satisfied all requirement
#     filtered_apogee_index = reduce(np.intersect1d, (DR14_fitlered_location, DR_fitlered_SNR_low, DR_fitlered_starflag,
#                                                     DR_fitlered_aspcapflag, DR_fitlered_vscatter,
#                                                     DR_fitlered_temp_lower,
#                                                     DR_fitlered_temp_upper))
#
#     ra_apogee = (hdulist[1].data['RA'])[filtered_apogee_index]
#     dec_apogee = (hdulist[1].data['DEC'])[filtered_apogee_index]
#     k_mag_apogee = (hdulist[1].data['K'])[filtered_apogee_index]
#     teff = (hdulist[1].data['PARAM'][:, 0])[filtered_apogee_index]
#
#     ra_gaia = np.array([])
#     dec_gaia = np.array([])
#     pmra_gaia = np.array([])
#     pmdec_gaia = np.array([])
#     parallax_gaia = np.array([])
#     parallax_error_gaia = np.array([])
#     mag_gaia = np.array([])
#
#     for i in tgas_list:
#         gaia = fits.open(i)
#         ra_gaia = np.concatenate((ra_gaia, gaia[1].data['RA']))
#         dec_gaia = np.concatenate((dec_gaia, gaia[1].data['DEC']))
#         pmra_gaia = np.concatenate((pmra_gaia, gaia[1].data['PMRA']))
#         pmdec_gaia = np.concatenate((pmdec_gaia, gaia[1].data['PMDEC']))
#         parallax_gaia = np.concatenate((parallax_gaia, gaia[1].data['parallax']))
#         parallax_error_gaia = np.concatenate((parallax_error_gaia, gaia[1].data['parallax_error']))
#         # mag_gaia = np.concatenate((mag_gaia, gaia[1].data['phot_g_mean_mag']))
#
#     bad_index = np.where(parallax_gaia <= 0)
#     ra_gaia = np.delete(ra_gaia, bad_index)
#     dec_gaia = np.delete(dec_gaia, bad_index)
#     pmra_gaia = np.delete(pmra_gaia, bad_index)
#     pmdec_gaia = np.delete(pmdec_gaia, bad_index)
#     parallax_gaia = np.delete(parallax_gaia, bad_index)
#     parallax_error_gaia = np.delete(parallax_error_gaia, bad_index)
#     # mag_gaia = np.delete(mag_gaia, bad_index)
#
#     bad_index = np.where((parallax_error_gaia / parallax_gaia) > 0.2)
#     ra_gaia = np.delete(ra_gaia, bad_index)
#     dec_gaia = np.delete(dec_gaia, bad_index)
#     pmra_gaia = np.delete(pmra_gaia, bad_index)
#     pmdec_gaia = np.delete(pmdec_gaia, bad_index)
#     parallax_gaia = np.delete(parallax_gaia, bad_index)
#     parallax_error_gaia = np.delete(parallax_error_gaia, bad_index)
#     # mag_gaia = np.delete(mag_gaia, bad_index)
#     # absmag = to_absmag(mag_gaia, parallax_gaia)
#
#     m1, m2, sep = astroNN.datasets.xmatch.xmatch(ra_apogee, ra_gaia, maxdist=2, colRA1=ra_apogee, colDec1=dec_apogee,
#                                                  epoch1=2000.,
#                                                  colRA2=ra_gaia, colDec2=dec_gaia, epoch2=2015., colpmRA2=pmra_gaia,
#                                                  colpmDec2=pmdec_gaia, swap=True)
#
#     # m2 fot Gaia, m1 for APOGEE
#
#     print('Total Numer of matches: ', len(m1))
#
#     parallax_gaia_percent = (parallax_error_gaia / parallax_gaia)
#
#     # train_len = int(len(m2)*0.6)
#
#     for tt in ['train', 'test']:
#         spec = []
#         if tt == 'train':
#             filtered_index = np.where(parallax_gaia_percent[m2] >= 0.1)
#             m1_1 = m1[filtered_index]
#             m2_2 = m2[filtered_index]
#         else:
#             filtered_index = np.where(parallax_gaia_percent[m2] < 0.1)
#             m1_1 = m1[filtered_index]
#             m2_2 = m2[filtered_index]
#
#         absmag = mag_to_absmag(k_mag_apogee[m1_1], parallax_gaia[m2_2] / 1000)
#         for index in m1_1:
#             apogee_id = ((hdulist[1].data['APOGEE_ID'])[filtered_apogee_index])[index]
#             location_id = ((hdulist[1].data['LOCATION_ID'])[filtered_apogee_index])[index]
#
#             warningflag = None
#             warningflag, path = combined_spectra(dr=apogee_dr, location=location_id, apogee=apogee_id, verbose=0)
#             if warningflag is None:
#                 combined_file = fits.open(path)
#                 _spec = combined_file[1].data  # Pseudo-comtinumm normalized flux
#                 _spec = gap_delete(_spec, dr=apogee_dr)
#                 spec.extend([_spec])
#
#         print('Creating {}_{}.h5'.format(h5name, tt))
#         h5f = h5py.File('{}_{}.h5'.format(h5name, tt), 'w')
#         h5f.create_dataset('spectra', data=spec)
#         h5f.create_dataset('teff', data=teff[m1_1])
#         h5f.create_dataset('absmag', data=absmag)
#         h5f.create_dataset('RA', data=ra_apogee[m1_1])
#         h5f.create_dataset('DEC', data=dec_apogee[m1_1])
#
#     return None
