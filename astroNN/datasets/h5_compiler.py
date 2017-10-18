# ---------------------------------------------------------#
#   astroNN.datasets.h5_compiler: compile h5 files for NN
# ---------------------------------------------------------#

import os
from astropy.io import fits
import numpy as np
import h5py
from functools import reduce
import astroNN.apogeetools.downloader

currentdir = os.getcwd()


def compile_apogee_training(h5name=None, dr=None, starflagcut=True, aspcapflagcut=True, vscattercut=1, SNRlow=200,
                            SNRhigh=9999, tefflow=4000, teffhigh=5500, ironlow=-3):
    """
    NAME: compile_apogee_training
    PURPOSE: compile apogee data to a training dataset
    INPUT:
        dr= 13 or 14
        starflagcut = True (Cut star with starflag != 0), False (do nothing)
        aspcapflagcut = True (Cut star with aspcapflag != 0), False (do nothing)
        vscattercut = scalar for maximum scattercut
        SNRlow/SNRhigh = SNR lower cut and SNR upper cut
        tefflow/teffhigh = Teff lower cut and Teff upper cut
        ironlow = lower limit of Fe/H dex

    OUTPUT: (just data compilation)
    HISTORY:
        2017-Oct-15 Henry Leung
    """
    if h5name is None:
        raise ValueError('Please specift the dataset name using h5name="..."')

    if dr is None:
        dr = 14
        print('dr is not provided, using default dr=14')

    if dr == 13:
        allstarepath = os.path.join(currentdir, 'apogee_dr13\\allStar-l30e.2.fits')
        # Check if directory exists
        if not os.path.exists(allstarepath):
            os.makedirs(allstarepath)
            print('allStar catalog DR13 not found, now using astroNN.apogeetools.downloader.all_star(dr=13) to download it')
            astroNN.apogeetools.downloader.allstar(dr=13)
            print('Checking if you have downloaded DR13 combined spectra now')
            astroNN.apogeetools.downloader.combined_spectra(dr=13)
        else:
            print('allStar catalog DR13 has found successfully, now loading it')
    elif dr == 14:
        allstarepath = os.path.join(currentdir, 'apogee_dr14\\allStar-l31c.2.fits')
        # Check if directory exists
        if not os.path.exists(allstarepath):
            os.makedirs(allstarepath)
            print('allStar catalog DR13 not found, now using astroNN.apogeetools.downloader.all_star(dr=14) to download it')
            astroNN.apogeetools.downloader.allstar(dr=14)
            print('Checking if you have downloaded DR14 combined spectra now')
            astroNN.apogeetools.downloader.combined_spectra(dr=14)
        else:
            print('allStar catalog DR14 has found successfully, now loading it')
    else:
        raise ValueError('astroNN only supports DR13 and DR14 APOGEE')

    # Loading Data form FITS files
    hdulist = fits.open(allstarepath)
    print('Now processing allStar {} catalog'.format(dr))
    starflag = hdulist[1].data['STARFLAG']
    aspcapflag = hdulist[1].data['ASPCAPFLAG']
    vscatter = hdulist[1].data['VSCATTER']
    SNR = hdulist[1].data['SNR']
    location_id = hdulist[1].data['LOCATION_ID']
    apogee_id = hdulist[1].data['APOGEE_ID']
    RA = hdulist[1].data['RA']
    DEC = hdulist[1].data['DEC']
    teff = hdulist[1].data['PARAM'][:, 0]
    logg = hdulist[1].data['PARAM'][:, 1]
    MH = hdulist[1].data['PARAM'][:, 3]
    alpha_M = hdulist[1].data['PARAM'][:, 6]
    C = hdulist[1].data['X_H'][:, 0]
    Cl = hdulist[1].data['X_H'][:, 1]
    N = hdulist[1].data['X_H'][:, 2]
    O = hdulist[1].data['X_H'][:, 3]
    Na = hdulist[1].data['X_H'][:, 4]
    Mg = hdulist[1].data['X_H'][:, 5]
    Al = hdulist[1].data['X_H'][:, 6]
    Si = hdulist[1].data['X_H'][:, 7]
    Ca = hdulist[1].data['X_H'][:, 11]
    Ti = hdulist[1].data['X_H'][:, 12]
    Ti2 = hdulist[1].data['X_H'][:, 13]
    Fe = hdulist[1].data['X_H'][:, 17]
    Ni = hdulist[1].data['X_H'][:, 19]

    total = range(len(starflag))

    if starflagcut is True:
        DR_fitlered_starflag = np.where(starflag == 0)[0]
    else:
        DR_fitlered_starflag = total

    if aspcapflagcut is True:
        DR_fitlered_aspcapflag = np.where(aspcapflag == 0)[0]
    else:
        DR_fitlered_aspcapflag = total
    DR_fitlered_temp_lower = np.where((tefflow <= teff))[0]
    DR_fitlered_temp_upper = np.where((teffhigh >= teff))[0]
    DR_fitlered_vscatter = np.where(vscatter < vscattercut)[0]
    DR_fitlered_Fe = np.where(Fe > ironlow)[0]
    DR_fitlered_logg = np.where(logg != -9999)[0]
    DR_fitlered_snrlow = np.where(SNR > SNRlow)[0]
    DR_fitlered_snrhigh = np.where(SNR < SNRhigh)[0]

    # There are some location_id=1 to avoid
    DR14_fitlered_location = np.where(location_id > 1)[0]

    # Here we found the common indices that satisfied all requirement
    filtered_index = reduce(np.intersect1d,
                            (DR_fitlered_starflag, DR_fitlered_aspcapflag, DR_fitlered_temp_lower,
                             DR_fitlered_vscatter, DR_fitlered_Fe, DR_fitlered_logg, DR_fitlered_snrlow,
                             DR_fitlered_snrhigh, DR14_fitlered_location, DR_fitlered_temp_upper,
                             DR_fitlered_temp_lower))
    print('Total entry after filtering: ', filtered_index.shape)
    print('Total Visit there: ', np.sum(hdulist[1].data['NVISITS'][filtered_index]))

    spec = []
    temp = []
    Fe = []
    logg = []
    C = []
    Cl = []
    N = []
    alpha_M = []
    Ap_ID = []
    RA = []
    DEC = []
    SNR = []
    O = []
    Na = []
    Mg = []
    Ca = []

    print('Filtering the data according to the cuts you specified or detfault cuts')
    for index in filtered_index:
        filename = hdulist[1].data['APOGEE_ID'][index]
        filename = 'aspcapStar-r8-l31c.2-{}.fits'.format(filename)
        combined_file = fits.open(os.path.join(currentdir, 'apogee_dr14\\', filename))
        _spec = combined_file[3].data   # combined_file[3].data is the ASPCAP best fit spectrum
        _spec = np.delete(_spec, np.where(_spec == 0)) # Delete the gap between sensors

        Ap_ID.extend([filename])
        spec.extend([_spec])
        SNR.extend([hdulist[1].data['SNR'][index]])
        RA.extend([hdulist[1].data['RA'][index]])
        DEC.extend([hdulist[1].data['DEC'][index]])
        temp.extend([hdulist[1].data['PARAM'][index, 0]])
        Fe.extend([hdulist[1].data['X_H'][index, 17]])
        logg.extend([hdulist[1].data['PARAM'][index, 1]])
        C.extend([hdulist[1].data['X_H'][index, 0]])
        Cl.extend([hdulist[1].data['X_H'][index, 1]])
        N.extend([hdulist[1].data['X_H'][index, 2]])
        O.extend([hdulist[1].data['X_H'][index, 3]])
        alpha_M.extend([hdulist[1].data['PARAM'][index, 6]])
        Na.extend([hdulist[1].data['X_H'][index, 4]])
        Mg.extend([hdulist[1].data['X_H'][index, 5]])
        Ca.extend([hdulist[1].data['X_H'][index, 11]])

    print('Creating {}.h5'.format(h5name))
    h5f = h5py.File('{}.h5'.format(h5name), 'w')
    h5f.create_dataset('spectra', data=spec)
    h5f.create_dataset('RA', data=RA)
    h5f.create_dataset('DEC', data=DEC)
    h5f.create_dataset('temp', data=temp)
    h5f.create_dataset('SNR', data=SNR)
    h5f.create_dataset('Fe', data=Fe)
    h5f.create_dataset('logg', data=logg)
    h5f.create_dataset('C', data=C)
    h5f.create_dataset('N', data=N)
    h5f.create_dataset('alpha_M', data=alpha_M)

    h5f.close()
    print('Successfully created {}.h5 in {}'.format(h5name, currentdir))

    return None


def compile_gaia(h5name=None, dr=None):
    """
    NAME: compile_gaia
    PURPOSE: compile gaia data to a h5 file
    INPUT:
        dr= 13 or 14
        starflagcut = True (Cut star with starflag != 0), False (do nothing)
        aspcapflagcut = True (Cut star with aspcapflag != 0), False (do nothing)
        vscattercut = scalar for maximum scattercut
        SNRlow/SNRhigh = SNR lower cut and SNR upper cut
        tefflow/teffhigh = Teff lower cut and Teff upper cut
        ironlow = lower limit of Fe/H dex

    OUTPUT: (just operations)
    HISTORY:
        2017-Oct-15 Henry Leung
    """
    return None