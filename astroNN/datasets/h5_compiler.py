# ---------------------------------------------------------#
#   astroNN.datasets.h5_compiler: compile h5 files for NN
# ---------------------------------------------------------#

import os
from functools import reduce

import h5py
import numpy as np
from astropy.io import fits

import astroNN.apogeetools.downloader
import astroNN.gaiatools.downloader
import astroNN.datasets.xmatch

currentdir = os.getcwd()
_APOGEE_DATA = os.getenv('SDSS_LOCAL_SAS_MIRROR')
_GAIA_DATA = os.getenv('GAIA_TOOLS_DATA')


def apogeeid_digit(arr):
    """
    NAME: apogeeid_digit
    PURPOSE: Extract digits from apogeeid because its too painful to deal with APOGEE ID in h5py
    INPUT:
        arr = apogee_id
    OUTPUT: apogee_id with digits only
    HISTORY:
        2017-Oct-26 Henry Leung
    """
    return str(''.join(filter(str.isdigit, arr)))


def gap_delete(single_spec, dr=14):
    """
    NAME: gap_delete
    PURPOSE: delete the gap between APOGEE camera
    INPUT:
        single_spec = single spectra array
        dr = 13 or 14
    OUTPUT: corrected array
    HISTORY:
        2017-Oct-26 Henry Leung
    """
    if dr == 14:
        arr1 = np.arange(0, 246, 1)
        arr2 = np.arange(3274, 3585, 1)
        arr3 = np.arange(6080, 6344, 1)
        arr4 = np.arange(8335, 8575, 1)
        single_spec = np.delete(single_spec, arr4)
        single_spec = np.delete(single_spec, arr3)
        single_spec = np.delete(single_spec, arr2)
        single_spec = np.delete(single_spec, arr1)
        return single_spec
    else:
        raise ValueError('DR13 not supported')


def compile_apogee(h5name=None, dr=None, starflagcut=True, aspcapflagcut=True, vscattercut=1, SNRtrain_low=200,
                   SNRtrain_high=99999, tefflow=4000, teffhigh=5500, ironlow=-3, SNRtest_low=100, SNRtest_high=200):
    """
    NAME: compile_apogee
    PURPOSE: compile apogee data to a training and testing dataset
    INPUT:
        h5name = name of h5 dataset you want to create
        dr = 13 or 14
        starflagcut = True (Cut star with starflag != 0), False (do nothing)
        aspcapflagcut = True (Cut star with aspcapflag != 0), False (do nothing)
        vscattercut = scalar for maximum scattercut
        SNRlow/SNRhigh = SNR lower cut and SNR upper cut
        tefflow/teffhigh = Teff lower cut and Teff upper cut for training set
        ironlow = lower limit of Fe/H dex
        SNRtest_low/SNRtest_high = SNR lower cut and SNR upper cut for testing set

    OUTPUT: {h5name}_train.h5   {h5name}_test.h5
    HISTORY:
        2017-Oct-15 Henry Leung
    """
    if h5name is None:
        raise ValueError('Please specift the dataset name using h5name="..."')

    if dr is None:
        dr = 14
        print('dr is not provided, using default dr=14')

    if dr == 13:
        allstarepath = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/allStar-l30e.2.fits')
        # Check if directory exists
        if not os.path.exists(allstarepath):
            print(
                'allStar catalog DR13 not found, now using astroNN.apogeetools.downloader.allstar(dr=13) to download it')
            astroNN.apogeetools.downloader.allstar(dr=13)
        else:
            print('allStar catalog DR13 has found successfully, now loading it')
    elif dr == 14:
        allstarepath = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/allStar-l31c.2.fits')
        # Check if directory exists
        if not os.path.exists(allstarepath):
            print(
                'allStar catalog DR14 not found, now using astroNN.apogeetools.downloader.allstar(dr=14) to download it')
            astroNN.apogeetools.downloader.allstar(dr=14)
        else:
            print('allStar catalog DR14 has found successfully, now loading it')
    else:
        raise ValueError('astroNN only supports DR13 and DR14 APOGEE')

    # Loading Data form FITS files
    hdulist = fits.open(allstarepath)
    print('Now processing allStar DR{} catalog'.format(dr))
    starflag = hdulist[1].data['STARFLAG']
    aspcapflag = hdulist[1].data['ASPCAPFLAG']
    vscatter = hdulist[1].data['VSCATTER']
    SNR = hdulist[1].data['SNR']
    location_id = hdulist[1].data['LOCATION_ID']
    teff = hdulist[1].data['PARAM'][:, 0]
    logg = hdulist[1].data['PARAM'][:, 1]
    Fe = hdulist[1].data['X_H'][:, 17]

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
    DR_fitlered_snrlow = np.where(SNR > SNRtrain_low)[0]
    DR_fitlered_snrhigh = np.where(SNR < SNRtrain_high)[0]
    DR_fitlered_SNRtest_low = np.where(SNR > SNRtest_low)[0]
    DR_fitlered_SNRtest_high = np.where(SNR < SNRtest_high)[0]

    # There are some location_id=1 to avoid
    DR14_fitlered_location = np.where(location_id > 1)[0]

    # Here we found the common indices that satisfied all requirement
    filtered_train_index = reduce(np.intersect1d, (DR_fitlered_starflag, DR_fitlered_aspcapflag, DR_fitlered_temp_lower,
                                                   DR_fitlered_vscatter, DR_fitlered_Fe, DR_fitlered_logg,
                                                   DR_fitlered_snrlow,
                                                   DR_fitlered_snrhigh, DR14_fitlered_location, DR_fitlered_temp_upper))

    filtered_test_index = reduce(np.intersect1d, (DR_fitlered_starflag, DR_fitlered_aspcapflag, DR_fitlered_temp_lower,
                                                  DR_fitlered_vscatter, DR_fitlered_Fe, DR_fitlered_logg,
                                                  DR_fitlered_SNRtest_low,
                                                  DR_fitlered_SNRtest_high, DR14_fitlered_location,
                                                  DR_fitlered_temp_upper))

    print('Total entry after filtering: ', filtered_train_index.shape[0])
    print('Total Visit there: ', np.sum(hdulist[1].data['NVISITS'][filtered_train_index]))

    for tt in ['train', 'test']:
        spec = []
        spec_bestfit = []
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

        if tt == 'train':
            filtered_index = filtered_train_index
        else:
            filtered_index = filtered_test_index

        print('Filtering the dataset according to the cuts you specified or default cuts for the {}ing dataset'.format(
            tt))

        for index in filtered_index:
            apogee_id = hdulist[1].data['APOGEE_ID'][index]
            location_id = hdulist[1].data['LOCATION_ID'][index]
            warningflag = None
            if dr == 13:
                filename = 'aspcapStar-r6-l30e.2-{}.fits'.format(apogee_id)
                path = os.path.join(_APOGEE_DATA, 'dr13/apogee/spectro/redux/r6/stars/l30e/l30e.2/', str(location_id),
                                    filename)
                if not os.path.exists(path):
                    warningflag = astroNN.apogeetools.downloader.combined_spectra(dr=dr, location=location_id,
                                                                                  apogee=apogee_id)
            elif dr == 14:
                filename = 'aspcapStar-r8-l31c.2-{}.fits'.format(apogee_id)
                path = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/', str(location_id),
                                    filename)
                if not os.path.exists(path):
                    warningflag = astroNN.apogeetools.downloader.combined_spectra(dr=dr, location=location_id,
                                                                                  apogee=apogee_id)
            else:
                raise ValueError('astroNN only supports DR13 and DR14 APOGEE')
            if warningflag is None:
                combined_file = fits.open(path)
                _spec = combined_file[1].data  # Pseudo-comtinumm normalized flux
                _spec_bestfit = combined_file[3].data  # Best fit spectrum for training generative model
                _spec = gap_delete(_spec, dr=14)  # Delete the gap between sensors
                _spec_bestfit = gap_delete(_spec_bestfit, dr=14)  # Delete the gap between sensors

                spec.extend([_spec])
                spec_bestfit.extend([_spec_bestfit])
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

        print('Creating {}_{}.h5'.format(h5name, tt))
        h5f = h5py.File('{}_{}.h5'.format(h5name, tt), 'w')
        h5f.create_dataset('spectra', data=spec)
        h5f.create_dataset('spectrabestfit', data=spec_bestfit)
        h5f.create_dataset('index', data=filtered_index)
        h5f.create_dataset('SNR', data=SNR)
        h5f.create_dataset('RA', data=RA)
        h5f.create_dataset('DEC', data=DEC)
        h5f.create_dataset('teff', data=teff)
        h5f.create_dataset('logg', data=logg)
        h5f.create_dataset('M', data=MH)
        h5f.create_dataset('alpha', data=alpha_M)
        h5f.create_dataset('C', data=C)
        h5f.create_dataset('Cl', data=Cl)
        h5f.create_dataset('N', data=N)
        h5f.create_dataset('O', data=O)
        h5f.create_dataset('Na', data=Na)
        h5f.create_dataset('Mg', data=Mg)
        h5f.create_dataset('Al', data=Al)
        h5f.create_dataset('Si', data=Si)
        h5f.create_dataset('P', data=P)
        h5f.create_dataset('S', data=S)
        h5f.create_dataset('K', data=K)
        h5f.create_dataset('Ca', data=Ca)
        h5f.create_dataset('Ti', data=Ti)
        h5f.create_dataset('Ti2', data=Ti2)
        h5f.create_dataset('V', data=V)
        h5f.create_dataset('Cr', data=Cr)
        h5f.create_dataset('Mn', data=Mn)
        h5f.create_dataset('Fe', data=Fe)
        h5f.create_dataset('Ni', data=Ni)
        h5f.create_dataset('Cu', data=Cu)
        h5f.create_dataset('Ge', data=Ge)
        h5f.create_dataset('Rb', data=Rb)
        h5f.create_dataset('Y', data=Y)
        h5f.create_dataset('Nd', data=Nd)
        h5f.close()
        print('Successfully created {}_{}.h5 in {}'.format(h5name, tt, currentdir))

    return None


def compile_gaia(h5name=None, gaia_dr=None, apogee_dr=None, existh5=None, SNR_low=100, vscattercut=1):
    """
    NAME: compile_gaia
    PURPOSE: compile gaia data to a h5 file
    INPUT:
        gaia_dr= 1
        apogee_dr=14
    OUTPUT: (just operations)
    HISTORY:
        2017-Nov-08 Henry Leung
    """
    if h5name is None:
        raise ValueError('Please specift the dataset name using h5name="..."')
    if gaia_dr is None:
        gaia_dr = 1
        print('gaia_dr is not provided, using default gaia_dr=1')
    if apogee_dr is None and existh5 is not None:
        apogee_dr = 14
        print('apogee_dr is not provided, using default apogee_dr=14')

    if apogee_dr == 14 and existh5 is None:
        allstarepath = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/allStar-l31c.2.fits')
        # Check if directory exists
        if not os.path.exists(allstarepath):
            print(
                'allStar catalog DR14 not found, now using astroNN.apogeetools.downloader.allstar(dr=14) to download it')
            astroNN.apogeetools.downloader.allstar(dr=14)
        else:
            print('allStar catalog DR14 has found successfully, now loading it')
    else:
        raise ValueError('Only apogee dr14 is supported')

    if gaia_dr == 1:
        astroNN.gaiatools.downloader.tgas(dr=1)
    else:
        raise ValueError('Only gaia dr1 is supported')

    if existh5 is not None:
        data = existh5 + '_train.h5'

    tefflow = 4000
    teffhigh = 5500

    hdulist = fits.open(allstarepath)
    SNR = hdulist[1].data['SNR']
    starflag = hdulist[1].data['STARFLAG']
    aspcapflag = hdulist[1].data['ASPCAPFLAG']
    vscatter = hdulist[1].data['VSCATTER']
    location_id = hdulist[1].data['LOCATION_ID']
    teff = hdulist[1].data['PARAM'][:, 0]

    DR_fitlered_SNR_low = np.where(SNR > SNR_low)[0]
    DR_fitlered_starflag = np.where(starflag == 0)[0]
    DR_fitlered_aspcapflag = np.where(aspcapflag == 0)[0]
    DR_fitlered_vscatter = np.where(vscatter < vscattercut)[0]
    DR_fitlered_temp_lower = np.where((tefflow <= teff))[0]
    DR_fitlered_temp_upper = np.where((teffhigh >= teff))[0]
    # There are some location_id=1 to avoid
    DR14_fitlered_location = np.where(location_id > 1)[0]

    # Here we found the common indices that satisfied all requirement
    filtered_apogee_index = reduce(np.intersect1d, (DR14_fitlered_location, DR_fitlered_SNR_low, DR_fitlered_starflag,
                                                    DR_fitlered_aspcapflag))

    ra_apogee = (hdulist[1].data['RA'])[filtered_apogee_index]
    dec_apogee = (hdulist[1].data['DEC'])[filtered_apogee_index]

    ra_gaia = np.array([])
    dec_gaia = np.array([])
    pmra_gaia = np.array([])
    pmdec_gaia = np.array([])
    parallax_gaia = np.array([])
    parallax_error_gaia = np.array([])
    mag_gaia = np.array([])

    folderpath = os.path.join(_GAIA_DATA, 'Gaia/tgas_source/fits/')
    for i in range(0, 16, 1):
        filename = 'TgasSource_000-000-0{:02d}.fits'.format(i)
        fullfilename = os.path.join(folderpath, filename)
        gaia = fits.open(fullfilename)
        ra_gaia = np.concatenate((ra_gaia, gaia[1].data['RA']))
        dec_gaia = np.concatenate((dec_gaia, gaia[1].data['DEC']))
        pmra_gaia = np.concatenate((pmra_gaia, gaia[1].data['PMRA']))
        pmdec_gaia = np.concatenate((pmdec_gaia, gaia[1].data['PMDEC']))
        parallax_gaia = np.concatenate((parallax_gaia, gaia[1].data['parallax']))
        parallax_error_gaia = np.concatenate((parallax_error_gaia, gaia[1].data['parallax_error']))
        mag_gaia = np.concatenate((mag_gaia, gaia[1].data['phot_g_mean_mag']))

    bad_index = np.where(parallax_gaia <= 0)
    ra_gaia = np.delete(ra_gaia, bad_index)
    dec_gaia = np.delete(dec_gaia, bad_index)
    pmra_gaia = np.delete(pmra_gaia, bad_index)
    pmdec_gaia = np.delete(pmdec_gaia, bad_index)
    parallax_gaia = np.delete(parallax_gaia, bad_index)
    parallax_error_gaia = np.delete(parallax_error_gaia, bad_index)
    mag_gaia = np.delete(mag_gaia, bad_index)

    bad_index = np.where((parallax_error_gaia / parallax_gaia) > 0.2)
    ra_gaia = np.delete(ra_gaia, bad_index)
    dec_gaia = np.delete(dec_gaia, bad_index)
    pmra_gaia = np.delete(pmra_gaia, bad_index)
    pmdec_gaia = np.delete(pmdec_gaia, bad_index)
    parallax_gaia = np.delete(parallax_gaia, bad_index)
    mag_gaia = np.delete(mag_gaia, bad_index)
    absmag = mag_gaia + 5 * (np.log10(parallax_gaia / 1000) + 1)

    m1, m2, sep = astroNN.datasets.xmatch.xmatch(ra_apogee, ra_gaia, maxdist=2, colRA1=ra_apogee, colDec1=dec_apogee, epoch1=2000.,
                         colRA2=ra_gaia, colDec2=dec_gaia, epoch2=2015., colpmRA2=pmra_gaia, colpmDec2=pmdec_gaia, swap=True)

    print('Numer: ', len(m1))

    train_len = int(len(m2)*0.6)

    for tt in ['train', 'test']:
        spec = []
        if tt == 'train':
            filtered_index = m1[0:train_len]
            m2_2 = m2[0:train_len]
        else:
            filtered_index = m1[train_len:]
            m2_2 = m2[train_len:]

        for index in filtered_index:
            apogee_id = ((hdulist[1].data['APOGEE_ID'])[filtered_apogee_index])[index]
            location_id = ((hdulist[1].data['LOCATION_ID'])[filtered_apogee_index])[index]

            warningflag = None
            if apogee_dr == 14:
                filename = 'aspcapStar-r8-l31c.2-{}.fits'.format(apogee_id)
                path = os.path.join(_APOGEE_DATA, 'dr14/apogee/spectro/redux/r8/stars/l31c/l31c.2/', str(location_id),
                                    filename)
                if not os.path.exists(path):
                    warningflag = astroNN.apogeetools.downloader.combined_spectra(dr=apogee_dr, location=location_id,
                                                                                  apogee=apogee_id)
            else:
                raise ValueError('astroNN only supports DR13 and DR14 APOGEE')
            if warningflag is None:
                combined_file = fits.open(path)
                _spec = combined_file[1].data  # Pseudo-comtinumm normalized flux
                _spec = gap_delete(_spec, dr=apogee_dr)
                spec.extend([_spec])

        print('Creating {}_{}.h5'.format(h5name, tt))
        h5f = h5py.File('{}_{}.h5'.format(h5name, tt), 'w')
        h5f.create_dataset('spectra', data=spec)
        h5f.create_dataset('absmag', data=absmag[m2_2])

    return None
