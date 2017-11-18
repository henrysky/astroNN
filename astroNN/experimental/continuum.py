import numpy as np


def chebyshev(dispersion, fluxes, flux_uncertainties, continuum_mask,
              degree=2, regions=None):
    """
    Fit Chebyshev polynomials to the flux values in the continuum mask. Fluxes
    from multiple stars can be given, and the resulting continuum will have the
    same shape as `fluxes`.
    :param dispersion:
        The dispersion values for each of the flux values.
    :param fluxes:
        The flux values.
    :param flux_uncertainties:
        The observational uncertainties associated with the fluxes.
    :param continuum_mask:
        A mask for continuum pixels to use.
    :param degree: [optional]
        The degree of Chebyshev polynomial to use in each region.
    :param regions: [optional]
        Split up the continuum fitting into different wavelength regions. For
        example, APOGEE spectra could be splitted into three chunks:
        `[(15150, 15800), (15890, 16430), (16490, 16950)]`
    """

    if regions is None:
        region_masks = [np.ones_like(dispersion, dtype=bool)]
    else:
        region_masks = \
            [(e >= dispersion) * (dispersion >= s) for s, e in regions]

    dispersion = np.array(dispersion).flatten()
    fluxes = np.atleast_2d(fluxes)
    flux_uncertainties = np.atleast_2d(flux_uncertainties)

    N_stars = fluxes.shape[0]
    assert fluxes.shape[1] == dispersion.size

    i_variances = 1.0 / flux_uncertainties ** 2

    # Use only continuum pixels.
    i_variances[:, ~continuum_mask] += 200 ** 2
    # i_variances[~np.isfinite(i_variances)] = 0

    continuum = np.ones_like(fluxes, dtype=float)
    for i, (flux, i_var) in enumerate(zip(fluxes, i_variances)):
        for region_mask in region_masks:
            fitted_mask = region_mask * continuum_mask
            f = np.polynomial.chebyshev.Chebyshev.fit(
                x=dispersion[fitted_mask], y=flux[fitted_mask],
                w=i_var[fitted_mask], deg=degree)

            continuum[i, region_mask] *= f(dispersion[region_mask])

    return continuum
