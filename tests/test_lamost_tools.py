import numpy as np
import pytest
from astroNN.lamost import pseudo_continuum, wavelength_solution


def test_wavelength_solution():
    wavelength_solution()
    wavelength_solution(dr=5)
    with pytest.raises(ValueError):
        wavelength_solution(dr=1)


def test_norm():
    pseudo_continuum(np.ones(3909), np.ones(3909))
