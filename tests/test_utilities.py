import os
import unittest

import numpy.testing as npt


class UtilitiesTestCase(unittest.TestCase):
    def test_checksum(self):
        import astroNN
        from astroNN.shared.downloader_tools import filehash
        anderson2017_path = os.path.join(os.path.dirname(astroNN.__path__[0]), 'astroNN', 'data',
                                         'anderson_2017_dr14_parallax.npz')
        md5_pred = filehash(anderson2017_path, algorithm='md5')
        sha1_pred = filehash(anderson2017_path, algorithm='sha1')
        sha256_pred = filehash(anderson2017_path, algorithm='sha256')

        # read answer hashed by Windows Get-FileHash
        self.assertEqual(md5_pred, '9C714F5FE22BB7C4FF9EA32F3E859D73'.lower())
        self.assertEqual(sha1_pred, '733C0227CF93DB0CD6106B5349402F251E7ED735'.lower())
        self.assertEqual(sha256_pred, '36C265C907F440114D747DA21D2A014D32B5E442D541F183C0EE862F5865FD26'.lower())
        self.assertRaises(ValueError, filehash, anderson2017_path, algorithm='sha123')

    def test_normalizer(self):
        from astroNN.nn.utilities.normalizer import Normalizer
        from astroNN.config import MAGIC_NUMBER
        import numpy as np

        data = np.random.normal(0, 1, (100, 10))
        magic_idx = (10, 5)
        data[magic_idx] = MAGIC_NUMBER

        # create a normalizer instance for mode 0
        normer = Normalizer(mode=0)
        norm_data = normer.normalize(data)
        self.assertEqual(norm_data[magic_idx], MAGIC_NUMBER)  # make sure normalizer preserve magic_number
        # test demoralize
        data_denorm = normer.denormalize(norm_data)
        # make sure demoralizer preserve magic_number
        self.assertEqual(data_denorm[magic_idx], MAGIC_NUMBER)
        npt.assert_array_almost_equal(data_denorm, data)
        npt.assert_array_almost_equal(norm_data, data)

        # create a normalizer instance for mode 1
        normer = Normalizer(mode=1)
        norm_data = normer.normalize(data)
        self.assertEqual(norm_data[magic_idx], MAGIC_NUMBER)  # make sure normalizer preserve magic_number
        # test demoralize
        data_denorm = normer.denormalize(norm_data)
        # make sure demoralizer preserve magic_number
        self.assertEqual(data_denorm[magic_idx], MAGIC_NUMBER)
        npt.assert_array_almost_equal(data_denorm, data)

        # test mode='3s' can do identity transformation
        s3_norm = Normalizer(mode='3s')
        data = np.random.normal(0, 1, (100, 10))
        npt.assert_array_almost_equal(s3_norm.denormalize(s3_norm.normalize(data)), data, decimal=5)

        data_8bit = np.random.randint(0, 256, (100, 50, 50))
        normer = Normalizer(mode=255)
        norm_data_8bit = normer.normalize(data_8bit)
        self.assertEqual(np.max(norm_data_8bit), 1.)  # make sure max of normalized image is 1.
        self.assertEqual(np.min(norm_data_8bit), 0.)  # make sure max of normalized image is 0.

        normer = Normalizer(mode={'input': 255, 'aux': 0})
        norm_data_dict = normer.normalize({'input': data_8bit, 'aux': data})
        self.assertEqual(np.max(norm_data_dict['input']), 1.)  # make sure max of normalized image is 1.
        self.assertEqual(np.min(norm_data_dict['input']), 0.)  # make sure max of normalized image is 0.
        npt.assert_array_almost_equal(norm_data_dict['aux'], data)  # make sure aux data is not normalized in this case

        errorous_norm = Normalizer(mode=-1234)
        self.assertRaises(ValueError, errorous_norm.normalize, data)

    def test_cpu_gpu_management(self):
        from astroNN.shared.nn_tools import cpu_fallback

        cpu_fallback(flag=True)
        cpu_fallback(flag=False)

        # make sure flag=2 raise error
        self.assertRaises(ValueError, cpu_fallback, flag=2)

    def test_h5name_check(self):
        from astroNN.datasets.h5 import h5name_check

        # make sure h5name=None raise error
        self.assertRaises(ValueError, h5name_check, None)

    def test_config(self):
        from astroNN.config import config_path

        config_path(flag=0)
        config_path(flag=1)
        config_path(flag=2)

    def test_patching(self):
        import astroNN.data
        from astroNN.shared.patch_util import Patch

        diff = os.path.join(astroNN.data.datapath(), 'tf1_12.patch')
        patch = Patch(diff)
        patch_file_path = "travis_tf_1_12.py"
        if os.path.exists(patch_file_path) is False:
            patch_file_path = os.path.join("tests", "travis_tf_1_12.py")

        with open(patch_file_path, 'r') as f:
            original_text = f.read()

        patch.apply(patch_file_path)
        with open(patch_file_path, 'r') as f:
            patched_text = f.read()

        patch.apply(patch_file_path)
        with open(patch_file_path, 'r') as f:
            patched_twice_text = f.read()

        patch.revert(patch_file_path)
        with open(patch_file_path, 'r') as f:
            unpatched_text = f.read()

        patch.revert(patch_file_path)
        with open(patch_file_path, 'r') as f:
            unpatched_twice_text = f.read()

        # assert patching, patching twice and unpatching work correctly
        self.assertNotEqual(original_text, patched_text)
        self.assertEqual(patched_twice_text, patched_text)
        self.assertEqual(original_text, unpatched_text)
        self.assertEqual(unpatched_twice_text, unpatched_text)
        
    def test_pltstyle(self):
        from astroNN.shared import pylab_style
        
        pylab_style()


    # def test_loader(self):
    #     import numpy as np
    #
    #     a = np.random.normal(0, 1, (100000, 7514))
    #     b = np.random.normal(0, 1, (7514))
    #     c = np.random.normal(0, 1, (7514))
    #
    #     func = lambda a,b,c: (a - b)/c
    #
    #     for i in range(int(100000/10000)):
    #         func(a[i:i*10000], b, c)


if __name__ == '__main__':
    unittest.main()
