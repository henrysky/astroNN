import unittest
import os
import numpy.testing as npt


class UtilitiesTestCase(unittest.TestCase):
    def test_checksum(self):
        import astroNN
        from astroNN.shared.downloader_tools import md5_checksum, sha1_checksum, sha256_checksum
        anderson2017_path = os.path.join(os.path.dirname(astroNN.__path__[0]), 'astroNN', 'data',
                                         'anderson_2017_parallax.npz')
        md5_pred = md5_checksum(anderson2017_path)
        sha1_pred = sha1_checksum(anderson2017_path)
        sha256_pred = sha256_checksum(anderson2017_path)

        # read answer hashed by Windows Get-FileHash
        self.assertEqual(md5_pred, 'E92160A08920447866F91DCBDD7151C0'.lower())
        self.assertEqual(sha1_pred, 'BB8E7CE24672A98CB51DB9FB21424237A0572711'.lower())
        self.assertEqual(sha256_pred, '80AD94EF4631C804171425A0810D90BD0C7E6766364972714BF7CA9FF3C4BDD0'.lower())

    def test_normalizer(self):
        from astroNN.nn.utilities.normalizer import Normalizer
        from astroNN import MAGIC_NUMBER
        import numpy as np

        data = np.random.normal(0, 1, (100, 10))
        magic_idx = (10, 5)
        data[magic_idx] = MAGIC_NUMBER

        # create a normalizer instance
        normer = Normalizer(mode=1)
        norm_data = normer.normalize(data)

        # make sure normalizer preserve magic_number
        self.assertEqual(norm_data[magic_idx], MAGIC_NUMBER)

        # test demoralize
        data_denorm = normer.denormalize(norm_data)
        # make sure demoralizer preserve magic_number
        self.assertEqual(data_denorm[magic_idx], MAGIC_NUMBER)
        npt.assert_array_almost_equal(data_denorm, data)

    def test_cpu_gpu_management(self):
        from astroNN.shared.nn_tools import cpu_fallback

        cpu_fallback(flag=0)
        # os environ is string
        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '-1')

        cpu_fallback(flag=1)
        # make sure flag =1 will delete the environ
        self.assertEqual(any(x == "CUDA_VISIBLE_DEVICES" for x in os.environ), False)


if __name__ == '__main__':
    unittest.main()
