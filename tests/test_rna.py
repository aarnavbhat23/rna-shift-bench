import unittest
import numpy as np
import torch
from dataset import mutate, oracle
from transformer import FoldSurrogate, encode
from train import conformal_radius


class RNAInvariants(unittest.TestCase):
    def test_mutation_distance(self):
        source = 'ACGU'*8
        target = mutate(source, 8, np.random.default_rng(3))
        self.assertEqual(sum(a != b for a,b in zip(source, target)), 8)

    def test_padding_is_invisible(self):
        torch.manual_seed(7)
        model = FoldSurrogate().eval()
        with torch.no_grad():
            alone = model(encode(['ACGUACGU']))[0]
            together = model(encode(['ACGUACGU', 'ACGU'*8]))[0]
        torch.testing.assert_close(alone, together, atol=1e-6, rtol=1e-5)

    def test_family_not_row_quantile(self):
        self.assertEqual(conformal_radius(np.arange(20), np.repeat(np.arange(10), 2)), 19)
        self.assertTrue(np.isinf(conformal_radius(np.array([1.]), np.array([0]))))

    def test_oracle_structure(self):
        energy, paired, structure = oracle('GGGGAAAACCCC')
        self.assertLess(energy, 0)
        self.assertGreater(paired, 0)
        self.assertEqual(len(structure), 12)


if __name__ == '__main__':
    unittest.main()
