# Author: bbrighttaer
# Project: ivpgan
# Date: 5/24/19
# Time: 12:56 AM
# File: test_math.py


from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import unittest
from ivpgan.utils.math import *


class MathsTest(unittest.TestCase):
    def test_segment_sum(self):
        segment_ids = torch.tensor([0, 0, 0, 1, 2, 2, 3, 3])
        data = torch.tensor([5, 1, 7, 2, 3, 4, 1, 3])
        tensor = segment_sum(data, segment_ids)
        self.assertEqual(4, torch.sum(torch.tensor([13, 2, 7, 4]) == tensor))

    def test_unsorted_segment_sum(self):
        segment_ids = torch.tensor([0, 0, 0, 1, 2, 2, 3, 3])
        data = torch.tensor([5., 1, 7, 2, 3, 4, 1, 3])
        tensor = unsorted_segment_sum(data, segment_ids, 4)
        self.assertEqual(4, torch.sum(torch.tensor([13., 2, 7, 4]) == tensor))
        print("\n", tensor)

        index = torch.tensor([0, 0, 1, 1, 0, 1])
        data = torch.tensor([[5., 1., 7., 2., 3., 4.],
                             [5., 1., 7., 2., 3., 4.]]).t()
        tensor = unsorted_segment_sum(data, index, 2)
        answer = torch.tensor([[9., 9.],
                               [13., 13.]])
        print(tensor)
        self.assertEqual(4, torch.sum(answer == tensor), 'Unsorted segment sum test failed')
        print('\n', tensor)

    def test_unsorted_segment_max(self):
        segment_ids = torch.tensor([0, 0, 0, 1, 2, 2, 3, 3])
        data = torch.tensor([5., 1, 7, 2, 3, 4, 1, 3])
        tensor = unsorted_segment_max(data, segment_ids, 4)
        self.assertEqual(4, torch.sum(torch.tensor([7., 2, 4, 3]) == tensor))
        print("\n", tensor)

        index = torch.tensor([0, 0, 1, 1, 0, 1])
        data = torch.tensor([[5., 1., 7., 2., 3., 4.],
                             [5., 1., 7., 2., 3., 4.]]).t()
        tensor = unsorted_segment_max(data, index, 2)
        answer = torch.tensor([[5., 5.],
                               [7., 7.]])
        print(tensor)
        self.assertEqual(4, torch.sum(answer == tensor), 'Unsorted segment sum test failed')
        print('\n', tensor)

    def test_factors(self):
        res = factors(c_dim=1176, a_dim=49, limit=2000)
        print(res)

