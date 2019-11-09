# Author: bbrighttaer
# Project: ivpgan
# Date: 5/23/19
# Time: 11:19 AM
# File: test_io.py


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import ivpgan.utils.io as io
from ivpgan.utils.sim_data import DataNode


class IO_Tests(unittest.TestCase):
    def test_logging(self):
        logger = io.get_logger('Test_logger', filename='logger_test', level='info')
        self.assertIsNotNone(logger, 'No logger returned')

    def test_data_node(self):
        root = DataNode("root")

        child1 = DataNode("child1")
        child1.data = list(range(5))

        child2 = DataNode("child2")
        child2.data = list(range(5))

        child3 = DataNode("child3")
        dyna_list = []
        child3.data = {
            "key1": DataNode("val1", dyna_list),
            "key2": 2
        }
        root.data = [child1, child2, child3]
        for i in range(5):
            dyna_list.append(DataNode("dyna_%d" % (i + 1), list(range(3))))
        print(root.to_json_str())
        root.to_json()
