# Author: bbrighttaer
# Project: ivpgan
# Date: 7/9/19
# Time: 1:03 PM
# File: sim_data.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os


def _to_dict(data):
    # if it is a literal or leave node (value) then no further processing is needed (base condition)
    if not (isinstance(data, DataNode) or isinstance(data, list) or isinstance(data, set)
            or isinstance(data, dict)):
        return data

    # if it is a ``DataNode`` then it should be processed further
    elif isinstance(data, DataNode):
        return {data.label: _to_dict(data.data)}

    # process list and set items
    elif isinstance(data, list) or isinstance(data, set):
        return [_to_dict(d) for d in data]

    # process dict items
    elif isinstance(data, dict):
        return {k: _to_dict(data[k]) for k in data}


class DataNode(object):
    """Gathers simulation data in a resource tree for later analysis"""

    def __init__(self, label, data=None):
        self.label = label
        self.data = data

    def to_json(self, path="./"):
        with open(os.path.join(path, self.label + ".json"), "w") as f:
            json.dump({self.label: _to_dict(self.data)}, f)

    def to_json_str(self):
        return json.dumps({self.label: _to_dict(self.data)})
