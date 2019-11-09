# Author: bbrighttaer
# Project: ivpgan
# Date: 5/24/19
# Time: 12:27 AM
# File: math.py


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict, namedtuple

import torch


def segment_sum(data, segment_ids):
    """
    Analogous to tf.segment_sum (https://www.tensorflow.org/api_docs/python/tf/math/segment_sum).

    :param data: A pytorch tensor of the data for segmented summation.
    :param segment_ids: A 1-D tensor containing the indices for the segmentation.
    :return: a tensor of the same type as data containing the results of the segmented summation.
    """
    if not all(segment_ids[i] <= segment_ids[i + 1] for i in range(len(segment_ids) - 1)):
        raise AssertionError("elements of segment_ids must be sorted")

    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have be a 1-D tensor")

    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError("segment_ids should be the same size as dimension 0 of input.")

    # t_grp = {}
    # idx = 0
    # for i, s_id in enumerate(segment_ids):
    #     s_id = s_id.item()
    #     if s_id in t_grp:
    #         t_grp[s_id] = t_grp[s_id] + data[idx]
    #     else:
    #         t_grp[s_id] = data[idx]
    #     idx = i + 1
    #
    # lst = list(t_grp.values())
    # tensor = torch.stack(lst)

    num_segments = len(torch.unique(segment_ids))
    return unsorted_segment_sum(data, segment_ids, num_segments)


def cuda(tensor):
    from ivpgan import cuda
    if cuda:
        return tensor.cuda()
    else:
        return tensor


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long()
        s = cuda(s)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = cuda(torch.zeros(*shape)).scatter_add(0, segment_ids.long(), data.float())
    tensor = tensor.type(data.dtype)
    return tensor


def unsorted_segment_max(data, segment_ids, num_segments):
    # TODO(bbrighttaer): Optimize this function
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_max.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    t_grp = OrderedDict()
    idx = 0
    for i, s_id in enumerate(segment_ids):
        s_id = s_id.item()
        if s_id in t_grp:
            t_grp[s_id] = torch.max(t_grp[s_id], data[idx])
        else:
            t_grp[s_id] = data[idx]
        idx = i + 1

    lst = list(t_grp.values())
    tensor = torch.stack(lst)
    return tensor


# def unsorted_segment_sum(data, segment_ids, num_segments):
#     """
#     Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.
#
#     :param data: A tensor whose segments are to be summed.
#     :param segment_ids: The segment indices tensor.
#     :param num_segments: The number of segments.
#     :return: A tensor of same data type as the data argument.
#     """
#     if len(data.shape) == 1:
#         data = torch.unsqueeze(data.squeeze(), dim=0)
#     if len(segment_ids.shape) == 1:
#         segment_ids = torch.unsqueeze(segment_ids.squeeze(), dim=0)
#     shape = list(segment_ids.shape[:-1]) + [num_segments]
#     zero_tensor = torch.zeros(shape)
#     tensor = zero_tensor.scatter_add(1, segment_ids, data)
#     return tensor

# def segment_sum(data, segment_ids):
#     """
#     Analogous to tf.segment_sum (https://www.tensorflow.org/api_docs/python/tf/math/segment_sum).
#
#     :param data: A pytorch tensor of the data for segmented summation.
#     :param segment_ids: A 1-D tensor containing the indices for the segmentation.
#     :return: a tensor of the same type as data containing the results of the segmented summation.
#     """
#     try:
#         assert data.shape[0] == segment_ids.shape[0]
#     except AssertionError:
#         logger = get_logger(level='error')
#         logger.error("segment_ids should be the same size as dimension 0 of input.")
#
#     grp = {}
#     for i, val in enumerate(data):
#         idx = segment_ids[i].item()
#         val = torch.sum(val)
#         if idx in grp:
#             grp[idx] = grp[idx] + val
#         else:
#             grp[idx] = val
#     rows = list(grp.values())
#     tensor = torch.tensor(rows, dtype=data.dtype)
#     return tensor