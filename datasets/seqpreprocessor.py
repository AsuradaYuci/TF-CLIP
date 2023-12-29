from __future__ import absolute_import
import os.path as osp
import torch
from PIL import Image


class SeqTrainPreprocessor(object):
    def __init__(self, seqset, dataset, seq_len, transform=None):
        super(SeqTrainPreprocessor, self).__init__()
        self.seqset = seqset
        self.identities = dataset.identities
        self.transform = transform
        self.seq_len = seq_len
        self.root = [dataset.images_dir]
        self.root.append(dataset.other_dir)

    def __len__(self):
        return len(self.seqset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):

        start_ind, end_ind, pid, label, camid = self.seqset[index]

        imgseq = []
        flowseq = []
        for ind in range(start_ind, end_ind):
            fname = self.identities[pid][camid][ind]
            fpath_img = osp.join(self.root[0], fname)
            imgrgb = Image.open(fpath_img).convert('RGB')
            fpath_flow = osp.join(self.root[1], fname)
            flowrgb = Image.open(fpath_flow).convert('RGB')
            imgseq.append(imgrgb)
            flowseq.append(flowrgb)

        while len(imgseq) < self.seq_len:
            imgseq.append(imgrgb)
            flowseq.append(flowrgb)

        seq = [imgseq, flowseq]

        if self.transform is not None:
            seq = self.transform(seq)

        img_tensor = torch.stack(seq[0], 0)

        flow_tensor = torch.stack(seq[1], 0)

        return img_tensor, flow_tensor, label, camid


class SeqTestPreprocessor(object):

    def __init__(self, seqset, dataset, seq_len, transform=None):
        super(SeqTestPreprocessor, self).__init__()
        self.seqset = seqset
        self.identities = dataset.identities
        self.transform = transform
        self.seq_len = seq_len
        self.root = [dataset.images_dir]
        self.root.append(dataset.other_dir)

    def __len__(self):
        return len(self.seqset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):

        start_ind, end_ind, pid, label, camid = self.seqset[index]

        imgseq = []
        flowseq = []
        for ind in range(start_ind, end_ind):
            fname = self.identities[pid][camid][ind]
            fpath_img = osp.join(self.root[0], fname)
            imgrgb = Image.open(fpath_img).convert('RGB')
            fpath_flow = osp.join(self.root[1], fname)
            flowrgb = Image.open(fpath_flow).convert('RGB')
            imgseq.append(imgrgb)
            flowseq.append(flowrgb)

        while len(imgseq) < self.seq_len:
            imgseq.append(imgrgb)
            flowseq.append(flowrgb)

        seq = [imgseq, flowseq]

        if self.transform is not None:
            seq = self.transform(seq)

        img_tensor = torch.stack(seq[0], 0)

        if len(self.root) == 2:
            flow_tensor = torch.stack(seq[1], 0)
        else:
            flow_tensor = None

        return img_tensor, flow_tensor, pid, camid
