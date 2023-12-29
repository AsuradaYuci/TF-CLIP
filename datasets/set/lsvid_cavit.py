import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import h5py
import math
import os.path as osp
import numpy as np
from scipy.io import loadmat
from utils.serialization import write_json, read_json

class LSVID(object):
    """
    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 11310 (gallery)
    # cameras: 15
    """

    def __init__(self, root='../data/', sampling_step=64, dense=True, seq_len=16, stride=4):
        self.root = osp.join(root, 'LSVID')
        self.train_name_path = osp.join(self.root, 'info/list_sequence/list_seq_train.txt')
        self.test_name_path = osp.join(self.root, 'info/list_sequence/list_seq_test.txt')
        self.test_query_IDX_path = osp.join(self.root, 'info/data/info_test.mat')
        self.split_json_path = osp.join(self.root, 'data_path')

        self._check_before_run()

        if not osp.exists(self.split_json_path):
            tracklet_train = self._get_names(self.train_name_path)
            tracklet_test = self._get_names(self.test_name_path)

            test_query_IDX = h5py.File(self.test_query_IDX_path, mode='r')['query'][0,:]
            test_query_IDX = np.array(test_query_IDX, dtype=int)
            test_query_IDX -= 1
            tracklet_test_query = tracklet_test[test_query_IDX, :]
            
            test_gallery_IDX = [i for i in range(tracklet_test.shape[0]) if i not in test_query_IDX]
            tracklet_test_gallery = tracklet_test[test_gallery_IDX, :]

            train, num_train_tracklets, num_train_pids, num_train_imgs = \
                self._process_data(tracklet_train, home_dir='tracklet_train', relabel=True)
            
            test_query, num_test_query_tracklets, num_test_query_pids, num_test_query_imgs = \
                self._process_data(tracklet_test_query, home_dir='tracklet_test', relabel=False)

            test_gallery, num_test_gallery_tracklets, num_test_gallery_pids, num_test_gallery_imgs = \
                self._process_data(tracklet_test_gallery, home_dir='tracklet_test', relabel=False)

            print("Saving dataset to {}".format(self.split_json_path))
            dataset_dict = {
                'train': train,
                'num_train_tracklets': num_train_tracklets,
                'num_train_pids': num_train_pids,
                'num_train_imgs': num_train_imgs,
                'test_query': test_query,
                'num_test_query_tracklets': num_test_query_tracklets,
                'num_test_query_pids': num_test_query_pids,
                'num_test_query_imgs': num_test_query_imgs,
                'test_gallery': test_gallery,
                'num_test_gallery_tracklets': num_test_gallery_tracklets,
                'num_test_gallery_pids': num_test_gallery_pids,
                'num_test_gallery_imgs': num_test_gallery_imgs,
            }
            write_json(dataset_dict, self.split_json_path)

        else:
            dataset = read_json(self.split_json_path)
            train = dataset['train']
            num_train_tracklets = dataset['num_train_tracklets']
            num_train_pids = dataset['num_train_pids']
            num_train_imgs = dataset['num_train_imgs']
            test_query = dataset['test_query']
            num_test_query_tracklets = dataset['num_test_query_tracklets']
            num_test_query_pids = dataset['num_test_query_pids']
            num_test_query_imgs = dataset['num_test_query_imgs']
            test_gallery = dataset['test_gallery']
            num_test_gallery_tracklets = dataset['num_test_gallery_tracklets']
            num_test_gallery_pids = dataset['num_test_gallery_pids']
            num_test_gallery_imgs = dataset['num_test_gallery_imgs']
        
        num_imgs_per_tracklet = num_train_imgs + num_test_gallery_imgs + num_test_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_test_gallery_pids
        num_total_tracklets = num_train_tracklets + num_test_gallery_tracklets + num_test_query_tracklets

        self.num_train_pids = num_train_pids
        self.num_train_tracklets = num_train_tracklets

       
        self.num_test_query_tracklets = num_test_query_tracklets
        self.num_test_gallery_tracklets = num_test_gallery_tracklets
        
        self.num_total_pids = num_total_pids
        self.num_total_tracklets = num_total_tracklets

        self.min_num = min_num
        self.max_num = max_num
        self.avg_num = avg_num

        self.train = train
        self.query = test_query
        self.gallery = test_gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_test_query_pids
        self.num_gallery_pids = num_test_gallery_pids
        print("=> LS-VID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset       | # ids | # tracklets")
        print("  ------------------------------")
        print("  train        | {:5d} | {:8d}".format(self.num_train_pids, self.num_train_tracklets))
        # print("  train_dense  | {:5d} | {:8d}".format(self.num_train_pids, len(self.train_dense)))
        print("  test_query   | {:5d} | {:8d}".format(self.num_query_pids, self.num_test_query_tracklets))
        print("  test_gallery | {:5d} | {:8d}".format(self.num_gallery_pids, self.num_test_gallery_tracklets))
        print("  ------------------------------")
        print("  total        | {:5d} | {:8d}".format(self.num_total_pids, self.num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(self.min_num, self.max_num, self.avg_num))
    
    def _check_before_run(self):
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.test_query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.test_query_IDX_path))
    
    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                basepath, pid = new_line.split(' ')
                names.append([basepath, int(pid)])
        return np.array(names)
    
    def _process_data(self, meta_data, home_dir=None, relabel=False):
        assert home_dir in ['tracklet_train', 'tracklet_val', 'tracklet_test']

        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 1].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {int(pid): label for label, pid in enumerate(pid_list)}

        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            tracklet_path = osp.join(self.root, meta_data[tracklet_idx, 0]) + '*'
            img_paths = glob.glob(tracklet_path)
            img_paths.sort()
            pid = int(meta_data[tracklet_idx, 1])
            trackid, _, camid, _ = osp.basename(img_paths[0]).split('_')
            camid = int(camid)
            frame = [int(osp.basename(img_path).split('_')[-3]) for img_path in img_paths]

            camid -= 1
            new_ambi = pid

            num_imgs_per_tracklet.append(len(img_paths))
            tracklets.append((img_paths, pid, camid, trackid, frame, new_ambi))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet    