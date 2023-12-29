import os.path as osp
import numpy as np
import h5py
import glob
from utils.serialization import write_json, read_json


class LSVID(object):

    def __init__(self, root=None, sampling_step=48, *args, **kwargs):
        self._root = root
        self.train_name_path = osp.join(self._root, 'info/list_sequence/list_seq_train.txt')
        self.test_name_path = osp.join(self._root, 'info/list_sequence/list_seq_test.txt')
        self.query_IDX_path = osp.join(self._root, 'info/data/info_test.mat')

        self._check_before_run()

        # prepare meta data
        track_train = self._get_names(self.train_name_path)
        track_test = self._get_names(self.test_name_path)

        track_train = np.array(track_train)
        track_test = np.array(track_test)

        query_IDX = h5py.File(self.query_IDX_path, mode='r')['query'][0,:]   # numpy.ndarray (1980,)
        query_IDX = np.array(query_IDX, dtype=int)

        query_IDX -= 1  # index from 0
        track_query = track_test[query_IDX, :]

        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]

        self.split_train_dense_json_path = osp.join(self._root,'split_train_dense_{}.json'.format(sampling_step))
        self.split_train_json_path = osp.join(self._root, 'split_train.json')
        self.split_query_json_path = osp.join(self._root, 'split_query.json')
        self.split_gallery_json_path = osp.join(self._root, 'split_gallery.json')

        train, num_train_tracklets, num_train_pids, num_train_imgs, num_train_cams, num_train_vids = \
            self._process_data(track_train, json_path=self.split_train_json_path, relabel=True)

        train_dense, num_train_tracklets_dense, num_train_pids_dense, num_train_imgs_dense, _, _ = \
            self._process_data(track_train, json_path=self.split_train_dense_json_path, relabel=True, sampling_step=sampling_step)

        query, num_query_tracklets, num_query_pids, num_query_imgs, _, _ = \
            self._process_data(track_query, json_path=self.split_query_json_path, relabel=False)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, _, _ = \
            self._process_data(track_gallery, json_path=self.split_gallery_json_path, relabel=False)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        print("=> LS-VID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        if sampling_step != 0:
            print("  train_d  | {:5d} | {:8d}".format(num_train_pids_dense, num_train_tracklets_dense))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        if sampling_step != 0:
            self.train = train_dense
        else:
            self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

        self.num_train_cams = num_train_cams
        self.num_train_vids = num_train_vids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self._root):
            raise RuntimeError("'{}' is not available".format(self._root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                basepath, pid = new_line.split(' ')
                names.append([basepath, int(pid)])
        return names

    def _process_data(self,
                      meta_data,
                      relabel=False,
                      json_path=None,
                      sampling_step=0):
        if osp.exists(json_path):
            split = read_json(json_path)
            return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet'], split['num_cams'], split['num_tracks']

        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 1].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {int(pid): label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []
        cams = []

        for tracklet_idx in range(num_tracklets):
            tracklet_path = osp.join(self._root, meta_data[tracklet_idx, 0]) + '*'
            img_paths = glob.glob(tracklet_path)  # avoid .DS_Store
            img_paths.sort()
            pid = int(meta_data[tracklet_idx, 1])
            _, _, camid, _ = osp.basename(img_paths[0]).split('_')[:4]
            cams += [int(camid)]
            camid = int(camid)

            if pid == -1: continue  # junk images are just ignored
            assert 1 <= camid <= 15
            if relabel: pid = pid2label[pid]
            camid -= 1  # index starts from 0
            
            if sampling_step == 0:
                tracklets.append((img_paths, pid, camid, 1))
            else:
                num_sampling = len(img_paths) // sampling_step
                for idx in range(num_sampling):
                    if idx == num_sampling - 1:
                        tracklets.append((img_paths[idx * sampling_step:], pid, camid, 1))
                    else:
                        tracklets.append((img_paths[idx * sampling_step: (idx + 1) * sampling_step], pid, camid, 1))
            num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)
        cams = set(cams)
        num_cams = len(cams)

        print("Saving split to {}".format(json_path))
        split_dict = {'tracklets': tracklets, 'num_tracklets': num_tracklets, 'num_pids': num_pids,
            'num_imgs_per_tracklet': num_imgs_per_tracklet, 'num_cams' : num_cams, 'num_tracks' : 1}
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_cams, 1

# class LSVID(object):
#     """
#     LS-VID
#     Reference:
#     Li J, Wang J, Tian Q, Gao W and Zhang S Global-Local Temporal Representations for Video Person Re-Identification[J]. ICCV, 2019
#     Dataset statistics:
#     # identities: 3772
#     # tracklets: 2831 (train) + 3504 (query) + 7829 (gallery)
#     # cameras: 15
#     Note:
#     # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
#     # gallery imgs with label=-1 can be remove, which do not influence on final performance.
#     Args:
#         min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
#     """

#     def __init__(self, root=None, sampling_step=48, *args, **kwargs):
#         self._root = root
#         self.train_name_path = osp.join(self._root, 'list_sequence/list_seq_train.txt')
#         self.test_name_path = osp.join(self._root, 'list_sequence/list_seq_test.txt')
#         self.query_IDX_path = osp.join(self._root, 'test/data/info_test.mat')

#         self._check_before_run()

#         # prepare meta data
#         track_train = self._get_names(self.train_name_path)
#         track_test = self._get_names(self.test_name_path)

#         track_train = np.array(track_train)
#         track_test = np.array(track_test)

#         query_IDX = h5py.File(self.query_IDX_path, mode='r')['query'][0,:]   # numpy.ndarray (1980,)
#         query_IDX = np.array(query_IDX, dtype=int)

#         query_IDX -= 1  # index from 0
#         track_query = track_test[query_IDX, :]

#         gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
#         track_gallery = track_test[gallery_IDX, :]

#         self.split_train_dense_json_path = osp.join(self._root,'split_train_dense_{}.json'.format(sampling_step))
#         self.split_train_json_path = osp.join(self._root, 'split_train.json')
#         self.split_query_json_path = osp.join(self._root, 'split_query.json')
#         self.split_gallery_json_path = osp.join(self._root, 'split_gallery.json')

#         train, num_train_tracklets, num_train_pids, num_train_imgs = \
#             self._process_data(track_train, json_path=self.split_train_json_path, relabel=True)

#         train_dense, num_train_tracklets_dense, num_train_pids_dense, num_train_imgs_dense = \
#             self._process_data(track_train, json_path=self.split_train_dense_json_path, relabel=True, sampling_step=sampling_step)

#         query, num_query_tracklets, num_query_pids, num_query_imgs = \
#             self._process_data(track_query, json_path=self.split_query_json_path, relabel=False)

#         gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
#             self._process_data(track_gallery, json_path=self.split_gallery_json_path, relabel=False)

#         num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
#         min_num = np.min(num_imgs_per_tracklet)
#         max_num = np.max(num_imgs_per_tracklet)
#         avg_num = np.mean(num_imgs_per_tracklet)

#         num_total_pids = num_train_pids + num_gallery_pids
#         num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

#         print("=> LS-VID loaded")
#         print("Dataset statistics:")
#         print("  ------------------------------")
#         print("  subset   | # ids | # tracklets")
#         print("  ------------------------------")
#         print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
#         if sampling_step != 0:
#             print("  train_d  | {:5d} | {:8d}".format(num_train_pids_dense, num_train_tracklets_dense))
#         print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
#         print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
#         print("  ------------------------------")
#         print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
#         print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
#         print("  ------------------------------")

#         if sampling_step != 0:
#             self.train = train_dense
#         else:
#             self.train = train
#         self.query = query
#         self.gallery = gallery

#         self.num_train_pids = num_train_pids
#         self.num_query_pids = num_query_pids
#         self.num_gallery_pids = num_gallery_pids

#     def _check_before_run(self):
#         """Check if all files are available before going deeper"""
#         if not osp.exists(self._root):
#             raise RuntimeError("'{}' is not available".format(self._root))
#         if not osp.exists(self.train_name_path):
#             raise RuntimeError("'{}' is not available".format(self.train_name_path))
#         if not osp.exists(self.test_name_path):
#             raise RuntimeError("'{}' is not available".format(self.test_name_path))
#         if not osp.exists(self.query_IDX_path):
#             raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

#     def _get_names(self, fpath):
#         names = []
#         with open(fpath, 'r') as f:
#             for line in f:
#                 new_line = line.rstrip()
#                 basepath, pid = new_line.split(' ')
#                 names.append([basepath, int(pid)])
#         return names

#     def _process_data(self,
#                       meta_data,
#                       relabel=False,
#                       json_path=None,
#                       sampling_step=0):
#         if osp.exists(json_path):
#             split = read_json(json_path)
#             return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

#         num_tracklets = meta_data.shape[0]
#         pid_list = list(set(meta_data[:, 1].tolist()))
#         num_pids = len(pid_list)

#         if relabel: pid2label = {int(pid): label for label, pid in enumerate(pid_list)}
#         tracklets = []
#         num_imgs_per_tracklet = []

#         vids_per_pid_count = np.zeros(len(pid_list))

#         for tracklet_idx in range(num_tracklets):
#             tracklet_path = osp.join(self._root, meta_data[tracklet_idx, 0]) + '*'
#             img_paths = glob.glob(tracklet_path)  # avoid .DS_Store
#             img_paths.sort()
#             pid = int(meta_data[tracklet_idx, 1])
#             _, _, camid, _ = osp.basename(img_paths[0]).split('_')[:4]
#             camid = int(camid)

#             if pid == -1: continue  # junk images are just ignored
#             assert 1 <= camid <= 15
#             if relabel: pid = pid2label[pid]
#             camid -= 1  # index starts from 0

#             num_sampling = len(img_paths) // sampling_step
#             if num_sampling == 0:
#                 tracklets.append((img_paths, pid, camid))
#             else:
#                 for idx in range(num_sampling):
#                     if idx == num_sampling - 1:
#                         tracklets.append((img_paths[idx * sampling_step:], pid, camid))
#                     else:
#                         tracklets.append((img_paths[idx * sampling_step: (idx + 1) * sampling_step], pid, camid))
#             num_imgs_per_tracklet.append(len(img_paths))

#         num_tracklets = len(tracklets)

#         print("Saving split to {}".format(json_path))
#         split_dict = {'tracklets': tracklets, 'num_tracklets': num_tracklets, 'num_pids': num_pids,
#             'num_imgs_per_tracklet': num_imgs_per_tracklet, }
#         write_json(split_dict, json_path)

#         return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet