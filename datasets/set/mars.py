from __future__ import print_function, absolute_import

import os.path as osp
from scipy.io import loadmat
import numpy as np

from utils.serialization import write_json, read_json

import json
import os
import torch
import shutil
import errno


# def mkdir_if_missing(directory):
#     if not osp.exists(directory):
#         try:
#             os.makedirs(directory)
#         except OSError as e:
#             if e.errno != errno.EEXIST:
#                 raise


# def read_json(fpath):
#     with open(fpath, 'r') as f:
#         obj = json.load(f)
#     return obj

# def write_json(obj, fpath):
#     mkdir_if_missing(osp.dirname(fpath))
#     with open(fpath, 'w') as f:
#         json.dump(obj, f, indent=4, separators=(',', ': '))

class infostruct(object):
	pass


class Mars(object):
	#   change MARS dataset path here


	def __init__(self, root='../data/Mars/', min_seq_len=0):

		self.root = root
		self.train_name_path = osp.join(self.root, 'info/train_name.txt')
		self.test_name_path = osp.join(self.root, 'info/test_name.txt')
		self.track_train_info_path = osp.join(self.root, 'info/tracks_train_info.mat')
		self.track_test_info_path = osp.join(self.root, 'info/tracks_test_info.mat')
		self.query_IDX_path = osp.join(self.root, 'info/query_IDX.mat')
		self.split_train_json_path = osp.join(self.root, 'split_train.json')
		self.split_query_json_path = osp.join(self.root, 'split_query.json')
		self.split_gallery_json_path = osp.join(self.root, 'split_gallery.json')

		train_names = self._get_names(self.train_name_path)  # <class 'list'>: <Len: 509914> '0001C1T0001F001.jpg'
		test_names = self._get_names(self.test_name_path)  # <class 'list'>: <Len: 681089> '00-1C1T0001F001.jpg'
		track_train = loadmat(self.track_train_info_path)[
			'track_train_info']  # numpy.ndarray (8298, 4) [[1 16 1 1],[17 95 1 1] ...]
		track_test = loadmat(self.track_test_info_path)[
			'track_test_info']  # numpy.ndarray (12180, 4)  [[1 24 -1 1][25 34 -1 1]]

		query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze()  # numpy.ndarray (1980,)  [4130, 4138...]
		query_IDX -= 1  # index from 0 [4129,4137....]
		track_query = track_test[query_IDX, :]  # 对应行的小段视频信息,[[171610 171649 2 1],[172214 172313 2 2]...]

		gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]  # gallery = 10200
		track_gallery = track_test[gallery_IDX, :]  # <class 'tuple'>: (12180, 4)      [[1 24 -1 1][25 34 -1 1]...]
		self._check_before_run()
		train, num_train_tracklets, num_train_pids, num_train_imgs, num_train_cams, num_train_vids = \
			self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True,
			                   min_seq_len=min_seq_len, json_path=self.split_train_json_path)

		query, num_query_tracklets, num_query_pids, num_query_imgs, query_pid, query_camid = \
			self._process_gallery_data(test_names, track_query, home_dir='bbox_test', relabel=False,
			                           min_seq_len=min_seq_len, json_path=self.split_query_json_path, )

		gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs, gallery_pid, gallery_camid = \
			self._process_gallery_data(test_names, track_gallery, home_dir='bbox_test', relabel=False,
			                           min_seq_len=min_seq_len, json_path=self.split_gallery_json_path)

		num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
		min_num = np.min(num_imgs_per_tracklet)
		max_num = np.max(num_imgs_per_tracklet)
		avg_num = np.mean(num_imgs_per_tracklet)

		num_total_pids = num_train_pids + num_query_pids
		num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

		print("=> MARS loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  subset   | # ids | # tracklets")
		print("  ------------------------------")
		print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
		print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
		print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
		print("  ------------------------------")
		print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
		print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
		print("  ------------------------------")

		self.train = train
		self.query = query
		self.gallery = gallery

		self.num_train_pids = num_train_pids
		self.num_query_pids = num_query_pids
		self.num_gallery_pids = num_gallery_pids

		self.queryinfo = infostruct()
		self.queryinfo.pid = query_pid
		self.queryinfo.camid = query_camid
		self.queryinfo.tranum = num_query_imgs

		self.galleryinfo = infostruct()
		self.galleryinfo.pid = gallery_pid
		self.galleryinfo.camid = gallery_camid
		self.galleryinfo.tranum = num_gallery_imgs

		self.num_train_cams = num_train_cams
		self.num_train_vids = num_train_vids

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.root):
			raise RuntimeError("'{}' is not available".format(self.root))
		if not osp.exists(self.train_name_path):
			raise RuntimeError("'{}' is not available".format(self.train_name_path))
		if not osp.exists(self.test_name_path):
			raise RuntimeError("'{}' is not available".format(self.test_name_path))
		if not osp.exists(self.track_train_info_path):
			raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
		if not osp.exists(self.track_test_info_path):
			raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
		if not osp.exists(self.query_IDX_path):
			raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

	def _get_names(self, fpath):
		names = []
		with open(fpath, 'r') as f:
			for line in f:
				new_line = line.rstrip()
				names.append(new_line)
		return names

	def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0, json_path=''):
		# if osp.exists(json_path):
		#     print("=> {} generated before, awesome!".format(json_path))
		#     split = read_json(json_path)
		#     return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet'], split['num_cams'], split['num_tracks']
		print("=> Automatically generating split (might take a while for the first time, have a coffe)")
		assert home_dir in ['bbox_train', 'bbox_test']
		num_tracklets = meta_data.shape[0]  # 8298  TODO: 要不要增加?
		pid_list = list(set(meta_data[:, 2].tolist()))  # pid = 625 => [1 3 5 7 9...]
		num_pids = len(pid_list)

		if relabel:
			pid2label = {pid: label for label, pid in enumerate(pid_list)}  # {1:0,3:1,5:2,...}
		tracklets = []
		num_imgs_per_tracklet = []
		cams = []

		for tracklet_idx in range(num_tracklets):
			data = meta_data[tracklet_idx, ...]  # [1 16 1 1]
			start_index, end_index, pid, camid = data

			cams += [int(camid)]

			if pid == -1:
				continue  # junk images are just ignored
			assert 1 <= camid <= 6
			if relabel:
				pid = pid2label[pid]  # pid = 0
			camid -= 1
			# index starts from 0
			img_names = names[start_index - 1:end_index]
			# <class 'list'>:['0001C1T0001F001.jpg'.. '0001C1T0001F016.jpg']

			# make sure image names correspond to the same person
			pnames = [img_name[:4] for img_name in img_names]  # pnames = ['0001','0001'...]
			assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

			# make sure all images are captured under the same camera
			camnames = [img_name[5] for img_name in img_names]  # camnames = ['1','1'...]
			assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

			# append image names with directory information
			# '/media/ying/0BDD17830BDD1783/ReIdDataset/Mars/bbox_train/0001/0001C1T0001F001.jpg'
			img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]  # list<16>
			# print(img_paths)

			if len(img_paths) >= min_seq_len:
				img_paths = tuple(img_paths)
				tracklets.append(
					(img_paths, int(pid), int(camid), 1))  # (('.jpg','.jpg','每张图片的路径'), 0'行人id', 0'camid' trackid)
				num_imgs_per_tracklet.append(len(img_paths))  # [16,79,15...'每个小段视频包含的图片帧数目']

		num_tracklets = len(tracklets)  # 8298

		cams = set(cams)
		num_cams = len(cams)

		print("Saving split to {}".format(json_path))
		split_dict = {
			'tracklets': tracklets,
			'num_tracklets': num_tracklets,
			'num_pids': num_pids,
			'num_imgs_per_tracklet': num_imgs_per_tracklet,
			'num_cams': num_cams,
			'num_tracks': 1
		}
		# write_json(split_dict, json_path)

		return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_cams, 1

	def _process_gallery_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0, json_path=''):
		# if osp.exists(json_path):
		#     print("=> {} generated before, awesome!".format(json_path))
		#     split = read_json(json_path)
		#     return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet'], split['pids'], split['camid']
		visual_pid = [68, 202, 254, 256, 284, 298, 348, 358, 378, 422, 582, 830, 1016, 1100, 1122, 1126, 1190, 1208,
		              1292, 1408]
		assert home_dir in ['bbox_train', 'bbox_test']
		num_tracklets = meta_data.shape[0]  # 8298  TODO: 要不要增加?
		pid_list = list(set(meta_data[:, 2].tolist()))  # pid = 625 => [1 3 5 7 9...]
		num_pids = len(pid_list)  # 626  622

		if relabel:
			pid2label = {pid: label for label, pid in enumerate(pid_list)}  # {1:0,3:1,5:2,...}
		tracklets = []
		num_imgs_per_tracklet = []
		gallery_pid = []
		gallery_camid = []

		for tracklet_idx in range(num_tracklets):
			data = meta_data[tracklet_idx, ...]  # [1 16 1 1]
			start_index, end_index, pid, camid = data
			if pid in visual_pid:
				if pid == -1:
					continue  # junk images are just ignored
				assert 1 <= camid <= 6
				if relabel:
					pid = pid2label[pid]  # pid = 0
				camid -= 1
				# index starts from 0
				img_names = names[start_index - 1:end_index]
				# <class 'list'>:['0001C1T0001F001.jpg'.. '0001C1T0001F016.jpg']

				# make sure image names correspond to the same person
				pnames = [img_name[:4] for img_name in img_names]  # pnames = ['0001','0001'...]
				assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

				# make sure all images are captured under the same camera
				camnames = [img_name[5] for img_name in img_names]  # camnames = ['1','1'...]
				assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

				# append image names with directory information
				# '/media/ying/0BDD17830BDD1783/ReIdDataset/Mars/bbox_train/0001/0001C1T0001F001.jpg'
				img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in
				             img_names]  # list<16>
				if len(img_paths) >= min_seq_len:
					img_paths = tuple(img_paths)
					tracklets.append(
						(img_paths, int(pid), int(camid), 1))  # (('.jpg','.jpg','每张图片的路径'), 0'行人id', 0'camid' )
					num_imgs_per_tracklet.append(len(img_paths))  # [16,79,15...'每个小段视频包含的图片帧数目']
				gallery_pid.append(int(pid))
				gallery_camid.append(int(camid))
		num_tracklets = len(tracklets)  # 8298
		print("Saving split to {}".format(json_path))
		split_dict = {
			'tracklets': tracklets,
			'num_tracklets': num_tracklets,
			'num_pids': num_pids,
			'num_imgs_per_tracklet': num_imgs_per_tracklet,
			'pids': gallery_pid,
			'camid': gallery_camid,
		}
		# write_json(split_dict, json_path)
		return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, gallery_pid, gallery_camid


if __name__ == '__main__':
	# test
	dataset = Mars()