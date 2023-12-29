import torch
from torch.utils.data import DataLoader

import utils.spatial_transforms as ST
import utils.temporal_transforms as TT
import utils.transforms as T
import utils.seqtransforms as SeqT
# from torchvision.transforms import InterpolationMode
# import torchvision.transforms as T

from datasets.video_loader_xh import VideoDataset
from datasets.samplers import RandomIdentitySampler, RandomIdentitySamplerForSeq, RandomIdentitySamplerWYQ
from datasets.seqpreprocessor import SeqTrainPreprocessor, SeqTestPreprocessor

from datasets.set.mars import Mars
from datasets.set.ilidsvidsequence import iLIDSVIDSEQUENCE
from datasets.set.lsvid import LSVID

__factory = {
	'mars': Mars,
	'ilidsvidsequence': iLIDSVIDSEQUENCE,
	'lsvid': LSVID
}


def train_collate_fn(batch):
	"""
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
	imgs, pids, camids, viewids, _ = zip(*batch)
	pids = torch.tensor(pids, dtype=torch.int64)
	viewids = torch.tensor(viewids, dtype=torch.int64)
	camids = torch.tensor(camids, dtype=torch.int64)
	return torch.stack(imgs, dim=0), pids, camids, viewids,


def val_collate_fn(batch):
	imgs, pids, camids, viewids, img_paths = zip(*batch)
	viewids = torch.tensor(viewids, dtype=torch.int64)
	camids_batch = torch.tensor(camids, dtype=torch.int64)
	return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def train_collate_fn_seq(batch):
	"""
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
	imgs, flows, pids, camids = zip(*batch)
	viewids = 1
	pids = torch.tensor(pids, dtype=torch.int64)
	viewids = torch.tensor(viewids, dtype=torch.int64)
	camids = torch.tensor(camids, dtype=torch.int64)
	return torch.stack(imgs, dim=0), pids, camids, viewids,


def val_collate_fn_seq(batch):
	imgs, flows, pids, camids = zip(*batch)
	viewids = 1
	img_paths = None
	viewids = torch.tensor(viewids, dtype=torch.int64)
	camids_batch = torch.tensor(camids, dtype=torch.int64)
	return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):
	split_id = cfg.DATASETS.SPLIT
	seq_srd = cfg.INPUT.SEQ_SRD
	seq_len = cfg.INPUT.SEQ_LEN
	num_workers = cfg.DATALOADER.NUM_WORKERS

	if cfg.DATASETS.NAMES != 'mars' and cfg.DATASETS.NAMES != 'duke' and cfg.DATASETS.NAMES != 'lsvid':

		dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, split_id=split_id, seq_len=seq_len,
		                                        seq_srd=seq_srd, num_val=1)

		num_classes = dataset.num_trainval_ids
		cam_num = dataset.num_train_cams
		view_num = dataset.num_train_vids

		train_set = SeqTrainPreprocessor(dataset.trainval, dataset, seq_len,
		                                 transform=SeqT.Compose([SeqT.RectScale(256, 128),
		                                                         SeqT.RandomHorizontalFlip(),
		                                                         SeqT.RandomSizedEarser(),
		                                                         SeqT.ToTensor(),
		                                                         SeqT.Normalize(mean=[0.485, 0.456, 0.406],
		                                                                        std=[0.229, 0.224, 0.225])]))

		train_set_normal = SeqTrainPreprocessor(dataset.trainval, dataset, seq_len,
		                                        transform=SeqT.Compose([SeqT.RectScale(256, 128),
		                                                                SeqT.ToTensor(),
		                                                                SeqT.Normalize(mean=[0.485, 0.456, 0.406],
		                                                                               std=[0.229, 0.224, 0.225])]))
		val_set = SeqTestPreprocessor(dataset.query + dataset.gallery, dataset, seq_len,
		                              transform=SeqT.Compose([SeqT.RectScale(256, 128),
		                                                      SeqT.ToTensor(),
		                                                      SeqT.Normalize(mean=[0.485, 0.456, 0.406],
		                                                                     std=[0.229, 0.224, 0.225])]))

		train_loader_stage2 = DataLoader(
			train_set,
			sampler=RandomIdentitySamplerForSeq(dataset.trainval, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
			                                    num_instances=cfg.DATALOADER.NUM_INSTANCE),
			batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
			num_workers=num_workers,
			drop_last=True,
			collate_fn=train_collate_fn_seq,
		)

		train_loader_stage1 = DataLoader(
			train_set_normal,
			batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH,
			shuffle=True,
			num_workers=num_workers,
			drop_last=True,
			collate_fn=train_collate_fn_seq
		)

		val_loader = DataLoader(
			val_set,
			batch_size=cfg.TEST.IMS_PER_BATCH,
			shuffle=False,
			num_workers=num_workers,
			drop_last=False,
			collate_fn=val_collate_fn_seq
		)

	else:
		dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

		transform_train = SeqT.Compose([SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
		                                SeqT.RandomHorizontalFlip(),
		                                SeqT.RandomSizedEarser(),
		                                SeqT.ToTensor(),
		                                SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

		transform_test = SeqT.Compose([SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
		                               SeqT.ToTensor(),
		                               SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

		train_set = VideoDataset(dataset.train, seq_len=seq_len, sample='rrs_train', transform=transform_train)
		train_set_normal = VideoDataset(dataset.train, seq_len=seq_len, sample='dense', transform=transform_test)

		num_classes = dataset.num_train_pids  # 625
		cam_num = dataset.num_train_cams  # 6
		view_num = dataset.num_train_vids  # 1

		train_loader_stage2 = DataLoader(
			train_set,
			sampler=RandomIdentitySampler(dataset.train, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
			                              num_instances=cfg.DATALOADER.NUM_INSTANCE),
			batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
			# batch_size=1,
			num_workers=num_workers,
			drop_last=True,
			pin_memory=True,
			collate_fn=train_collate_fn,
		)

		train_loader_stage1 = DataLoader(
			train_set_normal,
			# batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH,
			batch_size=1,
			shuffle=True,
			num_workers=num_workers,
			drop_last=True,
			pin_memory=True,
			collate_fn=train_collate_fn
		)

		sampler_method = 'rrs_test'
		batch_size_eval = 30

		val_set = VideoDataset(dataset.query + dataset.gallery, seq_len=seq_len, sample=sampler_method,
		                       transform=transform_test)

		val_loader = DataLoader(
			val_set,
			batch_size=batch_size_eval,
			shuffle=False,
			pin_memory=True,
			num_workers=num_workers,
			drop_last=False,
			collate_fn=val_collate_fn
		)

	return train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes, cam_num, view_num


def make_eval_all_dataloader(cfg):
	split_id = cfg.DATASETS.SPLIT
	seq_srd = cfg.INPUT.SEQ_SRD
	seq_len = cfg.INPUT.SEQ_LEN
	num_workers = cfg.DATALOADER.NUM_WORKERS

	if cfg.DATASETS.NAMES != 'mars' and cfg.DATASETS.NAMES != 'duke' and cfg.DATASETS.NAMES != 'lsvid':

		dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, split_id=split_id, seq_len=seq_len,
		                                        seq_srd=seq_srd, num_val=1)

		num_classes = dataset.num_trainval_ids
		cam_num = dataset.num_train_cams
		view_num = dataset.num_train_vids

		val_set = SeqTestPreprocessor(dataset.query + dataset.gallery, dataset, seq_len,
		                              transform=SeqT.Compose([SeqT.RectScale(256, 128),
		                                                      SeqT.ToTensor(),
		                                                      SeqT.Normalize(mean=[0.485, 0.456, 0.406],
		                                                                     std=[0.229, 0.224, 0.225])]))
		val_loader = DataLoader(
			val_set,
			batch_size=cfg.TEST.IMS_PER_BATCH,
			shuffle=False,
			num_workers=num_workers,
			drop_last=False,
			collate_fn=val_collate_fn_seq
		)

	else:
		dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

		transform_test = SeqT.Compose([SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
		                               SeqT.ToTensor(),
		                               SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

		num_classes = dataset.num_train_pids  # 625
		cam_num = dataset.num_train_cams  # 6
		view_num = dataset.num_train_vids  # 1

		sampler_method = 'dense'
		batch_size_eval = 1

		val_set = VideoDataset(dataset.query + dataset.gallery, seq_len=seq_len, sample=sampler_method,
		                       transform=transform_test)

		val_loader = DataLoader(
			val_set,
			batch_size=batch_size_eval,
			shuffle=False,
			pin_memory=True,
			num_workers=num_workers,
			drop_last=False,
			collate_fn=val_collate_fn
		)

	return val_loader, len(dataset.query), num_classes, cam_num, view_num


def make_eval_rrs_dataloader(cfg):
	split_id = cfg.DATASETS.SPLIT
	seq_srd = cfg.INPUT.SEQ_SRD
	seq_len = cfg.INPUT.SEQ_LEN
	num_workers = cfg.DATALOADER.NUM_WORKERS

	if cfg.DATASETS.NAMES != 'mars' and cfg.DATASETS.NAMES != 'duke' and cfg.DATASETS.NAMES != 'lsvid':

		dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, split_id=split_id, seq_len=seq_len,
		                                        seq_srd=seq_srd, num_val=1)

		num_classes = dataset.num_trainval_ids
		cam_num = dataset.num_train_cams
		view_num = dataset.num_train_vids

		val_set = SeqTestPreprocessor(dataset.query + dataset.gallery, dataset, seq_len,
		                              transform=SeqT.Compose([SeqT.RectScale(256, 128),
		                                                      SeqT.ToTensor(),
		                                                      SeqT.Normalize(mean=[0.485, 0.456, 0.406],
		                                                                     std=[0.229, 0.224, 0.225])]))
		val_loader = DataLoader(
			val_set,
			batch_size=cfg.TEST.IMS_PER_BATCH,
			shuffle=False,
			num_workers=num_workers,
			drop_last=False,
			collate_fn=val_collate_fn_seq
		)

	else:
		dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

		transform_test = SeqT.Compose([SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
		                               SeqT.ToTensor(),
		                               SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

		num_classes = dataset.num_train_pids  # 625
		cam_num = dataset.num_train_cams  # 6
		view_num = dataset.num_train_vids  # 1

		sampler_method = 'rrs_test'
		batch_size_eval = 30

		val_set = VideoDataset(dataset.query + dataset.gallery, seq_len=seq_len, sample=sampler_method,
		                       transform=transform_test)

		val_loader = DataLoader(
			val_set,
			batch_size=batch_size_eval,
			shuffle=False,
			pin_memory=True,
			num_workers=num_workers,
			drop_last=False,
			collate_fn=val_collate_fn
		)

	return val_loader, len(dataset.query), num_classes, cam_num, view_num