import os

def main():
	# train_cmd = "python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset ltcc --cfg configs/res50_cels_cal.yaml --gpu 0,1"
	train_cmd = "python train.py"
	# print(train_cmd)
	os.system(train_cmd)


if __name__ == "__main__":
	main()
