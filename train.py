import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from generator import custom_dataset
from model import EAST
from loss import Loss
import os
import time
import numpy as np
import cfg

def train(train_img_path, pths_path, batch_size, lr,decay, num_workers, epoch_iter, interval,pretained):
	file_num = len(os.listdir(train_img_path))
	trainset = custom_dataset(train_img_path)
	train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)
	
	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST()
	# TODO 可能是bug
	if os.path.exists(pretained):
		model.load_state_dict(torch.load(pretained))

	data_parallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=decay)
	# scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)

	for epoch in range(epoch_iter):	
		model.train()
		optimizer.step()
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, gt_map) in enumerate(train_loader):
			start_time = time.time()
			img, gt_map = img.to(device),gt_map.to(device)
			east_detect = model(img)
			loss = criterion(gt_map, east_detect)

			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
		
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)
		if (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(pths_path, cfg.train_task_id+'_model_epoch_{}.pth'.format(epoch+1)))


# def test():


if __name__ == '__main__':
	train_img_path = os.path.join(cfg.data_dir,cfg.train_image_dir_name)
	pths_path      = './saved_model'
	batch_size     = 10
	lr             = 1e-3
	decay          =5e-4
	num_workers    = 4
	epoch_iter     = 600
	save_interval  = 5
	pretained = './saved_model/mb3_512_model_epoch_535.pth'
	train(train_img_path, pths_path, batch_size, lr, decay,num_workers, epoch_iter, save_interval,pretained)
	
