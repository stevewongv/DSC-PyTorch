import os
import sys
import cv2
import argparse
import numpy as np
import logging
import time 
from DSC import DSC
import torch
from torch import nn
from torch.nn import MSELoss
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
import skimage.measure as ms
import progressbar
import skimage.io as io
import PIL.Image as I
from dataset import TrainValDataset, TestDataset
from misc import crf_refine
import shutil
from utils import MyWcploss

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
torch.cuda.manual_seed_all(2018)
torch.manual_seed(2018)
torch.backends.cudnn.benchmark = True


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        

class Session:
    def __init__(self):
        self.device = torch.device("cuda")
        
        self.log_dir = './logdir'
        self.model_dir = './SBU_model'
        ensure_dir(self.log_dir)
        ensure_dir(self.model_dir)
        self.log_name = 'train_SBU_alpha_1'
        self.val_log_name = 'val_SBU_alpha_1'
        logger.info('set log dir as %s' % self.log_dir)
        logger.info('set model dir as %s' % self.model_dir)

        self.test_data_path = '../SBU-shadow/SBU-Test/'                             # test dataset txt file path
        self.train_data_path = '../SBU-shadow/SBUTrain4KRecoveredSmall/SBU.txt'          # train dataset txt file path
        
        self.multi_gpu = True
        self.net = DSC().to(self.device)
        self.bce = MyWcploss().to(self.device)
        
        self.step = 0
        self.save_steps = 200
        self.num_workers = 16
        self.batch_size = 4
        self.writers = {}
        self.dataloaders = {}
        self.shuffle = True
        self.opt = optimizer = optim.SGD([
        {'params': [param for name, param in self.net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * 5e-3},
        {'params': [param for name, param in self.net.named_parameters() if name[-4:] != 'bias'],
         'lr': 5e-3, 'weight_decay': 5e-4}
        ], momentum= 0.9)

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)

        out['lr'] = self.opt.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v) 
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    def get_dataloader(self, dataset_name, train_mode=True):
        dataset = {
            True: TrainValDataset,
            False: TestDataset,
        }[train_mode](dataset_name)
        self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=self.batch_size, 
                            shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True)
        if train_mode:
            return iter(self.dataloaders[dataset_name])
        else:
            return self.dataloaders[dataset_name]

    def save_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        if self.multi_gpu : 
            obj = {
                'net': self.net.module.state_dict(),
                'clock': self.step,
                'opt': self.opt.state_dict(),
            }
        else: 
            obj = {
                'net': self.net.state_dict(),
                'clock': self.step,
                'opt': self.opt.state_dict(),
            }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name,mode='train'):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            return
        self.net.load_state_dict(obj['net'])
        if mode == 'train':
            self.step = obj['clock']
        if mode == 'test':
            path = '../realtest/{}/'.format(self.model_dir[2:])
            ensure_dir(path)
            shutil.copy(ckp_path,path)
    
        
    def inf_batch(self, name, batch):
        if name == 'test':
            torch.set_grad_enabled(False)
        O, B,= batch['O'], batch['B']
        O, B = O.to(self.device), B.to(self.device)
        
        predicts= self.net(O)
        predict_4, predict_3, predict_2, predict_1, predict_0, predict_g, predict_f = predicts
        if name == 'test':
            predicts = [F.sigmoid(predict_4), F.sigmoid(predict_3), F.sigmoid(predict_2), \
               F.sigmoid(predict_1), F.sigmoid(predict_0), F.sigmoid(predict_g), \
               F.sigmoid(predict_f)]
            return  predicts

        loss_4 = self.bce(predict_4, B)
        loss_3 = self.bce(predict_3, B)
        loss_2 = self.bce(predict_2, B)
        loss_1 = self.bce(predict_1, B)
        loss_0 = self.bce(predict_0, B)
        loss_g = self.bce(predict_g, B)
        loss_f = self.bce(predict_f, B)
        predicts = [F.sigmoid(predict_4), F.sigmoid(predict_3), F.sigmoid(predict_2), \
               F.sigmoid(predict_1), F.sigmoid(predict_0), F.sigmoid(predict_g), \
               F.sigmoid(predict_f)]
        loss = loss_4 + loss_3 + loss_2 + loss_1 + loss_0 + loss_g + loss_f
        # log
        losses = {
            'loss_all' : loss.item(),
            'loss_0' : loss_0.item(),
            'loss_1' : loss_1.item(),
            'loss_2' : loss_2.item(),
            'loss_3' : loss_3.item(),
            'loss_4' : loss_4.item(),
            'loss_g' : loss_g.item(),
            'loss_f' : loss_f.item()
        }
        
        return  predicts, loss, losses


    def save_mask(self, name, img_lists,m = 0):
        data, label, predicts = img_lists

        data, label= (data.numpy() * 255).astype('uint8'), (label.numpy()  * 255).astype('uint8')

        label = np.tile(label,(3,1,1))

        h, w = 400,400

        gen_num = (2,1)

        predict_4, predict_3, predict_2, predict_1, predict_0, predict_g, predict_f = predicts

        predict_4, predict_3, predict_2, predict_1, predict_0, predict_g, predict_f = \
        (np.tile(predict_4.cpu().data * 255,(3,1,1))).astype('uint8'), \
        (np.tile(predict_3.cpu().data * 255,(3,1,1))).astype('uint8'), \
        (np.tile(predict_2.cpu().data * 255,(3,1,1))).astype('uint8'), \
        (np.tile(predict_1.cpu().data * 255,(3,1,1))).astype('uint8'), \
        (np.tile(predict_0.cpu().data * 255,(3,1,1))).astype('uint8'), \
        (np.tile(predict_g.cpu().data * 255,(3,1,1))).astype('uint8'), \
        (np.tile(predict_f.cpu().data * 255,(3,1,1))).astype('uint8')
        
        img = np.zeros((gen_num[0] * h, gen_num[1] * 9 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], label[idx],predict_4[idx], predict_3[idx], predict_2[idx], predict_1[idx], predict_0[idx], predict_g[idx], predict_f[idx]]
                    for k in range(9):
                        col = (j * 9 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        # print(tmp.shape)
                        img[row: row+h, col: col+w] = tmp 

        img_file = os.path.join(self.log_dir, '%d_%s.jpg' % (self.step, name))
        io.imsave(img_file, img)



def run_train_val(ckp_name='latest'):
    sess = Session()
    sess.load_checkpoints(ckp_name)
    if sess.multi_gpu :
        sess.net = nn.DataParallel(sess.net)
    sess.tensorboard(sess.log_name)
    sess.tensorboard(sess.val_log_name)

    dt_train = sess.get_dataloader(sess.train_data_path)
    dt_val = sess.get_dataloader(sess.train_data_path)

    while sess.step <= 5000:
        # sess.sche.step()
        sess.opt.param_groups[0]['lr'] = 2 * 5e-3 * (1 - float(sess.step) / 5000
                                                                ) ** 0.9
        sess.opt.param_groups[1]['lr'] = 5e-3 * (1 - float(sess.step) / 5000
                                                            ) ** 0.9
        sess.net.train()
        sess.net.zero_grad()
        
        batch_t = next(dt_train)
        
        # out, loss, losses, predicts
        pred_t, loss_t, losses_t = sess.inf_batch(sess.log_name, batch_t)
        sess.write(sess.log_name, losses_t)

        loss_t.backward()

        sess.opt.step() 
        if sess.step % 10 == 0:
            sess.net.eval()
            batch_v = next(dt_val)
            pred_v, loss_v, losses_v = sess.inf_batch(sess.val_log_name, batch_v)
            sess.write(sess.val_log_name, losses_v)
        if sess.step % int(sess.save_steps / 5) == 0:
            sess.save_checkpoints('latest')
        if sess.step % int(sess.save_steps / 10) == 0:
            sess.save_mask(sess.log_name, [batch_t['image'],  batch_t['B'],pred_t])
            if sess.step % 10 == 0:
                sess.save_mask(sess.val_log_name, [batch_v['image'], batch_v['B'],pred_v])
            logger.info('save image as step_%d' % sess.step)
        if sess.step % (sess.save_steps * 5) == 0:
            sess.save_checkpoints('step_%d' % sess.step)
            logger.info('save model as step_%d' % sess.step)
        sess.step += 1


def run_test(ckp_name):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints(ckp_name,'test')
    if sess.multi_gpu :
        sess.net = nn.DataParallel(sess.net)
    sess.batch_size = 1
    sess.shuffle = False
    sess.outs = -1
    dt = sess.get_dataloader(sess.test_data_path, train_mode=False)
    

    input_names = open(sess.test_data_path+'SBU.txt').readlines()
    widgets = [progressbar.Percentage(),progressbar.Bar(),progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets,maxval=len(dt)).start()
    for i, batch in enumerate(dt):

        pred = sess.inf_batch('test', batch)
        image = I.open(sess.test_data_path+input_names[i].split(' ')[0]).convert('RGB')
        final = I.fromarray((pred[-1].cpu().data * 255).numpy().astype('uint8')[0,0,:,:])
        final = np.array(final.resize(image.size))
        final_crf = crf_refine(np.array(image),final)
        ensure_dir('./results')
        io.imsave('./results/'+input_names[i].split(' ')[0].split('/')[1][:-3]+'png',final_crf)
        bar.update(i+1)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', default='test')
    parser.add_argument('-m', '--model', default='latest')

    args = parser.parse_args(sys.argv[1:])
    
    if args.action == 'train':
        run_train_val(args.model)
    elif args.action == 'test':
        run_test(args.model)

