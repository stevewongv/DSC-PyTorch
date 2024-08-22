import os
import sys
import cv2
import argparse
import numpy as np
import logging
import time 
from DSC_sr import DSC
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
from dataset_sr import TrainValDataset, TestDataset
from misc import crf_refine
import shutil
from utils import MyWcploss, ShadowRemovalL1Loss
import time


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
torch.cuda.set_device(0)

iter_num = 320000 #160000

def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        

class Session:
    def __init__(self):
        self.device = torch.device("cuda")
        
        # SRD
        self.log_dir = './SRD256_logdir.v2_tanh'
        self.model_dir = './SRD256_model.v2_tanh'
        ensure_dir(self.log_dir)
        ensure_dir(self.model_dir)
        self.log_name = 'train_SRD256_alpha_1'
        self.val_log_name = 'val_SRD256_alpha_1'
        logger.info('set log dir as %s' % self.log_dir)
        logger.info('set model dir as %s' % self.model_dir)
        # self.test_data_path = '/home/zhxing/Datasets/SRD_inpaint4shadow_fix/'
        self.test_data_path = '/home/zhxing/Datasets/DESOBA_xvision/'
        self.train_data_path = '/home/zhxing/Datasets/SRD_inpaint4shadow_fix/train_dsc.txt'

        # ISTD
        # self.log_dir = './ISTD+256_logdir.v2_tanh'
        # self.model_dir = './ISTD+256_model.v2_tanh'
        # ensure_dir(self.log_dir)
        # ensure_dir(self.model_dir)
        # self.log_name = 'train_ISTD+256_alpha_1'
        # self.val_log_name = 'val_ISTD+256_alpha_1'
        # logger.info('set log dir as %s' % self.log_dir)
        # logger.info('set model dir as %s' % self.model_dir)
        # self.test_data_path = '/home/zhxing/Datasets/ISTD+/'
        # self.train_data_path = '/home/zhxing/Datasets/ISTD+/train_dsc+.txt'

        self.multi_gpu = False
        self.net = DSC().to(self.device)
        # self.bce = MyWcploss().to(self.device)
        self.l1_loss = ShadowRemovalL1Loss().to(self.device)
        
        self.step = 0
        self.save_steps = 20000
        self.num_workers = 16
        self.batch_size = 2 #4
        self.writers = {}
        self.dataloaders = {}
        self.shuffle = True
        self.opt = optimizer = optim.SGD([
        {'params': [param for name, param in self.net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * 5e-3},
        {'params': [param for name, param in self.net.named_parameters() if name[-4:] != 'bias'],
         'lr': 1 * 5e-3, 'weight_decay': 5e-4}
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
        O, B = batch['O'], batch['B']
        O, B = O.to(self.device), B.to(self.device)

        predicts = self.net(O)
        predict_4, predict_3, predict_2, predict_1, predict_0, predict_g, predict_f = predicts
        
        if name == 'test':
            # No sigmoid for shadow removal task
            return predicts

        # Calculate losses without sigmoid
        loss_4 = self.l1_loss(predict_4, B)
        loss_3 = self.l1_loss(predict_3, B)
        loss_2 = self.l1_loss(predict_2, B)
        loss_1 = self.l1_loss(predict_1, B)
        loss_0 = self.l1_loss(predict_0, B)
        loss_g = self.l1_loss(predict_g, B)
        loss_f = self.l1_loss(predict_f, B)

        loss = loss_4 + loss_3 + loss_2 + loss_1 + loss_0 + loss_g + loss_f

        # Log the losses
        losses = {
            'loss_all': loss.item(),
            'loss_0': loss_0.item(),
            'loss_1': loss_1.item(),
            'loss_2': loss_2.item(),
            'loss_3': loss_3.item(),
            'loss_4': loss_4.item(),
            'loss_g': loss_g.item(),
            'loss_f': loss_f.item()
        }

        return predicts, loss, losses


    def save_mask(self, name, img_lists):
        data, label, predicts = img_lists

        # 将数据和标签从LAB转换为RGB，并确保缩放和转换
        data = (data.numpy().transpose(0, 2, 3, 1) * 255).astype('uint8')  # 假设数据格式为 (N, C, H, W)
        label = (label.numpy().transpose(0, 2, 3, 1) * 255).astype('uint8')  # 假设标签格式为 (N, C, H, W)
        
        # 将预测转换为numpy数组，确保它们是3通道图像并缩放到255
        predicts = [
            (predict.cpu().data.numpy().transpose(0, 2, 3, 1) * 255).astype('float32')  # 假设预测格式为 (N, C, H, W)
            for predict in predicts
        ]

        # LAB到RGB转换
        def lab_to_rgb(lab_img):
            lab_img = lab_img.astype('float32')
            lab_img[:, :, 0] = lab_img[:, :, 0] * 100 / 255.0  # L通道范围 [0, 100]
            lab_img[:, :, 1] = lab_img[:, :, 1] - 128  # a通道范围 [-128, 127]
            lab_img[:, :, 2] = lab_img[:, :, 2] - 128  # b通道范围 [-128, 127]
            lab_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
            lab_img = np.clip(lab_img * 255, 0, 255).astype('uint8')
            return lab_img

        data = np.array([lab_to_rgb(img) for img in data])
        label = np.array([lab_to_rgb(img) for img in label])
        predicts = [np.array([lab_to_rgb(img) for img in predict]) for predict in predicts]

        h, w = predicts[-1].shape[1:3]
        num_preds = len(predicts)
        gen_num = (2, 1) if len(data) > 1 else (1, 1)

        # 准备输出图像
        img = np.zeros((gen_num[0] * h, gen_num[1] * (2 + num_preds) * w, 3), dtype='uint8')

        for i in range(gen_num[0]):
            row = i * h
            for j in range(gen_num[1]):
                idx = i * gen_num[1] + j
                tmp_list = [data[idx], label[idx]] + [predict[idx] for predict in predicts]
                
                for k in range(len(tmp_list)):
                    col = (j * (2 + num_preds) + k) * w
                    tmp = tmp_list[k]
                    img[row: row + h, col: col + w] = tmp

        # 保存图像
        img_file = os.path.join(self.log_dir, f'{self.step}_{name}.jpg')
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

    while sess.step <= iter_num:
        # sess.sche.step()
        sess.opt.param_groups[0]['lr'] = 2 * 5e-3 * (1 - float(sess.step) / iter_num
                                                                ) ** 0.9
        sess.opt.param_groups[1]['lr'] = 5e-3 * (1 - float(sess.step) / iter_num
                                                            ) ** 0.9
        sess.net.train()
        sess.net.zero_grad()
        
        try:
            batch_t = next(dt_train)
        except StopIteration:
            dt_train = iter(sess.get_dataloader(sess.train_data_path))
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

# Zhxing, for run_test function
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def crf_refine(image, prediction):
    # Dummy implementation of crf_refine, replace with your actual implementation
    return prediction  # Modify as per your actual CRF implementation

import os
import time
import numpy as np
import torch.nn as nn
import progressbar
from PIL import Image
import cv2

def run_test(ckp_name):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints(ckp_name, 'test')

    num_params = sum(p.numel() for p in sess.net.parameters())
    print(f'Number of model parameters: {num_params}')
    
    if sess.multi_gpu:
        sess.net = nn.DataParallel(sess.net)
    
    sess.batch_size = 1
    sess.shuffle = False
    sess.outs = -1
    dt = sess.get_dataloader(sess.test_data_path, train_mode=False)
    
    input_names = open(os.path.join(sess.test_data_path, 'test_dsc.txt')).readlines() # "test.txt"
    widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=len(dt)).start()
    
    time_all = []
    for i, batch in enumerate(dt):
        start_time = time.time()
        pred = sess.inf_batch('test', batch)
        end_time = time.time()
        time_all.append(end_time - start_time)
        saved_pred = pred[-1]  # tensor, shape 1,3,256,256, value [-1,1], should scaled to LAB space and then scaled to rgb space to save the image

        # Scale the prediction to LAB space
        saved_pred = (saved_pred.cpu().data.numpy().transpose(0, 2, 3, 1) * 255).astype('float32')  # (N, C, H, W) to (N, H, W, C)
        
        # LAB to RGB conversion
        def lab_to_rgb(lab_img):
            lab_img = lab_img.astype('float32')
            lab_img[:, :, 0] = lab_img[:, :, 0] * 100 / 255.0  # L channel range [0, 100]
            lab_img[:, :, 1] = lab_img[:, :, 1] - 128  # a channel range [-128, 127]
            lab_img[:, :, 2] = lab_img[:, :, 2] - 128  # b channel range [-128, 127]
            lab_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
            lab_img = np.clip(lab_img * 255, 0, 255).astype('uint8')
            return lab_img

        saved_pred_rgb = np.array([lab_to_rgb(img) for img in saved_pred])

        # Save the image
        image_name = input_names[i].strip().split('/')[-1]
        output_path = os.path.join('./test_sr/SRD256_DESOBA', image_name)
        # output_path = os.path.join('./test_sr/ISTD+256', image_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(saved_pred_rgb[0]).save(output_path)
        
        bar.update(i + 1)
    
    avg_time = np.mean(time_all)
    avg_fps = 1 / avg_time
    print(f'Inference speed: {avg_fps:.2f} images/second')

    bar.finish()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', default='test')
    parser.add_argument('-m', '--model', default='latest')

    args = parser.parse_args(sys.argv[1:])
    
    if args.action == 'train':
        run_train_val(args.model)
    elif args.action == 'test':
        run_test(args.model)

