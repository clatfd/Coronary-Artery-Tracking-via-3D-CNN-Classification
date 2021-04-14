# -*- coding: UTF-8 -*-
# @Time    : 14/05/2020 16:48
# @Author  : BubblyYi
# @FileName: trainner.py
# @Software: PyCharm

import os
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from time import time
import sys
from datetime import datetime
class Trainer(object):
    def __init__(self, pj_prefix, batch_size, num_workers, train_dataset, val_dataset, model, model_name, optimizer, criterion,max_points=500, start_epoch=0, max_epoch=1000, initial_lr=0.01, checkpoint_path=None):
        self.pj_prefix = pj_prefix
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_points = max_points
        self.criterion = criterion
        self.criterion_2 = torch.nn.MSELoss()
        self.bceloss = torch.nn.BCELoss()
        self.all_tr_loss = []
        self.all_val_loss = []

        self.all_tr_direction_loss = []
        self.all_val_direction_loss = []

        self.all_tr_radius_loss = []
        self.all_val_radius_loss = []

        self.all_tr_dis_loss = []
        self.all_val_dis_loss = []

        self.all_tr_err = []
        self.all_val_err = []

        self.best_test_loss = 2**31
        self.log_file = None

        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.checkpoint_path = checkpoint_path
        self.output_folder = "logs"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def train_step(self, epoch):
        self.print_to_log_file("\nEpoch: ", epoch + 1)
        self.model.train()
        training_loss = 0.
        train_loss = 0.
        train_direction_loss = 0.
        train_radius_loss = 0.
        train_dis_loss = 0.
        correct = 0
        total = 0
        punishment_factor = 15
        punishment_factor_2 = 1

        for idx, (inputs, labels, r, d) in enumerate(self.train_loader):
            if idx%1==0:
                print('\rstep',idx,'/',len(self.train_loader),end='')
            inputs, labels, r, d = inputs.to(self.device), labels.to(self.device), r.to(self.device), d.to(self.device)
            outputs, outputs_dis = self.model(inputs)
            outputs = outputs.view((len(labels),self.max_points+1))
            outputs_3 = outputs_dis.view((len(labels)))
            outputs_1 = outputs[:,:len(outputs[0])-1]
            outputs_2 = outputs[:,-1]
            outputs_1 = torch.nn.functional.softmax(outputs_1,1)
            outputs_3 = torch.sigmoid(outputs_3)
            loss_1 = self.criterion(outputs_1.float(), labels.float())
            loss_2 = self.criterion_2(outputs_2.float(), r.float())
            loss_3 = self.bceloss(outputs_3.float(), d.float())
            train_direction_loss+=loss_1.item()
            train_radius_loss+=loss_2.item()
            train_dis_loss+=loss_3.item()

            loss = loss_1+punishment_factor*loss_2+punishment_factor_2*loss_3
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            training_loss += loss.item()
            train_loss += loss.item()
            total += labels.size(0)

        print_str = "Train Loss:{:.5f} Direction Train Loss:{:.5f} Radius Train Loss:{:.5f} Discriminate Train Loss:{:.5f}".format(training_loss / len(self.train_loader),
                                                      train_direction_loss / len(self.train_loader),
                                                      train_radius_loss / len(self.train_loader),
                                                      train_dis_loss / len(self.train_loader))

        self.print_to_log_file(print_str)

        return train_loss / len(self.train_loader), 1. - correct / total, train_direction_loss/ len(self.train_loader),train_radius_loss/ len(self.train_loader), train_dis_loss / len(self.train_loader)

    def val_step(self, epoch):
        self.model.eval()
        test_loss = 0.
        val_direction_loss = 0.
        val_radius_loss = 0.
        val_dis_loss = 0.
        correct = 0
        total = 0
        punishment_factor = 15
        punishment_factor2 = 1
        if True:
            for idx, (inputs, labels, r, d) in enumerate(self.val_loader):
                inputs, labels, r, d = inputs.to(self.device), labels.to(self.device), r.to(self.device), d.to(self.device)
                outputs, outputs_dis = self.model(inputs)

                outputs = outputs.view((len(labels), self.max_points+1))
                outputs_3 = outputs_dis.view((len(labels)))

                outputs_1 = outputs[:, :len(outputs[0]) - 1]
                outputs_2 = outputs[:, -1]
                outputs_1 = torch.nn.functional.softmax(outputs_1,1)
                outputs_3 = torch.sigmoid(outputs_3)

                loss_1 = self.criterion(outputs_1.float(), labels.float())
                loss_2 = self.criterion_2(outputs_2.float(), r.float())
                loss_3 = self.bceloss(outputs_3.float(), d.float())
                val_direction_loss+=loss_1.item()
                val_radius_loss+=loss_2.item()
                val_dis_loss+=loss_3.item()
                loss = loss_1+punishment_factor*loss_2+punishment_factor2*loss_3
                test_loss += loss.item()
                total += labels.size(0)
        print_str = "Val Loss:{:.5f} Direction Val Loss:{:.5f} Radius Val Loss:{:.5f}".format(test_loss / len(self.val_loader),
                        val_direction_loss / len(self.val_loader),
                        val_radius_loss / len(self.val_loader))

        self.print_to_log_file(print_str)
        print("test loss",test_loss/len(self.val_loader))
        print("best test loss", self.best_test_loss)
        if (test_loss/len(self.val_loader)) < self.best_test_loss or epoch%1==0:
            print("saving models")
            self.best_test_loss = test_loss/len(self.val_loader)
            save_fold = self.pj_prefix+"/classification_checkpoints"
            if not os.path.exists(save_fold):
                os.makedirs(save_fold)
            model_save_path = save_fold+"/"+ self.model_name + "_model_Epoch_"+str(epoch)+".pkl"
            self.save_best_checkpoint(model_save_path, test_loss, epoch)
            print_str = "Saving parameters to " + model_save_path
            self.print_to_log_file(print_str)

        return test_loss / len(self.val_loader), 1. - correct / total, val_direction_loss/ len(self.val_loader), val_radius_loss/ len(self.val_loader), val_dis_loss/ len(self.val_loader)

    def poly_lr(self, epoch, max_epochs, initial_lr, exponent=0.9):
        return initial_lr * (1 - epoch / max_epochs) ** exponent

    def lr_decay(self, epoch, max_epochs, initial_lr):
        for params in self.optimizer.param_groups:
            params['lr'] = self.poly_lr(epoch, max_epochs, initial_lr, exponent=1.5)
            lr = params['lr']
            print_str = "Learning rate adjusted to {}".format(lr)
            self.print_to_log_file(print_str)

    def plot_progress(self, epoch):

        x_epoch = list(range(len(self.all_tr_loss)))
        plt.plot(x_epoch, self.all_tr_direction_loss, color="b", linestyle="--", marker="*", label='train')
        plt.plot(x_epoch, self.all_val_direction_loss, color="r", linestyle="--", marker="*", label='val')
        plt.legend()
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        plt.savefig(self.pj_prefix+"/Direction_loss.jpg")
        plt.close()

        plt.plot(x_epoch, self.all_tr_radius_loss, color="b", linestyle="--", marker="*", label='train')
        plt.plot(x_epoch, self.all_val_radius_loss, color="r", linestyle="--", marker="*", label='val')
        plt.legend()
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        plt.savefig(self.pj_prefix+"/Radius_loss.jpg")
        plt.close()

        plt.plot(x_epoch, self.all_tr_dis_loss, color="b", linestyle="--", marker="*", label='train')
        plt.plot(x_epoch, self.all_val_dis_loss, color="r", linestyle="--", marker="*", label='val')
        plt.legend()
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        plt.savefig(self.pj_prefix + "/Dis_loss.jpg")
        plt.close()

        plt.plot(x_epoch, self.all_tr_loss, color="b", linestyle="--", marker="*", label='train')
        plt.plot(x_epoch, self.all_val_loss, color="r", linestyle="--", marker="*", label='val')
        plt.legend()
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        plt.savefig(self.pj_prefix+"/Total_loss.jpg")
        plt.close()


    def save_best_checkpoint(self, model_save_path, acc, epoch):
        checkpoint = {
            'net_dict': self.model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batch_size': self.batch_size,
            'train_loss': self.all_tr_loss,
            'train_err': self.all_tr_err,
            'tr_direction_loss': self.all_tr_direction_loss,
            'val_direction_loss': self.all_val_direction_loss,
            'tr_radius_loss': self.all_tr_radius_loss,
            'val_radius_loss': self.all_val_radius_loss,
            'tr_dis_loss': self.all_tr_dis_loss,
            'val_dis_loss': self.all_val_dis_loss,
            'val_loss': self.all_val_loss,
            'val_err': self.all_val_err,
            'initial_lr': self.initial_lr
        }
        torch.save(checkpoint, model_save_path)

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):

        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            if not os.path.isdir(self.output_folder):
                os.mkdir(self.output_folder)
            timestamp = datetime.now()
            self.log_file = os.path.join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                ctr += 1
        if also_print_to_console:
            print(*args)

    def run_train(self):
        print("Start training")
        self.model.to(self.device)
        if self.start_epoch!=0:
            save_fold = self.pj_prefix+"/classification_checkpoints"
            model_save_path = save_fold+"/"+ self.model_name + "_model_Epoch_"+str(self.start_epoch)+".pkl"
            self.load_last_checkpoint(model_save_path)
        for epoch in range(self.start_epoch, self.max_epoch):
            train_loss, train_err,train_d_loss,train_r_loss,train_dis_loss = self.train_step(epoch)
            val_loss, val_err,val_d_loss,val_r_loss,val_dis_loss = self.val_step(epoch)
            self.all_tr_loss.append(train_loss)
            self.all_tr_err.append(train_err)
            self.all_val_loss.append(val_loss)
            self.all_val_err.append(val_err)
            self.all_tr_direction_loss.append(train_d_loss)
            self.all_tr_radius_loss.append(train_r_loss)
            self.all_tr_dis_loss.append(train_dis_loss)
            self.all_val_direction_loss.append(val_d_loss)
            self.all_val_radius_loss.append(val_r_loss)
            self.all_val_dis_loss.append(val_dis_loss)
            self.plot_progress(epoch)
            self.lr_decay(epoch, self.max_epoch, self.initial_lr)

    def load_last_checkpoint(self, model_save_path):
        checkpoint = torch.load(model_save_path)
        net_dict = checkpoint['net_dict']
        self.model.load_state_dict(net_dict)
        opt = checkpoint['optimizer_state_dict']
        self.optimizer.load_state_dict(opt)
        self.all_tr_loss = checkpoint['train_loss']
        self.all_tr_err = checkpoint['train_err']
        self.all_val_loss = checkpoint['val_loss']
        self.all_val_err = checkpoint['val_err']
        self.initial_lr = checkpoint['initial_lr']
        self.all_tr_direction_loss = checkpoint['tr_direction_loss']
        self.all_val_direction_loss = checkpoint['val_direction_loss']
        self.all_tr_radius_loss = checkpoint['tr_radius_loss']
        self.all_val_radius_loss = checkpoint['val_radius_loss']
        self.all_tr_dis_loss = checkpoint['tr_dis_loss']
        self.all_val_dis_loss = checkpoint['val_dis_loss']

        print('Load from previous checkpoint',model_save_path)

