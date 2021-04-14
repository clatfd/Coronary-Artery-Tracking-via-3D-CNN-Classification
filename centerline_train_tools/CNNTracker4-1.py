
import sys
import pickle
sys.path.append(r'U:\LiChen\AICafe\CNNTracker')
sys.path.append('..')
from models.centerline_net import CenterlineNet
from data_provider_argu import DataGenerater
from centerline_trainner import Trainer
import torch
import os

def get_dataset(train_patch_list,val_patch_list):
    '''
    :return: train set,val set
    '''
    train_pre_fix_path = r"D:\LiChen\LATTEpatch"
    train_flag = 'train'
    train_transforms = None
    target_transform = None
    train_dataset = DataGenerater(train_patch_list, train_pre_fix_path, 500, train_transforms, train_flag, target_transform)

    val_pre_fix_path = r"D:\LiChen\LATTEpatch"
    val_flag = 'val'
    test_valid_transforms = None
    target_transform = None
    val_dataset = DataGenerater(val_patch_list, val_pre_fix_path, 500, test_valid_transforms, val_flag, target_transform)

    return train_dataset, val_dataset


def cross_entropy(a, y):
    epsilon = 1e-9
    return torch.mean(torch.sum(-y * torch.log10(a + epsilon) - (1 - y) * torch.log10(1 - a + epsilon), dim=1))


if __name__ == '__main__':
    taskdir = '//Desktop4/Dtensorflow\\LiChen\\AICafe\\CNNTracker'
    taskname = 'CNNTracker4-1'
    if not os.path.exists(taskdir + '/' + taskname):
        os.mkdir(taskdir + '/' + taskname)
    pj_prefix = taskdir + '/' + taskname
    dbname = 'CAREIIMERGEGT'
    icafe_dir = r'\\DESKTOP2\GiCafe\result/'+dbname
    with open(icafe_dir+'/db.list','rb') as fp:
        dblist = pickle.load(fp)
    train_list = dblist['train']
    val_list = dblist['val']
    test_list = dblist['test']
    pilist = [pi.split('/')[1] for pi in dblist['test']]
    print(len(pilist),'pilist')

    list_dir = r'D:\LiChen\LATTEpatch\careii_patch\offset\point_500_gp_1'
    train_patch_list = []
    val_patch_list = []
    for pi in train_list:
        if os.path.exists(list_dir + '/d' + pi.split('/')[1] + '_patch_info_500.csv'):
            train_patch_list.append(list_dir + '/d' + pi.split('/')[1] + '_patch_info_500.csv')
        else:
            print('not found',list_dir + '/d' + pi.split('/')[1] + '_patch_info_500.csv')
    for pi in val_list:
        if os.path.exists(list_dir + '/d' + pi.split('/')[1] + '_patch_info_500.csv'):
            val_patch_list.append(list_dir + '/d' + pi.split('/')[1] + '_patch_info_500.csv')
        else:
            print('not found',list_dir + '/d' + pi.split('/')[1] + '_patch_info_500.csv')

    # Here we use 8 fold cross validation, save_num means to use dataset0x as the validation set
    train_dataset, val_dataset = get_dataset(train_patch_list[:], val_patch_list[:])

    curr_model_name = "centerline_net"
    max_points = 500
    model = CenterlineNet(n_classes=max_points)

    batch_size = 512
    num_workers = 12

    criterion = cross_entropy
    inital_lr = 0.001

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=inital_lr,
                                 weight_decay=0.001)

    trainer = Trainer(pj_prefix,
                      batch_size,
                      num_workers,
                      train_dataset,
                      val_dataset,
                      model,
                      curr_model_name,
                      optimizer,
                      criterion,
                      max_points,
                      start_epoch=0,
                      max_epoch=100,
                      initial_lr=inital_lr)
    trainer.run_train()
