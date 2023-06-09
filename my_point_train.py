import random
import shutil
from pyuul import utils, VolumeMaker
from torch.utils.data import Dataset,DataLoader
import torch
import pandas as pd
import os
import timeit
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from my_point_net import *
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import time
import wandb

wandb.login()

config_default = {

    'csv_path': '/Users/bivekpokhrel/PycharmProjects/database/data/proteins-2023-04-15.csv',
    'source_folder': '/Users/bivekpokhrel/PycharmProjects/database/data/pdb_3',
    'destination_folder': '/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder',
    'checkpoint_folder': '/Users/bivekpokhrel/PycharmProjects/database/my_pointnet/my_checkpoint',
    'load_previous_epoch': False, # True if want to load
    'resume_epoch':0, # check the last epoch from the saved checkpoint
    'num_protiens': 6
                    }





sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'PointNet_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'optimizer': {
            'values': ['sgd']
        },
        'batch_size': {
            'values': [1,10]
        },
        'learning_rate': {
            'values': [0.001]
        },
        'feature_transform': {
            'values': [True]
        },
        'point_transform': {
            'values': [False]
        },
        'num_epoch': {
            'value': 1 # number of epoch to run after resume epoch
        }
    }
}


sweep_id= wandb.sweep(sweep_config,project="Pointnet_training3")






initial_time = timeit.default_timer()


def check_matches(list_to_check,index_no):
    sample=list_to_check[index_no]
    return sample

def validating_it(trans_df,pdb_list,labels,label_dict):
    """ to validate if pdb_list form trans_df and labels ( not input as it has coordinates) are matching or not
     by accessing it's key value from the label_dictionary"""
    sample_position_list=random.sample(range(0,config_default['num_protiens']),k=1)

    for sample_position in sample_position_list:

        label_value = int(labels[sample_position])
        print(f' !!! open pdb file and check if the {pdb_list[sample_position]} has {get_key_by_value(label_dict, label_value)} localization !!!')


def get_key_by_value(dict_obj, value):
    for k, v in dict_obj.items():
        if v == value:
            return k
    return None


def get_coords(protien_path):
    return utils.parsePDB(protien_path, keep_hetatm=False)[0]


## to make use of the padding (equal size) from pyuul making folder
def create_destination_folder(source,destination,df,col):
    """ make a folder"""
    not_entered_protiens=[]
    try:
        if not os.path.exists(destination):
            print('No directory making it !!')
            os.mkdir(destination)

        if os.path.exists(destination):
            print('Directory there, deleting and making new')
            shutil.rmtree(destination)
            os.mkdir(destination)
    except OSError:
        print(f'Error: Could not create a destination folder {destination}')
        exit()

    for i, pdb_id in enumerate(df[col]):
        file_name = pdb_id + '.pdb'

        try:

            source_file = os.path.join(source, file_name)
            destination_file = os.path.join(destination, file_name)
            # print(source_file)
            # print(destination_file)
            shutil.copy2(source_file, destination_file)


        except IOError:

            not_entered_protiens.append(pdb_id)
            print(f'Could not copy file {file_name} to destination folder {config_default["destination_folder"]}')
    # To make the array of the coordinates from the destination folder
    # input_array=get_coords(destination_folder)
    # print(input_array.shape)

    return not_entered_protiens
# creating a new dataframe for type_id=1 (transmembrane)
def make_lists(csv_path, padding):
    """ read OPM database csv -> select transmembrane -> create labels based on localization ->
    create list of inputs (i.e list of tensor) (pdbid) and labels without padding i.e individual size feed in network"""
    df = pd.read_csv(csv_path)

    trans_a_df = df[df['type_id'] == 1] # Selecting only the transmembrane protien
    trans_df = trans_a_df.iloc[:config_default['num_protiens'], :].copy() #make a copy of 50 pdb to avoid error
    trans_df['pdbid'] = trans_df['pdbid'].str.replace('[^\w]', '', regex=True) #remove "=...." extra charecters

    #make a dictionary from original df and sort them
    location=df['membrane_name_cache'].unique()
    label_dict={key:value for value,key in enumerate(sorted(location))}

    inputs=[] # coordinates -> list [  Num_protiens * tensor(3,NA) ]
    labels=[] # store labels -> list [ Num_protiens * tensor(1)]
    pdb_list=[]# also noting the pdbid in parallel and correct or not can be verified by checking labels and pdbid
    num_classes=24
    invalids = create_destination_folder(config_default['source_folder'], config_default['destination_folder'],trans_df,col='pdbid')
    if padding == False:
        for i, pdb_id in enumerate(trans_df['pdbid']):
            encoding_array = torch.zeros(num_classes)
            file_name=pdb_id+'.pdb'
            file_path=os.path.join(config_default['destination_folder'],file_name)

            inputs.append(get_coords(file_path).squeeze().permute(1,0))
            a=trans_df.loc[trans_df['pdbid'] == pdb_id,'membrane_name_cache'].iloc[0]
            labels.append(torch.tensor(label_dict[a]))
            pdb_list.append(pdb_id)

            #one hot encoding
            # encoding_array[label_dict[a]]=1
            # labels.append(encoding_array)
        validating_it(trans_df,pdb_list,labels,label_dict)
        return inputs,labels,pdb_list
    elif padding == True:
        all_tensor=get_coords(config_default['destination_folder']).permute(0,2,1)

        for i, pdb_id in enumerate(trans_df['pdbid']):
            encoding_array = torch.zeros(num_classes)
            inputs.append(all_tensor[i,:,:])
            file_name=pdb_id+'.pdb'
            file_path=os.path.join(config_default['destination_folder'],file_name)
            a=trans_df.loc[trans_df['pdbid']==pdb_id,'membrane_name_cache'].iloc[0]
            labels.append(torch.tensor(label_dict[a]))
            pdb_list.append(pdb_id)
        validating_it(trans_df,pdb_list,labels,label_dict)
        return inputs, labels, pdb_list
    else:
        print('Check the trans_folder')
        pass
    #

# inputs, labels, pdbs = make_lists(config_default['csv_path'], padding=False)
# print(len(inputs))
# print('***')
# print(inputs[0].shape)
# print('&&&')
# print(inputs[1].shape)
#
#
# print('^^^^^')
# print(len(labels))
# print('***')
# print(labels)
# print('&&&')
# print(labels[1].shape)
#



def collate_fn(batch):
    inputs, labels, indices = zip(*batch)
    return torch.stack(inputs), torch.stack(labels), indices








class Mydataset(Dataset):
    def __init__(self, images, classes):
        self.images=images
        self.classes=classes

    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        a=self.images[index]
        b=self.classes[index]
        a1=a.clone().detach()
        b1=b.clone().detach()
        return a1,b1,index
#create folder
# print('********')
#
#
def build_set(csv_path,padding):

    inputs, labels, pdbs = make_lists(csv_path, padding=padding)

    train_data, val_test_data, train_labels, val_test_labels = train_test_split(inputs, labels, test_size=0.2)
    val_data,test_data, val_labels,test_labels=train_test_split(val_test_data,val_test_labels,test_size=0.5)
    train_set=Mydataset(train_data,train_labels)
    val_set=Mydataset(val_data,val_labels)
    test_set=Mydataset(test_data,test_labels)

    return train_set, val_set, test_set

# print('set')
# inputset, labels, pdbs = build_set(config_default['csv_path'], padding=False)
# try_loader= DataLoader(inputset,batch_size=1)
# for i in try_loader:
#     print(i[0].shape)
# print(len(inputset))
# print('***')
# print(inputset)
# print('&&&')
# print( f' trainset {next(iter(inputset))[0].shape}')
#
# print( f' trainset {inputset}')
#
#
# print('^^^^^')
# print(len(labels))
# print('***')
# print(labels)
# print('&&&')
# print(labels[1].shape)


#
#
#
#     # print(f'train_loader: {len(train_loader)}')
#     # print(f'val_loader: {len(val_loader)}')
#     # print(f'test_loader: {len(test_loader)}')
#
# #
# # # criterion=config['criterion']
# # criterion=nn.CrossEntropyLoss()
#
def build_optimizer(network,learning_rate):

    optimizer=torch.optim.SGD(network.parameters(),lr=learning_rate)

    return optimizer

def train_epoch(network, train_loader, val_loader, optimizer, feature_transform):
    running_loss=0.0
    for i, (input,labels,index) in tqdm(enumerate(train_loader),leave=True,ncols=80):
        network.train()
        output, trans, trans_feat = network(input)
        optimizer.zero_grad()
        # loss=config['criterion'](output,labels)
        loss = F.nll_loss(output, labels)
        if feature_transform == True:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        running_loss +=loss.item()
        loss.backward()

        optimizer.step()

        # if (i+1) % 1 == 0: # to test the accuracy in between
    network.eval()
    with torch.no_grad():
        n_samples=0
        n_correct=0
        for images, label, _ in val_loader:
            outputs = network(images)[0]

            _, prediction = torch.max(outputs, 1) # outputs ->tensor , # max -> (value,index)

            n_samples += label.size(0)
            # print(f'Prediction: {prediction}, label: {label}')

            n_correct += (prediction == label).sum().item()

#
        acc = 100 * n_correct / n_samples
        # wandb.log({'batch_loss': loss.item(), 'accuracy' : acc})



        # wandb.log({'batch_loss':loss.item()})

    return running_loss/len(train_loader), acc

# trainset,valset,testset=build_set(config_default['csv_path'])
# print(f'len : {len(trainset)}')

# print( f' trainset {next(iter(trainset))[0].shape}')
#










def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        batch_size = config.batch_size
        feature_transform=config.feature_transform
        point_transform = config.point_transform

        # num_epoch =config.num_epoch

        num_epoch=1
        resume_epoch=config_default['resume_epoch']
        print(config)
        if batch_size > 1:
            batch_status_of_network = True
            padding=True
        elif batch_size==1:
            batch_status_of_network = False
            padding = False

        model=PointNetCls(k=24,point_transform=point_transform, feature_transform=feature_transform, batch_status_of_network=batch_status_of_network)


        if resume_epoch >= 1 and config_default['load_previous_epoch']:
            checkpoint = torch.load(config_default['checkpoint_folder'] + '_epoch_' + str(resume_epoch) + '.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']

        trainset,valset,testset=build_set(config_default['csv_path'],padding)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        optimizer=build_optimizer(model,learning_rate=config.learning_rate)
        print(config)
        for epoch in range(num_epoch+resume_epoch):
            avg_loss, accu= train_epoch(model,train_loader,val_loader,optimizer,feature_transform=feature_transform)
            wandb.log({'epoch_train_loss': avg_loss, 'val_accuracy' : accu, 'epoch': epoch + resume_epoch + 1})

            checkpoint= {
                'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'epoch_loss' : avg_loss

            }


            torch.save(checkpoint, config_default['checkpoint_folder'] + '_epoch_' + str(epoch) + '.pth') # saving the checkpoint per epoch


wandb.agent(sweep_id,train,count=4)






# # # Create a figure with two subplots
# fig, (ax1, ax2) = plt.subplots(1, 2)
#
# # Plot the first list on the first subplot
# ax1.plot(loss_list)
# ax1.set_xlabel('steps')
# ax1.set_ylabel('loss')
# ax1.set_title('Loss vs num_protiens_trained')
#
# # Plot the second list on the second subplot
# ax2.plot(acc_list)
# ax2.set_xlabel('steps')
# ax2.set_ylabel('Accuracy')
# ax2.set_title('Accuracy vs num_protiens_trained')
#
# # Adjust spacing between subplots
# plt.subplots_adjust(wspace=0.5)
#
# # Show the plot
# plt.show()
#
#
# # # labels=next(iter(train_dataloader))
# # # print(data)
# # # print(f'Shape {data.shape}')

# #
# batch_size = 10
# learning_rate=0.01
# feature_transform=False
# point_transform=False
#
# model=PointNetCls(k=24,point_transform=False, feature_transform=False, batch_status_of_network=True)
#
# trainset,valset,testset=build_set( config_default['csv_path'])
# train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
# val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
# test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
# optimizer=build_optimizer(model,learning_rate=learning_rate)
# for epoch in range(2):
#     avg_loss= train_epoch(model,train_loader,optimizer,feature_transform)
#     print(avg_loss)
#     print(epoch)

#
#
#     #         with torch.no_grad():
#     #             n_correct = 0
#     #             n_samples = 0
#     #
#     #             for images, label, _ in val_loader:
#     #                 outputs = model(images)[0]
#     #                 # outputs ->tensor
#     #                 # max -> (value,index)
#     #                 _, prediction = torch.max(outputs, 1)
#     #                 n_samples += labels.size(0)
#     #                 print(f'Prediction: {prediction}, label: {label}')
#     #
#     #                 n_correct += (prediction == label).sum().item()
#     #
#     #
#     #             acc = 100 * n_correct / n_samples
#     #
#     #             acc_list.append(acc)
#     #             print(f'accuracy: {acc} %')
#     #
#     #
#     # print('Training and validation Complete')
#     # print(loss_list)
#     # print(acc_list)
# # # # Create a figure with two subplots
# # fig, (ax1, ax2) = plt.subplots(1, 2)
# #
# # # Plot the first list on the first subplot
# # ax1.plot(loss_list)
# # ax1.set_xlabel('steps')
# # ax1.set_ylabel('loss')
# # ax1.set_title('Loss vs num_protiens_trained')
# #
# # # Plot the second list on the second subplot
# # ax2.plot(acc_list)
# # ax2.set_xlabel('steps')
# # ax2.set_ylabel('Accuracy')
# # ax2.set_title('Accuracy vs num_protiens_trained')
# #
# # # Adjust spacing between subplots
# # plt.subplots_adjust(wspace=0.5)
# #
# # # Show the plot
# # plt.show()
# #
#
# # print(utils.parsePDB(config['destination_folder'], keep_hetatm=False)[0].shape)
#
# # check_matches(pdbs,3)