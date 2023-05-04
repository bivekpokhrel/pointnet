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

    'csv_path': '/Users/bivekpokhrel/PycharmProjects/database/data/proteins-2023-04-15 (1).csv',
    'source_folder' : '/Users/bivekpokhrel/PycharmProjects/database/data/pdb_3',
    'destination_folder' : '/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder',
    'num_protiens': 10,


}

sweep_config = {
    'method' : random}

metric = {
    'name' : 'PointNet_training',
    'goal' : 'minimize'
}

parameter_dict = {
    'optimizer' : {
        'value': ['adam', 'sgd'] },
    'batch_size' : {
        'value' : [1,10,20]
    },

'learning_rate' :
        {
            'min': 0.001,
        'max' : 0.01,
        'distribution' : 'q_log_uniform',
        'q' :5},

    'feature_transform' : {
        'value' : [True, False]
    },

    'point_transform' :
        { 'value' : [True,False]
          },
    'epoch' : {
        'value' : 1
    }

}

sweep_config['metric']= metric
sweep_config['parameters']=parameter_dict
pprint(sweep_config)

sweep_id= wandb.sweep(sweep_config,project="PointNet_sweep")






initial_time = timeit.default_timer()


def check_matches(list_to_check,index_no):
    sample=list_to_check[index_no]
    return sample

def validating_it(trans_df,pdb_list,labels,label_dict):
    """ to validate if pdb_list form trans_df and labels ( not input as it has coordinates) are matching or not
     by accessing it's key value from the label_dictionary"""
    sample_position_list=random.sample(range(0,config_default['num_protiens']),k=10)

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
def make_lists(csv_path, padding=False):
    """ read OPM database csv -> select transmembrane -> create labels based on localization ->
    create list of inputs (pdbid) and labels without padding i.e individual size feed in network"""
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
# print(get_coords(config['destination_folder']).shape)
# print(get_coords(config['destination_folder']).permute(0,2,1).shape)



def collate_fn(batch):
    inputs, labels, indices = zip(*batch)
    return torch.stack(inputs), torch.stack(labels), indices

print('********')


def build_loader(batch_size,csv_path):

    inputs, labels, pdbs = make_lists(csv_path, padding=True)

    train_data, val_test_data, train_labels, val_test_labels = train_test_split(inputs, labels, test_size=0.2)
    val_data,test_data, val_labels,test_labels=train_test_split(val_test_data,val_test_labels,test_size=0.5)
    train_set=Mydataset(train_data,train_labels)
    val_set=Mydataset(val_data,val_labels)
    test_set=Mydataset(test_data,test_labels)





    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0, collate_fn=collate_fn)
    val_loader=DataLoader(val_set,batch_size=10,shuffle=False,num_workers=0, collate_fn=collate_fn)
    test_loader=DataLoader(test_set,batch_size=10,shuffle=False,num_workers=0, collate_fn=collate_fn)
    # print(f'train_loader: {len(train_loader)}')
    # print(f'val_loader: {len(val_loader)}')
    # print(f'test_loader: {len(test_loader)}')

    return train_loader,val_loader,test_loader
#
# # criterion=config['criterion']
# criterion=nn.CrossEntropyLoss()

def build_optimizer(network,learning_rate):

    optimizer=torch.optim.SGD(network.parameters(),lr=learning_rate)

    return optimizer

def train_epoch(network, loader, optimizer,feature_transform):
    running_loss=0.0
    for i, (input,labels,index) in tqdm(enumerate(loader[0]),leave=True,ncols=80):
        output, trans, trans_feat = network(input)
        optimizer.zero_grad()
        # loss=config['criterion'](output,labels)
        loss = F.nll_loss(output, labels)
        if feature_transform == True:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        running_loss +=loss.item()
        loss.backward()

        optimizer.step()

        wandb.log({'batch_loss':loss.item()})

    return running_loss/len(loader)




# # labels=next(iter(train_dataloader))
# # print(data)
# # print(f'Shape {data.shape}')
def train(config=None):
    with wandb.init(config=config):
        config=wandb.config
        model=PointNetCls(k=24,point_transform=config.point_transform, feature_transform=config.feature_tansform, batch_status_of_network=True)

        loader=build_loader(config.batch_size,config_default['csv_path'])
        optimizer=build_optimizer(model,learning_rate=config.learning_rate)
        for epoch in range(config.epochs):
            avg_loss= train_epoch(model,loader,optimizer,feature_transform=config.feature_tansform)
            wandb.log({'loss': avg_loss, 'epoch': epoch})

wandb.agent(sweep_id,train,count=2)




    #         with torch.no_grad():
    #             n_correct = 0
    #             n_samples = 0
    #
    #             for images, label, _ in val_loader:
    #                 outputs = model(images)[0]
    #                 # outputs ->tensor
    #                 # max -> (value,index)
    #                 _, prediction = torch.max(outputs, 1)
    #                 n_samples += labels.size(0)
    #                 print(f'Prediction: {prediction}, label: {label}')
    #
    #                 n_correct += (prediction == label).sum().item()
    #
    #
    #             acc = 100 * n_correct / n_samples
    #
    #             acc_list.append(acc)
    #             print(f'accuracy: {acc} %')
    #
    #
    # print('Training and validation Complete')
    # print(loss_list)
    # print(acc_list)
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

# print(utils.parsePDB(config['destination_folder'], keep_hetatm=False)[0].shape)

# check_matches(pdbs,3)