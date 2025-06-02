import funs

import torch
import os

import gc
import numpy as np
from torch.optim import Adam

def main(config):
    # Initialize
    model_name = f'stft_vae'
    funs.set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device)

    data_root_dirs = os.path.join(config.data_dir, config.snr)

    # data building
    machine_types = config.machine_types

    train_dirs = [os.path.join(data_root_dirs, directory, 'train') for directory in machine_types]
    val_dirs = [os.path.join(data_root_dirs, directory, 'val') for directory in machine_types]
    test_dirs = [os.path.join(data_root_dirs, directory, 'test') for directory in machine_types]

    gc.collect()
    torch.cuda.empty_cache()


    # make dataframe
    print('\nCreating train dataframe...\n' + '-' * 40)
    train_df = funs.make_dataframe(config, train_dirs)

    gc.collect()
    torch.cuda.empty_cache()
    print('\nCreating val dataframe...\n' + '-' * 40)
    val_df = funs.make_dataframe(config, val_dirs)

    gc.collect()
    torch.cuda.empty_cache()
    print('\nCreating test dataframe...\n' + '-' * 40)
    test_df = funs.make_dataframe(config, test_dirs)

    # make data, label
    data_list = []
    label_list = []
    for _, row in train_df.iterrows():
        segments = row['stft']
        label = row['label']
        
        for seg in segments:
            data_list.append(seg)        
            label_list.append(label)

    train_data, train_label = np.array(data_list), np.array(label_list)
    
    data_list = []
    label_list = []
    for _, row in val_df.iterrows():
        segments = row['stft']
        label = row['label']
        
        for seg in segments:
            data_list.append(seg)        
            label_list.append(label)

    val_data, val_label = np.array(data_list), np.array(label_list)

    data_list = []
    label_list = []
    for _, row in test_df.iterrows():
        segments = row['stft']
        label = row['label']
        
        for seg in segments:
            data_list.append(seg)        
            label_list.append(label)

    test_data, test_label = np.array(data_list), np.array(label_list)

    # make dataset and dataloader
    train_dataset = funs.MIMIIDataset(train_data, train_label)
    val_dataset = funs.MIMIIDataset(val_data, val_label)
    test_dataset = funs.MIMIIDataset(test_data, test_label)

    train_loader = funs.get_dataloader(train_dataset, config.batch_size, shuffle = True)
    val_loader = funs.get_dataloader(val_dataset, config.batch_size, shuffle = False)
    test_loader = funs.get_dataloader(test_dataset, config.batch_size, shuffle = False)

    model = funs.Conv2DVAE(latent_dim=32).to(device)
    optimizer = Adam(model.parameters(), lr = float(config.learning_rate))
    loss = funs.VAELoss()

    # train 
    trainer = funs.Trainer(model, loss, optimizer, device)
    train_loss_list = trainer.train(config.epoch, train_loader)
    trainer.save(config.model_root, model_name=model_name)

    loss = funs.VAELoss(reduction='none')
    trainer = funs.Trainer(model, loss, optimizer, device)
    model_path = f'{config.model_root}/{model_name}.pt'
    trainer.model.load_state_dict(torch.load(model_path, weights_only=True))

    eval_loss_list, eval_auc_dic = trainer.eval(val_loader)
    test_loss_list, test_auc_dic = trainer.eval(test_loader)

if __name__=='__main__':
    config = funs.load_yaml('./config.yaml')

    main(config)