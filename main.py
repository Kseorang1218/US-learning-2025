import funs

import torch
import os

from torch.optim import Adam

def main(config):
    # Initialize
    funs.set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', device)

    data_root_dirs = os.path.join(config.data_dir, config.snr)

    # data building
    machine_types = config.machine_types

    train_dirs = [os.path.join(data_root_dirs, directory, 'train') for directory in machine_types]
    val_dirs = [os.path.join(data_root_dirs, directory, 'val') for directory in machine_types]
    test_dirs = [os.path.join(data_root_dirs, directory, 'test') for directory in machine_types]

    # make dataframe
    print('\nCreating train dataframe...\n' + '-' * 40)
    train_df = funs.make_dataframe(config, train_dirs)
    print('\nCreating val dataframe...\n' + '-' * 40)
    val_df = funs.make_dataframe(config, val_dirs)
    print('\nCreating test dataframe...\n' + '-' * 40)
    test_df = funs.make_dataframe(config, test_dirs)

    # make data, label
    train_data, train_label = funs.get_data_label_arrays(train_df, config.sample_size, config.overlap)
    val_data, val_label= funs.get_data_label_arrays(val_df, config.sample_size, config.overlap)
    test_data, test_label  = funs.get_data_label_arrays(test_df, config.sample_size, config.overlap)

    # make dataset and dataloader
    train_dataset = funs.MIMIIDataset(train_data, train_label)
    val_dataset = funs.MIMIIDataset(val_data, val_label)
    test_dataset = funs.MIMIIDataset(test_data, test_label)

    train_loader = funs.get_dataloader(train_dataset, config.batch_size, shuffle = True)
    val_loader = funs.get_dataloader(val_dataset, config.batch_size, shuffle = False)
    test_loader = funs.get_dataloader(test_dataset, config.batch_size, shuffle = False)

    model = funs.VAE(input_dim=config.sample_size, latent_dim=32).to(device)
    optimizer = Adam(model.parameters(), lr = float(config.learning_rate))
    loss = funs.VAELoss(reduction='sum')

    # train 
    trainer = funs.Trainer(model, loss, optimizer, device)
    train_loss_list = trainer.train(config.epoch, train_loader)
    trainer.save(config.model_root, model_name='vae')

    loss = funs.VAELoss(reduction='none')
    trainer = funs.Trainer(model, loss, optimizer, device)
    model_path = f'{config.model_root}/vae.pt'
    trainer.model.load_state_dict(torch.load(model_path, weights_only=True))

    eval_loss_list, eval_auc_dic = trainer.eval(val_loader)
    test_loss_list, test_auc_dic = trainer.eval(test_loader)

if __name__=='__main__':
    config = funs.load_yaml('./config.yaml')

    main(config)