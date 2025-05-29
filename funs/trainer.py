# trainer.py

import numpy as np
import torch
import os 

class Trainer:
    def __init__(self, model, loss, optimizer, device):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device

    def train(self, epoch, train_loader):
        print("\nStarting Training... \n" + "-" * 40)
        self.model.train()
        train_loss_list = []
        for epoch in range(0, epoch + 1):
            train_loss = self.training_step(train_loader)
            epoch_loss = np.mean(train_loss)
            print(f'[EPOCH: {epoch}] \nTrain Loss: {epoch_loss:.5f}\n')
            train_loss_list.append(epoch_loss)
        return train_loss_list

    def training_step(self, train_loader):
        train_loss_list = []

        for batch_idx, (data, label) in enumerate(train_loader):
            x = data.to(self.device)
            recon, mu, logvar = self.model(x)
            loss = self.loss(recon, x, mu, logvar)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss_list.append(loss.item())

        return train_loss_list
            
    def eval(self, eval_loader):
        print("\nStarting Evaluation... \n" + "-" * 40)
        self.model.eval()
        eval_loss = self.validation_step(eval_loader)
        print(f'Validation Loss: {np.mean(eval_loss):.5f}')

        return eval_loss

    def validation_step(self, eval_loader):
        eval_loss_list = []

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(eval_loader):
                x = data.to(self.device)
                label = label.to(self.device)

                recon, mu, logvar = self.model(x)
                loss = self.loss(recon, x, mu, logvar)

                eval_loss_list.append(loss.item())

        return eval_loss_list
    
    def save(self, root, model_name):
        os.makedirs(f'{root}/', exist_ok=True)

        torch.save(self.model.state_dict(), f'{root}/{model_name}.pt')