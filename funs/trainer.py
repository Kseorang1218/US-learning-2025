# trainer.py

import numpy as np
import torch
import os 
from sklearn.metrics import roc_auc_score

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
        eval_loss, auc_dic = self.validation_step(eval_loader)
        print(f'Validation Loss: {np.mean(eval_loss):.5f}')
        for fault, auc in auc_dic.items():
            print(f'{fault} AUC \t{auc:.5f}')

        return eval_loss, auc_dic

    def validation_step(self, eval_loader):
        eval_loss_list = []

        y_true = []
        y_pred = []
        fault_label_list=[]

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(eval_loader):
                x = data.to(self.device)
                label = label.to(self.device)

                recon, mu, logvar = self.model(x)
                loss = self.loss(recon, x, mu, logvar)

                eval_loss_list.append(loss.sum().item())

                y_true.extend((label > 0).int().tolist())
                y_pred.extend(loss.tolist())
                fault_label_list.extend(label.int().tolist())
            
            auc_dic = self.compute_auc(y_true, y_pred, fault_label_list)

        return eval_loss_list, auc_dic
    
    def compute_auc(self, y_true, y_pred, fault_label_list):
        fault_types = ['normal', 'fan', 'pump', 'slider', 'valve']

        auc_dic = {}
        auc_dic['Total'] = roc_auc_score(y_true, y_pred)

        for fault in fault_types:
            if fault == "normal":
                continue
            else:
                fault_indices = [
                    i
                    for i, label in enumerate(fault_label_list)
                    if (label == fault_types.index(fault) or label == 0)
                ]
                pred_labels = [y_pred[i] for i in fault_indices]
                true_labels = [y_true[i] for i in fault_indices]
                fault_auc = roc_auc_score(true_labels, pred_labels)
                auc_dic[fault] = fault_auc
        
        return auc_dic

    def save(self, root, model_name):
        os.makedirs(f'{root}/', exist_ok=True)
        torch.save(self.model.state_dict(), f'{root}/{model_name}.pt')


class TrainerAE:
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
            recon = self.model(x)
            loss = self.loss(recon, x)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss_list.append(loss.item())

        return train_loss_list
            
    def eval(self, eval_loader):
        print("\nStarting Evaluation... \n" + "-" * 40)
        self.model.eval()
        eval_loss, auc_dic = self.validation_step(eval_loader)
        print(f'Validation Loss: {np.mean(eval_loss):.5f}')
        for fault, auc in auc_dic.items():
            print(f'{fault} AUC \t{auc:.5f}')

        return eval_loss, auc_dic

    def validation_step(self, eval_loader):
        eval_loss_list = []

        y_true = []
        y_pred = []
        fault_label_list=[]

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(eval_loader):
                x = data.to(self.device)
                label = label.to(self.device)

                recon = self.model(x)
                loss = self.loss(recon, x)

                eval_loss_list.append(loss.item())

                # label이 배치이므로 한 개씩 처리
                y_true.extend((label > 0).int().tolist())      # True/False → 1/0
                y_pred.extend([loss.item()] * label.size(0))   # 배치 크기만큼 loss 복제
                fault_label_list.extend(label.int().tolist())  # 원본 라벨 기록
            
            auc_dic = self.compute_auc(y_true, y_pred, fault_label_list)

        return eval_loss_list, auc_dic
    
    def compute_auc(self, y_true, y_pred, fault_label_list):
        fault_types = ['normal', 'fan', 'pump', 'slider', 'valve']

        auc_dic = {}
        auc_dic['Total'] = roc_auc_score(y_true, y_pred)

        for fault in fault_types:
            if fault == "normal":
                continue
            else:
                fault_indices = [
                    i
                    for i, label in enumerate(fault_label_list)
                    if (label == fault_types.index(fault) or label == 0)
                ]
                pred_labels = [y_pred[i] for i in fault_indices]
                true_labels = [y_true[i] for i in fault_indices]
                fault_auc = roc_auc_score(true_labels, pred_labels)
                auc_dic[fault] = fault_auc
        
        return auc_dic

    def save(self, root, model_name):
        os.makedirs(f'{root}/', exist_ok=True)
        torch.save(self.model.state_dict(), f'{root}/{model_name}.pt')