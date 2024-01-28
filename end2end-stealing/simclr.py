import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
from loss import soft_cross_entropy, wasserstein_loss, soft_nn_loss, \
    pairwise_euclid_distance, SupConLoss, neg_cosine, regression_loss, \
    barlow_loss
import scipy.stats

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, stealing=False, victim_model=None, logdir='', loss=None,
                 *args,
                 **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model']
        self.model = nn.DataParallel(self.model).cuda()
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        try:
            self.retrain = kwargs['retrain']
            print("here")
        except:
            self.retrain = False
        self.log_dir = 'runs/' + logdir
        self.pathpre = f"/scratch/ssd004/scratch/{os.getenv('USER')}/checkpoint"
        if stealing:
            self.log_dir2 = f"{self.pathpre}/SimCLR/{self.args.epochs}{self.args.archstolen}{self.args.losstype}STEAL/"  # save logs here.
        else:
            self.log_dir2 = f"{self.pathpre}/SimCLR/{self.args.epochs}{self.args.arch}{self.args.losstype}TRAIN/"
            if self.retrain:
                self.log_dir2 = f"{self.pathpre}/SimCLR/102resnet34infonceSTEAL/"  # manually setting the path
        self.stealing = stealing
        self.loss = loss
        logname = 'training.log'
        if self.stealing:
            logname = f'training{self.args.datasetsteal}{self.args.num_queries}.log'
        if self.retrain:
            logname = f'retraining{self.args.samples}.log'
        if os.path.exists(os.path.join(self.log_dir2, logname)):
            if self.args.clear == "True":
                os.remove(os.path.join(self.log_dir2, logname))
        else:
            try:
                try:
                    os.mkdir(f"{self.pathpre}/SimCLR")
                    os.mkdir(self.log_dir2)
                except:
                    os.mkdir(self.log_dir2)
            except:
                pass  # print(f"Error creating directory at {self.log_dir2}")
        logging.basicConfig(
            filename=os.path.join(self.log_dir2, logname),
            level=logging.DEBUG)
        if self.stealing:
            self.victim_model = victim_model.to(self.args.device)
        if self.loss == "infonce":
            self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        elif self.loss == "softce":
            self.criterion = soft_cross_entropy
        elif self.loss == "wasserstein":
            self.criterion = wasserstein_loss()
        elif self.loss == "mse":
            self.criterion = nn.MSELoss().to(self.args.device)
        elif self.loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss == "softnn":
            self.criterion = soft_nn_loss
            self.tempsn = self.args.temperaturesn
        elif self.loss == "supcon":
            self.criterion = SupConLoss(temperature=self.args.temperature)
        elif self.loss == "symmetrized":
            self.criterion = nn.CosineSimilarity(dim=1)
        elif self.loss == "barlow":  # method from barlow twins
            self.criterion = barlow_loss
        else:
            raise RuntimeError(f"Loss function {self.loss} not supported.")
        self.criterion2 = nn.CosineSimilarity(dim=1)

    def info_nce_loss(self, features):
        n = int(features.size()[0] / self.args.batch_size)
        labels = torch.cat(
            [torch.arange(self.args.batch_size) for i in range(n)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(
            self.args.device)
        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.log_dir2, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {torch.cuda.is_available()}.")
        logging.info(f"Args: {self.args}")

        for epoch_counter in range(self.args.epochs):
            for images, truelabels in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    if self.loss == "softnn":
                        loss = self.criterion(self.args, features,
                                              pairwise_euclid_distance,
                                              self.tempsn)
                    elif self.loss == "supcon":
                        labels = truelabels
                        bsz = labels.shape[0]
                        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                        features = torch.cat(
                            [f1.unsqueeze(1), f2.unsqueeze(1)],
                            dim=1)
                        loss = self.criterion(features, labels)
                    else:
                        loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {loss}\t")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = f'{self.args.dataset}_checkpoint_{self.args.epochs}_{self.args.losstype}.pth.tar'
        if self.retrain:
            checkpoint_name = f'retrain{self.args.dataset}_checkpoint_{self.args.epochs}_{self.args.losstype}_{self.args.samples}.pth.tar'
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False,
            filename=os.path.join(self.log_dir2, checkpoint_name))
        logging.info(
            f"Model checkpoint and metadata has been saved at {self.log_dir2}")

    def steal(self, train_loader, num_queries):
        self.model.train()
        self.victim_model.eval()
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.log_dir2, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR stealing for {self.args.epochs} epochs.")
        logging.info(f"Using loss type: {self.loss}")
        logging.info(f"Training with gpu: {torch.cuda.is_available()}.")
        logging.info(f"Args: {self.args}")

        for epoch_counter in range(self.args.epochs):
            total_queries = 0
            for images, truelabels in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)
                query_features = self.victim_model(
                    images)  # victim model representations
                if self.loss != "symmetrized":
                    features = self.model(
                        images)  # current stolen model representation: 512x512 (512 images, 512/128 dimensional representation if head not used / if head used)
                if self.loss == "softce":
                    loss = self.criterion(features, F.softmax(features,
                                                              dim=1))  # F.softmax(query_features/self.args.temperature, dim=1))
                elif self.loss == "infonce":
                    all_features = torch.cat([features, query_features], dim=0)
                    logits, labels = self.info_nce_loss(all_features)
                    loss = self.criterion(logits, labels)
                elif self.loss == "bce":
                    loss = self.criterion(features, torch.round(torch.sigmoid(
                        query_features)))  # torch.round to convert it to one hot style representation
                elif self.loss == "softnn":
                    all_features = torch.cat([features, query_features], dim=0)
                    loss = self.criterion(self.args, all_features,
                                          pairwise_euclid_distance, self.tempsn)
                elif self.loss == "supcon":
                    all_features = torch.cat([F.normalize(features, dim=1),
                                              F.normalize(query_features,
                                                          dim=1)], dim=0)
                    labels = truelabels.repeat(
                        2)  # for victim and stolen features
                    bsz = labels.shape[0]
                    f1, f2 = torch.split(all_features, [bsz, bsz], dim=0)
                    all_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)],
                                             dim=1)
                    loss = self.criterion(all_features, labels)
                elif self.loss == "symmetrized":
                    # https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py#L294
                    # p is the output from the predictor (i.e. stolen model in this case)
                    # z is the output from the victim model (so the direct representation)
                    # when initializing, we need to include head for stolen, not for victim and set out_dim = 512
                    # first half of images includes all images under the first augmentation, second half includes under the second augmentation
                    x1 = images[:int(len(images) / 2)]
                    x2 = images[int(len(images) / 2):]
                    # p1 =  self.model(x1)
                    # p2 = self.model(x2) # output from stolen model for each augmentation (including head)
                    p1, p2, _, _ = self.model(x1, x2)
                    y1 = self.victim_model(x1).detach()
                    y2 = self.victim_model(
                        x2).detach()  # raw representations from victim
                    z1 = self.model.encoder.fc(y1)
                    z2 = self.model.encoder.fc(
                        y2)  # pass representations through attacker's encoder. This gives a better performance.
                    loss = -(self.criterion(p1, z2).mean() + self.criterion(p2,
                                                                            z1).mean()) * 0.5
                    # loss = neg_cosine(p1, z2)/2 + neg_cosine(p2, z1)/2 # same as above
                    # loss = (regression_loss(p1, z2) + regression_loss(p2, z1)).mean()
                elif self.loss == "barlow":
                    x1 = images[:int(len(images) / 2)]
                    x2 = images[int(len(images) / 2):]
                    p1 = self.model(x1)
                    p2 = self.model(x2)
                    y1 = self.victim_model(x1).detach()
                    y2 = self.victim_model(x2).detach()
                    P1 = torch.cat([p1, y1],
                                   dim=0)  # combine all representations on the first view
                    P2 = torch.cat([p2, y2],
                                   dim=0)  # combine all representations on the second view
                    loss = self.criterion(P1, P2, self.args.device)
                else:
                    loss = self.criterion(features, query_features)
                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                n_iter += 1
                total_queries += len(images)
                if total_queries >= num_queries:
                    break

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            logging.debug(
                f"Epoch: {epoch_counter}\tLoss: {loss}\t")

        logging.info("Stealing has finished.")
        # save model checkpoints
        checkpoint_name = f'stolen_checkpoint_{self.args.num_queries}_{self.loss}_{self.args.datasetsteal}.pth.tar'
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False,
            filename=os.path.join(self.log_dir2, checkpoint_name))
        logging.info(
            f"Stolen model checkpoint and metadata has been saved at {self.log_dir2}.")
