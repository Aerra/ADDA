import os
import time
import json
import argparse
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import bernoulli
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
import torchvision.transforms as transforms

import config
from models import EncoderNet, ClassifierNet, AddaDiscriminator
from dataset_access import DatasetAccess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_time = int(time.time())

def loop_iterable(iterable):
    while True:
        yield from iterable

def do_test(encoder, classifier, dataloader):
    encoder.eval()
    classifier.eval()
    total_accuracy = 0

    logging.info("Testing ready model")
    with torch.no_grad():
        for x, y_true in dataloader:
            x, y_true = x.to(device), y_true.to(device)
            y_pred = classifier(encoder(x))
            total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    
    mean_accuracy = total_accuracy / len(dataloader)
    return mean_accuracy

def random_transform(img):
    tr = random.randint(0, 3)
    # random rotation
    if tr == 0:
        return transforms.RandomRotation(random.randint(0, 180))(img)
    # salt&pepper
    elif tr == 1:
        ch, row, col = img.shape
        number_of_pixels = random.randint(300, row * col)
        for _ in range(number_of_pixels):
            x = random.randint(0, row - 1)
            y = random.randint(0, col - 1)
            for c in range(ch):
                img[c][x][y] = 255
        number_of_pixels = random.randint(300, row * col)
        for _ in range(number_of_pixels):
            x = random.randint(0, row - 1)
            y = random.randint(0, col - 1)
            for c in range(ch):
                img[c][x][y] = 0
        return img
    # horizontal flip
    elif tr == 2:
        return transforms.RandomHorizontalFlip(1)(img)
    # add gauss noise
    else:
        ch, row, col = img.shape
        gauss = np.random.normal(0,0.1**0.5,(ch,row,col))
        gauss = gauss.reshape(ch,row,col)
        img = img + gauss
        return img

def do_epoch(encoder_model, classifier_model, dataloader, criterion, batch_size, optim=None, train = True):
    total_loss = 0
    total_accuracy = 0

    px = config.px
    py = config.py
    if not train:
        px = 0
        py = 0
    # create bernoully distributions for img noise and y value
    x_bern = bernoulli.rvs(size=len(dataloader)*batch_size, p = px)
    y_bern = bernoulli.rvs(size=len(dataloader)*batch_size, p = py)

    bern_shift = 0
    for x, y_true in dataloader:
        # add some noise and random values for y
        if train == True and px != 0 and py != 0:
            for i in range(x.shape[0]):
                idx = i + bern_shift
                if y_bern[idx] == 1:
                    y_true[i] = random.randrange(9)
                if x_bern[idx] == 1:
                    x[i] = random_transform(x[i])
            bern_shift += batch_size

        x, y_true = x.to(device), y_true.to(device)
        y_pred = classifier_model(encoder_model(x))

        loss = criterion(y_pred, y_true)

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy

def train_source_model(source, encoder_model, classifier_model, batch_size, epochs, output_dir):
    # load dataset 1
    dataset = DatasetAccess(source)
    df = dataset.get_access()
    shuffled_indices = np.random.permutation(len(df))
    train_idx = shuffled_indices[:int(0.8*len(df))]
    val_idx = shuffled_indices[int(0.8*len(df)):]
    train_loader = DataLoader(df, batch_size=batch_size, drop_last=True,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(df, batch_size=batch_size, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=1, pin_memory=True)
    # import file where parse conf
    # train if model didn't create
    # check if model already exist
    # train model from source
    optim = torch.optim.Adam(list(encoder_model.parameters())+list(classifier_model.parameters()))
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
    criterion = torch.nn.CrossEntropyLoss() 
    best_accuracy = 0
    best_encoder_model = encoder_model
    best_classifier_model = classifier_model

    for epoch in range(1, epochs+1):
        # train
        encoder_model.train()
        classifier_model.train()
        train_loss, train_accuracy = do_epoch(encoder_model, classifier_model, train_loader, criterion, batch_size, optim=optim)
        # eval
        encoder_model.eval() 
        classifier_model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(encoder_model, classifier_model, val_loader, criterion, batch_size, optim=None, train=False)
        logging.info(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
               f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')
        if val_accuracy > best_accuracy:
            logging.info('Saving model...')
            best_accuracy = val_accuracy
            best_encoder_model = encoder_model
            best_classifier_model = classifier_model
            torch.save(encoder_model.state_dict(), output_dir+f"/source_encoder_{current_time}.pt")
            torch.save(classifier_model.state_dict(), output_dir+f"/source_classifier_{current_time}.pt")
        lr_schedule.step(val_loss)
    
    return best_encoder_model, best_classifier_model

def adda_train(source_encoder, target_encoder, source, target, batch_size, epochs, iterations, plot=False):
    # load dataset for adda train
    source_dataset = DatasetAccess(source)
    source_df = source_dataset.get_access()

    target_dataset = DatasetAccess(target, train=False)
    target_df = target_dataset.get_access()

    source_loader = DataLoader(source_df, batch_size=batch_size//2, shuffle=True, 
                              num_workers=1, pin_memory=True)
    target_loader = DataLoader(target_df, batch_size=batch_size//2, shuffle=True,
                            num_workers=1, pin_memory=True)
    
    source_encoder.eval()

    # load model
    target_encoder.load_state_dict(source_encoder.state_dict())
    discriminator = AddaDiscriminator().to(device)

    target_encoder.train()
    discriminator.train()
 
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=config.lr_discriminator, betas=(0.5,0.9))
    target_optim = torch.optim.Adam(target_encoder.parameters(), lr=config.lr_target_encoder, betas=(0.5,0.9))
    criterion = nn.CrossEntropyLoss()

    accuracy_stats = []
    loss_stats = []

    # adda train
    for epoch in range(1, epochs+1):
        batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))

        total_loss = 0
        total_accuracy = 0

        for i in range(iterations):
            (source_x, _), (target_x, _) = next(batch_iterator)
            source_x, target_x = source_x.to(device), target_x.to(device)

            source_features = source_encoder(source_x).view(source_x.shape[0], -1)
            target_features = target_encoder(target_x).view(target_x.shape[0], -1)

            # train discriminator
            discriminator_x = torch.cat([source_features, target_features])
            discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                             torch.zeros(target_x.shape[0], device=device)])

            preds = discriminator(discriminator_x).squeeze()

            loss = criterion(preds, discriminator_y.long())

            loss.backward()
            discriminator_optim.step()
            total_loss += loss.item()

            accuracy = ((preds.max(1)[1]).long() == discriminator_y.long()).float().mean().item()
            total_accuracy += accuracy

            # train target_encoder
            target_optim.zero_grad()
            discriminator_optim.zero_grad()
            target_features = target_encoder(target_x).view(target_x.shape[0], -1)
            
            discriminator_y = torch.ones(target_x.shape[0], device=device)

            preds = discriminator(target_features).squeeze()

            loss_new = criterion(preds, discriminator_y.long())

            loss_new.backward()
            target_optim.step()

        accuracy_stats.append(total_accuracy)
        loss_stats.append(total_loss)
        mean_loss = total_loss / (iterations)
        mean_accuracy = total_accuracy / (iterations)
        logging.info(f'EPOCH {epoch:03d}: discriminator_loss={mean_loss:.4f}, '
                   f'discriminator_accuracy={mean_accuracy:.4f}')

    if plot:
        logging.info("Plot graphics")
        train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
        sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", ax=axes[0]).set_title('Discriminator Accuracy/Epoch*Iterations')
        sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", ax=axes[1]).set_title('Discriminator Loss/Epoch*Iterations')
        discr_accuracy = open(output_dir+f"/discr_accuracy_{current_time}.txt", "w")
        for i in accuracy_stats:
            discr_accuracy.write(f"{i}\n")
        discr_accuracy.close()
        discr_loss = open(output_dir+f"/discr_loss_{current_time}.txt", "w")
        for i in loss_stats:
            discr_loss.write(f"{i}\n")
        discr_loss.close()

    # Save it
    torch.save(target_encoder.state_dict(), output_dir+f"/adda_encoder_{current_time}.pt")
    torch.save(discriminator.state_dict(), output_dir+f"/adda_discriminator_{current_time}.pt")
    return target_encoder


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on MNIST')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=30)
    arg_parser.add_argument('--dataconf', type=str)
    arg_parser.add_argument('--adda_iterations', type=int, default=500)
    arg_parser.add_argument('--source_enc_model', type=str, required=False)
    arg_parser.add_argument('--source_clf_model', type=str, required=False)
    arg_parser.add_argument('--adda_epochs', type=int, default=15)
    arg_parser.add_argument('--plot', type=bool, default=False)

    format_str = "%(asctime)s %(name)s [%(levelname)s] - %(message)s"
    logging.basicConfig(format=format_str)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    args = arg_parser.parse_args()

    logging.info(f"Running with {args}")

    # init seed
    seed = random.randint(1, 10000)
    logging.info("Use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # check config path
    data_json_path = str(config.DATA_DIR) + f"/data_args/{args.dataconf}.json"
    if not os.path.exists(data_json_path):
        raise ValueError(f"{data_json_path} doesn't exist")
    
    # open configuration json file
    dataconf = open(data_json_path, "r")
    dataconf = json.load(dataconf)

    output_dir = str(config.DATA_DIR) + f"/trained_models/{dataconf['dataset_name']}"
    logging.info(f"Create output dir {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load models
    h = config.img_h
    w = config.img_w
    kernel_size_1 = config.kernel_size_1
    kernel_size_2 = config.kernel_size_2
    out_ch_1 = config.out_ch_1
    out_ch_2 = config.out_ch_2

    source_encoder_model = EncoderNet(h, w, kernel_size_1, kernel_size_2, out_ch_1, out_ch_2).to(device)
    source_classifier_model = ClassifierNet().to(device)
    target_encoder_model = EncoderNet(h, w, kernel_size_1, kernel_size_2, out_ch_1, out_ch_2).to(device)

    if args.source_enc_model != None:
        logging.info(f"Load model from {args.source_enc_model}")
        source_encoder_model.load_state_dict(torch.load(args.source_enc_model))
        source_encoder = source_encoder_model
        logging.info(f"Load model from {args.source_clf_model}")
        source_classifier_model.load_state_dict(torch.load(args.source_clf_model))
        source_classifier = source_classifier_model
    elif os.path.exists(output_dir+f"/source_encoder_{current_time}.pt"):
        logging.info(f"Load model from {output_dir}/source_encoder_{current_time}.pt")        
        source_encoder_model.load_state_dict(torch.load(output_dir+f"/source_encoder_{current_time}.pt"))
        source_encoder = source_encoder_model
        logging.info(f"Load model from {output_dir}/source_classifier_{current_time}.pt")        
        source_classifier_model.load_state_dict(torch.load(output_dir+f"/source_classifier_{current_time}.pt"))
        source_classifier = source_classifier_model
    else:
        logging.info("Train new source model")
        source_encoder, source_classifier = train_source_model(dataconf["source"],
            source_encoder_model,
            source_classifier_model,
            args.batch_size,
            args.epochs,
            output_dir)

    # load dataset for adda train
    logging.info("Train ADDA classifier")
    target_encoder = adda_train(
        source_encoder,
        target_encoder_model,
        dataconf["source"],
        dataconf["target"],
        args.batch_size,
        args.adda_epochs,
        args.adda_iterations,
        args.plot)

    # test classifier
    target_dataset = DatasetAccess(dataconf["target"], train=False)
    target_df = target_dataset.get_access()
    target_loader = DataLoader(target_df, batch_size=1, shuffle=True,
                            num_workers=1, pin_memory=True)
    target_accuracy = do_test(target_encoder, source_classifier, target_loader)
    print(f'Accuracy on target data: {target_accuracy:.4f}')

    source_accuracy = do_test(source_encoder_model, source_classifier, target_loader)
    print(f'Accuracy on source only: {source_accuracy:.4f}')
