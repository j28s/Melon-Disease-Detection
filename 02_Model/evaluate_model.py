import pandas as pd
import os
import re
import tqdm
from sklearn.metrics import accuracy_score

from models import CNNModel, VITModel
from dataset import make_loader
from configures import CFG

import torch
import torch.nn as nn
import torch.optim as optim

from multiprocessing import Manager, freeze_support

def eval_fn(data_loader, model, criterion, epoch_loss = 0.0):
    model.eval().to(CFG['DEVICE'])
    criterion.eval()

    preds = []
    actuals = []

    with torch.no_grad():
        for i_batch, item in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            images = item['image'].to(CFG['DEVICE'])
            labels = item['label'].to(CFG['DEVICE'])

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            val_loss = criterion(outputs, labels)

            epoch_loss += val_loss.item()

            preds.extend(predicted.tolist())  # Add predicted values to the list
            actuals.extend(labels.tolist())  # Add actual values to the list

    return epoch_loss, preds, actuals


def run(models_list):

    criterion = nn.CrossEntropyLoss()

    epoch_list = []
    val_acc_list = []
    val_loss_list = []

    for model_name in models_list:
        model = VITModel(num_classes)
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name)))

        epoch = re.findall(r'\d+', model_name)

        # valid
        val_epoch_loss, val_preds, val_actuals = eval_fn(val_loader, model, criterion)


        val_acc = accuracy_score(val_actuals, val_preds)
        val_loss = val_epoch_loss / len(val_loader)

        epoch_list.append(epoch)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

    return epoch_list, val_acc_list, val_loss_list


# manager = Manager()
# img_cache = manager.dict()

val = pd.read_csv('../Output/validation_data.csv')

num_classes = 4

val_loader = make_loader(val, batch_size=CFG['BATCH_SIZE'], shuffle=False)

models_dir = r"C:\병해분류\disease-aug"
models_list = os.listdir(models_dir)

epoch_list, val_acc_list, val_loss_list = run(models_list)
df = pd.DataFrame({
    'Epoch': epoch_list,
    'Accuracy': val_acc_list,
    'Loss': val_loss_list,
})

df.to_csv("../Output/vit_results.csv")