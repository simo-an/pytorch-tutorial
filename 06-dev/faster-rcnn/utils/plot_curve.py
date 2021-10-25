import datetime
import matplotlib.pyplot as plt

def plot_loss_and_lr(train_loss, lr):
    try:
        x = list(range(len(train_loss)))
    except Exception as e:
        print(e)

def plot_map(mAP):
    try:
        x = list(range(len(mAP)))
    except Exception as e:
        print(e)