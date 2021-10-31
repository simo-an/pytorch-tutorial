import datetime
import matplotlib.pyplot as plt
from numpy.lib.npyio import save

def get_format_time():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# 绘制不同Epoch的 损失、学习率曲线
def plot_loss_and_lr(train_loss, lr, save = True):
    x = list(range(len(train_loss)))
    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(x, train_loss, 'r', label='loss')
    ax1.set_xlabel('step')
    ax1.set_ylabel('loss')
    ax1.set_title('Train loss and lr')
    plt.legend(loc='best')

    ax2 = ax1.twinx()
    ax2.plot(x, lr, label='lr')
    ax2.set_ylabel('learning rate')
    ax2.set_xlim(0, len(train_loss)) # 设置横坐标整数间隔
    plt.legend(loc='best')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

    if save:
        fig.subplots_adjust(right=0.8) # 防止出现图片保存不全
        fig.savefig(f'./loss_and_lr_{get_format_time()}.png')
        plt.close()
        print('save loss and lr curve successfully!')
    else:
        plt.show()

# 绘制平均精度折线图
def plot_mAP(mAP, save=True):
    x = list(range(len(mAP)))
    plt.plot(x, mAP, label='mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP Eval Graph')
    plt.xlim(0, len(mAP))
    plt.legend(loc='best')
    if save:
        plt.savefig(f'./mAP_{get_format_time()}.png')
        plt.close()
        print('save mAP curve successfully!')
    else:
        plt.show()