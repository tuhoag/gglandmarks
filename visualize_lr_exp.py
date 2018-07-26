import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from gglandmarks.experiments import min_images_and_classes_correlation as mincls

def visualize_acc():
    l0001path = './output/experiments/lr/my-resnets-lr=0.0001-cls=378-train-accuracy.csv'
    l001path = './output/experiments/lr/my-resnets-lr=0.001-cls=378-train-accuracy.csv'

    df0001 = pd.DataFrame.from_csv(l0001path)
    df001 = pd.DataFrame.from_csv(l001path)

    print(df0001.head())
    print(df001.head())

    fig, ax = plt.subplots()
    ax.plot(df0001['Step'], df0001['Value'], label='learning rate = 0.0001')
    ax.plot(df001['Step'], df001['Value'], label='learning rate = 0.001')
    ax.set(xlabel='Training Steps', ylabel='Training Accuracy')
    ax.legend()
    ax.grid()
    plt.savefig('./output/experiments/lr/acc.eps', format='eps', dpi=1000)
    plt.show()

def visualize_loss():
    l0001path = './output/experiments/lr/my-resnets-lr=0.0001-cls=378-train-loss.csv'
    l001path = './output/experiments/lr/my-resnets-lr=0.001-cls=378-train-loss.csv'

    df0001 = pd.DataFrame.from_csv(l0001path)
    df001 = pd.DataFrame.from_csv(l001path)

    print(df0001.head())
    print(df001.head())

    fig, ax = plt.subplots()
    ax.plot(df0001['Step'], df0001['Value'], label='learning rate = 0.0001')
    ax.plot(df001['Step'], df001['Value'], label='learning rate = 0.001')
    ax.set(xlabel='Training Steps', ylabel='Training Loss')
    ax.legend()
    ax.grid()
    plt.savefig('./output/experiments/lr/loss.eps', format='eps', dpi=1000)
    plt.show()

def visualize_network_acc():
    resnet_acc_path = './output/experiments/networks/resnets-eval-accuracy.csv'
    vgg_acc_path = './output/experiments/networks/vgg-eval-accuracy.csv'

    resnet_df = pd.DataFrame.from_csv(resnet_acc_path)
    vgg_df = pd.DataFrame.from_csv(vgg_acc_path)

    fig, ax = plt.subplots()
    ax.plot(resnet_df['Step'], resnet_df['Value'], label='ResNet-50')
    ax.plot(vgg_df['Step'], vgg_df['Value'], label='VGG-16')
    ax.grid()
    ax.legend()
    plt.savefig('./output/experiments/networks/acc.eps', format='eps', dpi=1000)
    plt.show()

# visualize_acc()
# visualize_loss()
# visualize_network_acc()
mincls.main()