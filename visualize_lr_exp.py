import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

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

visualize_acc()
visualize_loss()
