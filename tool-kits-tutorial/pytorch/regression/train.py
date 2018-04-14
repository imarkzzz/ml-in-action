import torch
import matplotlib.pyplot as plt
from brain import Net
from env import x, y

def train():
    net = Net(n_feature=1, n_hidden=10, n_output=1)
    # optimizer 是训练的工具
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

    plt.ion() # 画图
    plt.show()

    for t in range(100):
        prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值

        loss = loss_func(prediction, y)     # 计算两者的误差

        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss.backward()         # 误差反向传播, 计算参数更新值
        optimizer.step()        # 将参数更新值施加到 net 的 parameters 上

        # 可视化，采样显示减少训练时间
        if t % 5 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
            plt.pause(0.1)
            
def main():
    train()

if __name__ == '__main__':
    main()