import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
import numpy as np
import argparse
import math


# import atplotlib.patches as rect


def parse_ls_log_file(logfilename):
    cc = 0
    acc1 = np.zeros(90)
    acc5 = np.zeros(90)
    loss = np.zeros(90)
    # file = '/Users/yixu/PycharmProjects/StocOpt/venv/nips-ls-experiment/label_smoothing_drop/cifar100/'
    # file = '/Users/yixu/PycharmProjects/StocOpt/venv/nips-ls-experiment/cifar100/'
    file = '/Users/yixu/PycharmProjects/StocOpt/venv/nips-ls-experiment/dogs_birds_report_20200528/birds/'
    logfilename = file + logfilename
    for line in open(logfilename):
        tmp = line.split(' ')
        acc1[cc] = float(tmp[0])
        acc5[cc] = float(tmp[1])
        loss[cc] = float(tmp[2])
        cc += 1
    return acc1, acc5, loss


n = 90
lw = '3'
epoch = np.arange(0, n) + 1
fig = plt.figure()

# acc11, acc51, loss1 = parse_ls_log_file('round_0.test.ls_0.4_drop_20.log')
# acc12, acc52, loss2 = parse_ls_log_file('round_1.test.ls_0.4_drop_20.log')
# acc13, acc53, loss3 = parse_ls_log_file('round_2.test.ls_0.4_drop_20.log')
# acc14, acc54, loss4 = parse_ls_log_file('round_3.test.ls_0.4_drop_20.log')
# acc15, acc55, loss5 = parse_ls_log_file('round_4.test.ls_0.4_drop_20.log')
# acc1 = (acc11 + acc12 + acc13 + acc14 + acc15)/5
# acc5 = (acc51 + acc52 + acc53 + acc54 + acc55)/5
# loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5
# plt.plot(epoch, acc1, linewidth=lw, label="TSLA (20)") #, color = '#1f77b4')


acc11, acc51, loss1 = parse_ls_log_file('round_0.test.ls_0.4_drop_30.log')
acc12, acc52, loss2 = parse_ls_log_file('round_1.test.ls_0.4_drop_30.log')
acc13, acc53, loss3 = parse_ls_log_file('round_2.test.ls_0.4_drop_30.log')
acc14, acc54, loss4 = parse_ls_log_file('round_3.test.ls_0.4_drop_30.log')
acc15, acc55, loss5 = parse_ls_log_file('round_4.test.ls_0.4_drop_30.log')
acc1 = (acc11 + acc12 + acc13 + acc14 + acc15) / 5
acc5 = (acc51 + acc52 + acc53 + acc54 + acc55) / 5
loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
plt.plot(epoch, acc1, linewidth=lw, label="TSLA (30)")  # , color = '#1f77b4')

acc11, acc51, loss1 = parse_ls_log_file('round_0.test.ls_0.4_drop_40.log')
acc12, acc52, loss2 = parse_ls_log_file('round_1.test.ls_0.4_drop_40.log')
acc13, acc53, loss3 = parse_ls_log_file('round_2.test.ls_0.4_drop_40.log')
acc14, acc54, loss4 = parse_ls_log_file('round_3.test.ls_0.4_drop_40.log')
acc15, acc55, loss5 = parse_ls_log_file('round_4.test.ls_0.4_drop_40.log')
acc1 = (acc11 + acc12 + acc13 + acc14 + acc15) / 5
acc5 = (acc51 + acc52 + acc53 + acc54 + acc55) / 5
loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
plt.plot(epoch, acc1, linewidth=lw, label="TSLA (40)")  # , color = '#1f77b4')

acc11, acc51, loss1 = parse_ls_log_file('round_0.test.ls_0.4_drop_50.log')
acc12, acc52, loss2 = parse_ls_log_file('round_1.test.ls_0.4_drop_50.log')
acc13, acc53, loss3 = parse_ls_log_file('round_2.test.ls_0.4_drop_50.log')
acc14, acc54, loss4 = parse_ls_log_file('round_3.test.ls_0.4_drop_50.log')
acc15, acc55, loss5 = parse_ls_log_file('round_4.test.ls_0.4_drop_50.log')
acc1 = (acc11 + acc12 + acc13 + acc14 + acc15) / 5
acc5 = (acc51 + acc52 + acc53 + acc54 + acc55) / 5
loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
plt.plot(epoch, acc1, linewidth=lw, label="TSLA (50)")  # , color = '#1f77b4')

acc11, acc51, loss1 = parse_ls_log_file('round_0.test.ls_0.4_drop_60.log')
acc12, acc52, loss2 = parse_ls_log_file('round_1.test.ls_0.4_drop_60.log')
acc13, acc53, loss3 = parse_ls_log_file('round_2.test.ls_0.4_drop_60.log')
acc14, acc54, loss4 = parse_ls_log_file('round_3.test.ls_0.4_drop_60.log')
acc15, acc55, loss5 = parse_ls_log_file('round_4.test.ls_0.4_drop_60.log')
acc1 = (acc11 + acc12 + acc13 + acc14 + acc15) / 5
acc5 = (acc51 + acc52 + acc53 + acc54 + acc55) / 5
loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
plt.plot(epoch, acc1, linewidth=lw, label="TSLA (60)")  # , color = '#1f77b4')

acc11, acc51, loss1 = parse_ls_log_file('round_0.test.ls_0.4_drop_70.log')
acc12, acc52, loss2 = parse_ls_log_file('round_1.test.ls_0.4_drop_70.log')
acc13, acc53, loss3 = parse_ls_log_file('round_2.test.ls_0.4_drop_70.log')
acc14, acc54, loss4 = parse_ls_log_file('round_3.test.ls_0.4_drop_70.log')
acc15, acc55, loss5 = parse_ls_log_file('round_4.test.ls_0.4_drop_70.log')
acc1 = (acc11 + acc12 + acc13 + acc14 + acc15) / 5
acc5 = (acc51 + acc52 + acc53 + acc54 + acc55) / 5
loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
plt.plot(epoch, acc1, linewidth=lw, label="TSLA (70)")  # , color = '#1f77b4')

acc11, acc51, loss1 = parse_ls_log_file('round_0.test.ls_0.4_drop_80.log')
acc12, acc52, loss2 = parse_ls_log_file('round_1.test.ls_0.4_drop_80.log')
acc13, acc53, loss3 = parse_ls_log_file('round_2.test.ls_0.4_drop_80.log')
acc14, acc54, loss4 = parse_ls_log_file('round_3.test.ls_0.4_drop_80.log')
acc15, acc55, loss5 = parse_ls_log_file('round_4.test.ls_0.4_drop_80.log')
acc1 = (acc11 + acc12 + acc13 + acc14 + acc15) / 5
acc5 = (acc51 + acc52 + acc53 + acc54 + acc55) / 5
loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
plt.plot(epoch, acc1, linewidth=lw, label="TSLA (80)")  # , color = '#1f77b4')

acc11, acc51, loss1 = parse_ls_log_file('round_0.test.ls_0.4.log')
acc12, acc52, loss2 = parse_ls_log_file('round_1.test.ls_0.4.log')
acc13, acc53, loss3 = parse_ls_log_file('round_2.test.ls_0.4.log')
acc14, acc54, loss4 = parse_ls_log_file('round_3.test.ls_0.4.log')
acc15, acc55, loss5 = parse_ls_log_file('round_4.test.ls_0.4.log')
acc1 = (acc11 + acc12 + acc13 + acc14 + acc15) / 5
acc5 = (acc51 + acc52 + acc53 + acc54 + acc55) / 5
loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
plt.plot(epoch, acc1, '--', linewidth=lw, label="LSR")

acc11, acc51, loss1 = parse_ls_log_file('round_0.test.baseline.log')
acc12, acc52, loss2 = parse_ls_log_file('round_1.test.baseline.log')
acc13, acc53, loss3 = parse_ls_log_file('round_2.test.baseline.log')
acc14, acc54, loss4 = parse_ls_log_file('round_3.test.baseline.log')
acc15, acc55, loss5 = parse_ls_log_file('round_4.test.baseline.log')
acc1 = (acc11 + acc12 + acc13 + acc14 + acc15) / 5
acc5 = (acc51 + acc52 + acc53 + acc54 + acc55) / 5
loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
plt.plot(epoch, acc1, ':', linewidth=lw, label="baseline")

plt.legend(fontsize=12, loc='lower right', ncol=2)
plt.xlim([20, 90])
plt.ylim([73.6, 77.6])
plt.yticks(range(74, 78))
# plt.xlim([10,90])
# plt.ylim([70,77.6])
# plt.yticks(range(74,78))
fig.suptitle("CUB-2011", fontsize=15)
plt.xlabel("# epoch", fontsize=15)
plt.ylabel("Top-1 Accuracy", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
