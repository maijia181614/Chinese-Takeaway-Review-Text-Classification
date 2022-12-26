import time

import torch
from torch.utils.data import Dataset, DataLoader
import pandas
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
import numpy as np
import torch.nn as nn
import sklearn.metrics as metrics
from bertNet import BertClassfication
from tqdm import *
from charts import *
from bert_textCNN import TextCNN
from bert_textRNN import TextRNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resultPath = './result/'
chartsPath = './charts/'


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pandas.read_excel(data_path)

    def __getitem__(self, index):
       item = self.df.iloc[index, :]
       return item.values[0], item.values[1]

    def __len__(self):
        return self.df.shape[0]


def load_data(train_path, test_path, batch_size):
    train_data = pandas.read_excel(train_path)
    test_data = pandas.read_excel(test_path)
    print(train_data)
    print(test_data)
    mydataset = MyDataset(train_path)
    test_dataset = MyDataset(test_path)
    data_loader = DataLoader(dataset=mydataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    # for index, (label, sentenses) in enumerate(data_loader):
    #     print("标签：", label)
    #     print("句子：", sentenses)
    #     print("数据：", data_loader)
    #     print("长度：", len(data_loader))
    for index, (label, sentenses) in enumerate(test_data_loader):
        print("标签：", label)
        print("句子：", sentenses)
        print("数据：", data_loader)
        print("长度：", len(data_loader))
    return data_loader, test_data_loader

def train(train_data, test_data, trainModelIndex):
    # 选择模型
    trainModel = None
    lineName = None
    if (trainModelIndex == 0):
        trainModel = BertClassfication().to(device)
        lineName = "bert"
    if (trainModelIndex == 1):
        trainModel = TextCNN().to(device)
        lineName = "bert_textCNN"
    if (trainModelIndex == 2):
        trainModel = TextRNN().to(device)
        lineName = "bert_textRNN"
    print(lineName)

    pre_start = time.perf_counter()
    # 保存输入、输出标签列表
    output_label_list = []
    train_targets = []
    # 每xxx步训练后验证效果保存模型
    validateBatch = 150
    # 训练时统计的次数，用于判断验证时机
    totalBatchNum = 0
    # 当前评估最佳ACC
    bestAcc = 0
    # 制图数据
    # 评估数据步数
    collectDataBatch = 30
    totalAccList = []
    totalTestAccList = []
    xaxisBatchList = []
    totalLoss = 0
    totalLossList = []
    totalTestLossList = []
    totalF1scoreList = []
    totalTestF1scoreList = []
    totalAucscoreList = []
    totalTestAucscoreList = []

    epoch = 3

    # bert_model = BertClassfication().to(device)
    trainModel.train()
    # 损失函数与优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainModel.parameters(), lr=3e-5, eps=1e-8, weight_decay=1e-3)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_data),
                                                num_training_steps=epoch*len(train_data))
    process_bar = tqdm(total=epoch*len(train_data))
    for j in range(epoch):
        avg_loss = 0
        for index, (label, sentenses) in enumerate(train_data):
            inputs = sentenses
            targets = label.to(device)
            targets_numpy = targets.cpu().numpy().tolist()
            for target in targets_numpy:
                train_targets.append(target)
            optimizer.zero_grad()
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            outputs, __ = trainModel(inputs)
            print("outputs", outputs)
            outputs_label = torch.argmax(outputs, 1)
            outputs_label_numpy = outputs_label.cpu().numpy().tolist()
            for single in outputs_label_numpy:
                output_label_list.append(single)
            print("outputs_label", outputs_label)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            process_bar.update(1)


            avg_loss += loss.item()
            totalLoss += loss.item()
            print("当前loss:", avg_loss/(index+1))

            # 每xxx步训练后验证效果保存模型
            if totalBatchNum != 0 and totalBatchNum % validateBatch == 0:
                print("start test in training")
                testAcc, test_avg_loss, __, testAuc = evaluateBert(trainModel, test_data)
                if testAcc > bestAcc:
                    print("current best acc:" + str(bestAcc))
                    print("current test acc:" + str(testAcc))
                    bestAcc = testAcc
                    torch.save(trainModel, "./model/" + lineName + "_model_train.pth")
                trainModel.train()
            totalBatchNum += 1

            # 每xxx步计算各种统计值制图
            if totalBatchNum != 0 and totalBatchNum % collectDataBatch == 0:
                xaxisBatchList.append(str(totalBatchNum))
                # 训练统计
                tmpAcc = metrics.accuracy_score(train_targets, output_label_list)
                totalAccList.append(tmpAcc)

                tmpLoss = totalLoss / totalBatchNum
                totalLossList.append(tmpLoss)

                tmpF1 = metrics.f1_score(train_targets, output_label_list, average="binary")
                totalF1scoreList.append(tmpF1)

                tmpAuc = metrics.roc_auc_score(train_targets, output_label_list)
                totalAucscoreList.append(tmpAuc)

                # 测试统计
                testAccInTraining, test_avg_lossInTraining, testF1, testAuc = evaluateBert(trainModel, test_data)
                totalTestAccList.append(testAccInTraining)
                totalTestLossList.append(test_avg_lossInTraining)
                totalTestF1scoreList.append(testF1)
                totalTestAucscoreList.append(testAuc)
                trainModel.train()

    accLine(xaxisBatchList, totalAccList, totalTestAccList, chartsPath + lineName + "_acc")
    lossLine(xaxisBatchList, totalLossList, totalTestLossList, chartsPath + lineName + "_loss")
    f1Line(xaxisBatchList, totalF1scoreList, totalTestF1scoreList, chartsPath + lineName + "_f1")
    aucLine(xaxisBatchList, totalAucscoreList, totalTestAucscoreList, chartsPath + lineName + "_auc")

    pre_end = time.perf_counter()
    print('训练时间: %s Seconds' % (pre_end - pre_start))
    torch.save(trainModel, "./model/" + lineName + "_model_train.pth")

def evaluateBert(test_model, test_data):
    output_sentenses_list = []
    output_label_list = []
    test_targets = []
    test_model.eval()
    loss_function = nn.CrossEntropyLoss()
    avg_loss = 0
    test_index = 0
    # bert_model = BertClassfication().to(device)
    bert_model = test_model
    with torch.no_grad():
        for index, (label, sentenses) in enumerate(test_data):
            inputs = sentenses
            targets = label.to(device)
            targets_numpy = targets.cpu().numpy().tolist()
            for target in targets_numpy:
                test_targets.append(target)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            fc_output, outputs = bert_model(inputs)
            outputs_label = torch.argmax(fc_output, 1)
            for sentense in sentenses:
                output_sentenses_list.append(sentense)
            outputs_label_numpy = outputs_label.cpu().numpy().tolist()
            for single in outputs_label_numpy:
                output_label_list.append(single)
            loss = loss_function(fc_output, targets)
            avg_loss += loss.item()
            test_index = index

            print("当前loss:", avg_loss / (index + 1))
    output_dict = {"sentenses": output_sentenses_list,
                   "label": output_label_list}
    print("test_targets", test_targets)
    print("output_label_list", output_label_list)

    testReport(test_targets, output_label_list)
    testAcc = metrics.accuracy_score(test_targets, output_label_list)
    testF1 = metrics.f1_score(test_targets, output_label_list, average="binary")
    testAuc = metrics.roc_auc_score(test_targets, output_label_list)
    df = pandas.DataFrame(output_dict)
    df.to_excel(resultPath + "test_result.xlsx")
    return testAcc, avg_loss/test_index, testF1, testAuc

 # 打印分类报告
def testReport(test_targets, y_test_predict):
    from sklearn.metrics import classification_report
    test_targets = np.array(test_targets)
    print(classification_report(test_targets, y_test_predict, digits=4))

train_data, test_data = load_data("./dataset/waimai.xlsx", "./dataset/waimai_test.xlsx", 12)
# 0是bert 1是bert_textCNN 2是bert_textRNN
import sys
args = sys.argv
modelIndex = args[1]
train(train_data, test_data, int(modelIndex))
