# goldens = [0, 0, 1, 1]
# predictions = [1, 1, 0, 1]
# a = sorted(list(set(goldens) | set(predictions)))
# print(a)
#
# from sklearn.metrics import confusion_matrix
# import numpy as np
#
# conf_matrix = np.array(confusion_matrix(goldens, predictions))  # (n_class, n_class)
# print(f'conf_matrix-->{conf_matrix}')
# assert conf_matrix.shape[0] == len(a)
# for i in range(conf_matrix.shape[0]):  # 构建每个class的指标
#     print(f'conf_matrix[i, i]-->{conf_matrix[i, i]}')
#     print(f'sum(conf_matrix[:, i])-->{sum(conf_matrix[:, i])}')
#
#     print(f'sum(conf_matrix[i, :])-->{sum(conf_matrix[i, :])}')
#     precision = 0 if sum(conf_matrix[:, i]) == 0 else conf_matrix[i, i] / sum(conf_matrix[:, i])
#     print(f'precision--->{precision}')
#     recall = 0 if sum(conf_matrix[i, :]) == 0 else conf_matrix[i, i] / sum(conf_matrix[i, :])
#     print(f'recall-->{recall}')
#     f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
#     print(f'f1--->{f1}')
#     print("*"*80)

from sklearn.metrics import precision_score
'''
y_true = [1, 0, 2, 2]
y_pred = [1, 1, 2, 2]

precision_score(y_true, y_pred, average="weighted") = 0.625
计算过程：
1、考虑y_pred里面对每个样本的预测的准确率，看到预测两个类别【1， 2】
预测类别1的准确率（预测1的总个数其中预测对的占比）：1/2 = 0.5
预测类别2的准确率（预测2的总个数其中预测对的占比）：2/2 = 1
2、计算average="weighted"的精确率，考虑真实样本的占比
类别1的precision_:1/4 * 0.5 = 0.25*0.5 = 0.125
类别2的precision_:2/4 * 1 = 0.5
3、整体指标：0.5+0.125 = 0.625
'''
y_true = [1, 0, 2, 2]
y_pred = [1, 1, 2, 2]
print(precision_score(y_true, y_pred, average="weighted"))  #考虑真实值中的类别占比，然后乘以每个类别预测正确的比例
# 'micro' 和(np.sum(y_true == y_pred) / len(y_predict))
print(precision_score(y_true, y_pred, average="micro"))  # 看预测对的标签总个数，再除以pred的大小。


# # a = [1, 2, 3]
# # b = [2, 3, 4]
# # # a.extend()
# import numpy as np
#
# from sklearn.metrics import confusion_matrix
# goldens = ['体育', '财经', '体育', '计算机', '计算机']
# predictions = ['财经', '财经', '体育', '体育', '计算机']
# # goldens = ['计算机', '体育', '计算机', '计算机']
# # predictions = ['计算机', '体育',  '体育', '计算机']
# conf_matrix = np.array(confusion_matrix(goldens, predictions))  # (n_class, n_class)
# print(conf_matrix)

# def fun(a, b):
#     print('a', a)
#     print('b', b)
#
# fun(**{"b": 10, "a": 2})