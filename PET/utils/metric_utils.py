# coding='utf-8'
# （多）分类问题下的指标评估（acc, precision, recall, f1）。
from typing import List
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix


class ClassEvaluator(object):

	def __init__(self):
		# 初始化真实结果和预测结果的列表
		self.goldens = []  # 存储真实结果数据
		self.predictions = []  # 存储预测结果数据

	def add_batch(self,
				  pred_batch: List[List],
				  gold_batch: List[List]):
		"""
		添加一个batch中的prediction和gold列表，用于后续统一计算。

		Args:
			pred_batch (list): 模型预测标签列表, e.g. ->  [['体', '育'], ['财', '经'], ...]
			gold_batch (list): 真实标签标签列表, e.g. ->  [['体', '育'], ['财', '经'], ...]
		"""
		# 确保预测批次和真实批次长度一致，这是后续处理的前提条件
		assert len(pred_batch) == len(gold_batch)
		# print(f'pred_batch0--》{pred_batch}')
		# print(f'gold_batch0--》{gold_batch}')
		# 若遇到多个子标签构成一个标签的情况
		# 判断gold_batch的第一个元素是否为列表或元组类型
		if type(gold_batch[0]) in [list, tuple]:
			# 如果是，则将pred_batch中的每个元素转换为字符串后拼接起来
			pred_batch = [''.join([str(e) for e in ele]) for ele in pred_batch]
			# 同样地，也将gold_batch中的每个元素转换为字符串后拼接起来
			gold_batch = [''.join([str(e) for e in ele]) for ele in gold_batch]
		# print(f'pred_batch1--》{pred_batch}')
		# print(f'gold_batch1--》{gold_batch}')
		# 将真实结果的批次数据添加到self.goldens列表中
		self.goldens.extend(gold_batch)
		# print(f'self.goldens--》{self.goldens}')
		# 将预测结果的批次数据添加到self.predictions列表中
		self.predictions.extend(pred_batch)

	# print(f'self.predictions--->{ self.predictions}')

	def compute(self,
				round_num=2) -> dict:
		"""
		根据当前类中累积的变量值，计算当前的P, R, F1。

		Args:
			round_num (int): 计算结果保留小数点后几位, 默认小数点后2位。

		Returns:
			dict -> {
				'accuracy': 准确率,
				'precision': 精准率,
				'recall': 召回率,
				'f1': f1值,
				'class_metrics': {
					'0': {
							'precision': 该类别下的precision,
							'recall': 该类别下的recall,
							'f1': 该类别下的f1
						},
					...
				}
			}
		"""
		# print(f'self.goldens--》{self.goldens}')
		# print(f'self.predictions--》{self.predictions}')
		# 初始化类别集合、类别指标字典和结果字典，用于存储全局指标
		# 将 self.goldens 和 self.predictions 的集合合并，并进行排序，结果存储在变量 classes 中。
		classes, class_metrics, res = sorted(list(set(self.goldens) | set(self.predictions))), {}, {}
		# print(f'classes-->{classes}')

		# 构建全局指标
		# 计算并存储全局准确率
		res['accuracy'] = round(accuracy_score(self.goldens, self.predictions), round_num)
		# 计算并存储全局精确率
		res['precision'] = round(precision_score(self.goldens, self.predictions, average='weighted'), round_num)
		# 计算并存储全局召回率
		res['recall'] = round(recall_score(self.goldens, self.predictions, average='weighted'), round_num)
		# 计算并存储全局F1分数
		res['f1'] = round(f1_score(self.goldens, self.predictions, average='weighted'), round_num)
		# print(f'res-->{res}')
		# print(f'classes--》{classes}')
		# print(f'self.goldens--》{self.goldens}')
		# print(f'self.predictions--》{self.predictions}')

		try:
			# 计算混淆矩阵，并将其转换为numpy数组，形状为(n_class, n_class)
			conf_matrix = np.array(confusion_matrix(self.goldens, self.predictions))
			# print(f'conf_matrix-->{conf_matrix}')
			# 确保混淆矩阵的维度与类别数量匹配
			assert conf_matrix.shape[0] == len(classes)
			# 遍历每个类别，计算精确度(precision)、召回率(recall)和F1分数(f1)
			for i in range(conf_matrix.shape[0]):
				# 计算当前类别的精确度
				precision = 0 if sum(conf_matrix[:, i]) == 0 else (conf_matrix[i, i] / sum(conf_matrix[:, i]))
				# 计算当前类别的召回率
				recall = 0 if sum(conf_matrix[i, :]) == 0 else (conf_matrix[i, i] / sum(conf_matrix[i, :]))
				# 计算当前类别的F1分数
				f1 = 0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
				# 将当前类别的精确度、召回率和F1分数保存到字典中
				class_metrics[classes[i]] = {
					'precision': round(precision, round_num),
					'recall': round(recall, round_num),
					'f1': round(f1, round_num)
				}
			# 将所有类别的指标保存到结果字典中
			res['class_metrics'] = class_metrics
		except Exception as e:
			# 异常处理：当计算类别指标时发生异常，打印警告信息和相关数据
			print(f'[Warning] Something wrong when calculate class_metrics: {e}')
			print(f'-> goldens: {set(self.goldens)}')
			print(f'-> predictions: {set(self.predictions)}')
			print(f'-> diff elements: {set(self.predictions) - set(self.goldens)}')
			# 将结果字典中的类别指标设置为空字典
			res['class_metrics'] = {}

		return res

	def reset(self):
		"""
		重置积累的数值。
		"""
		self.goldens = []
		self.predictions = []


if __name__ == '__main__':
	from rich import print

	metric = ClassEvaluator()
	metric.add_batch(
		[['财', '经'], ['财', '经'], ['体', '育'], ['体', '育'], ['计', '算', '机']],
		[['体', '育'], ['财', '经'], ['体', '育'], ['计', '算', '机'], ['计', '算', '机']],
	)
	# metric.add_batch(
	#     [0, 0, 1, 1, 0],
	#     [1, 1, 1, 0, 0]
	# )
	res = metric.compute()
	print(res)
