# -*- coding:utf-8 -*-
import os
# Union 是 typing 模块中定义的一个类,用于表示多个类型中的任意一种类型
from typing import Union, List
from PET.pet_config import *

pc = ProjectConfig()


class Verbalizer(object):
	"""
	Verbalizer类，用于将一个Label对应到其子Label的映射。
	"""
	def __init__(self,
				 verbalizer_file: str,
				 tokenizer,
				 max_label_len: int
				 ):
		"""
		Args:
			verbalizer_file (str): verbalizer文件存放地址。
			tokenizer: 用于文本和id之间的转换。
			max_label_len (int): 标签长度，若大于则截断，若小于则补齐
		"""
		self.tokenizer = tokenizer
		self.label_dict = self.load_label_dict(verbalizer_file)
		self.max_label_len = max_label_len

	def load_label_dict(self, verbalizer_file: str):
		"""
		读取本地文件，构建verbalizer字典。

		Args:
			verbalizer_file (str): verbalizer文件存放地址。

		Returns:
			dict -> {
				'体育': ['篮球', '足球','网球', '排球',  ...],
				'酒店': ['宾馆', '旅馆', '旅店', '酒店', ...],
				...
				}
		"""
		# 初始化一个空字典，用于存储标签和子标签的关系
		label_dict = {}

		# 打开verbalizer文件，以只读模式，使用utf-8编码
		with open(verbalizer_file, 'r', encoding='utf-8') as f:
			# 读取文件的每一行
			for line in f.readlines():
				# 移除行尾的换行符，并按制表符('\t')分割标签和子标签
				label, sub_labels = line.strip().split('\t')
				# 将子标签按逗号(,)分割成列表，使用set去重后再转回列表，存储到label_dict中
				label_dict[label] = list(set(sub_labels.split(',')))
		# 返回处理后的标签和子标签的字典
		return label_dict

	def find_sub_labels(self, label: Union[list, str]):
		"""
		通过主标签找到所有的子标签。

		Args:
			label (Union[list, str]): 标签, 文本型 或 id_list, e.g. -> '体育' or [860, 5509]

		Returns:
			dict -> {
				'sub_labels': ['足球', '网球'],
				'token_ids': [[6639, 4413], [5381, 4413]]
			}
		"""
		# 如果传入的label为id列表，则通过tokenizer转换回字符串
		if type(label) == list:
			# 移除label中的pad_token_id，直到label中不再包含它
			while self.tokenizer.pad_token_id in label:
				label.remove(self.tokenizer.pad_token_id)
			# 将处理后的id列表转换为tokens，并拼接成字符串
			label = ''.join(self.tokenizer.convert_ids_to_tokens(label))
		# print(f'label-->{label}')
		# 检查转换后的label是否在标签字典中，如果不在则抛出异常
		if label not in self.label_dict:
			raise ValueError(f'Lable Error: "{label}" not in label_dict {list(self.label_dict)}.')

		# 从标签字典中获取与label对应的子标签
		sub_labels = self.label_dict[label]
		# print(f'sub_labels-->{sub_labels}')
		# 将子标签作为结果的一个部分存储在字典中
		ret = {'sub_labels': sub_labels}
		# 对每个子标签进行token化，并去除首尾token（通常为特殊符号）
		# print(f"self.tokenizer(sub_labels)-->{self.tokenizer(sub_labels)}")
		token_ids = [_id[1:-1] for _id in self.tokenizer(sub_labels)['input_ids']]
		# print(f'token_ids-->{token_ids}')
		# 遍历所有的token_ids，进行截断与补齐操作
		for i in range(len(token_ids)):
			# 如果长度>=max_label_len，对标签进行截断
			if len(token_ids[i]) >= self.max_label_len:
				token_ids[i] = token_ids[i][:self.max_label_len]
			# 如果长度<max_label_len，则使用pad_token_id进行补齐
			else:
				token_ids[i] = token_ids[i] + [self.tokenizer.pad_token_id] * (self.max_label_len - len(token_ids[i]))
		# 将处理后的token_ids存入ret字典中
		ret['token_ids'] = token_ids
		return ret

	def batch_find_sub_labels(self, label: List[Union[list, str]]):
		"""
		批量找到子标签。

		Args:
			label (List[list, str]): 标签列表, [[4510, 5554], [860, 5509]] or ['体育', '电脑']

		Returns:
			list -> [
						{
							'sub_labels': ['笔记本', '电脑'],
							'token_ids': [[5011, 6381, 3315], [4510, 5554]]
						},
						...
					]
		"""
		return [self.find_sub_labels(l) for l in label]

	def get_common_sub_str(self,
						   str1: str,
						   str2: str
						   ):
		"""
		寻找最大公共子串(连续子序列)。
		str1:abcd
		str2:abadbcdba
		"""
		# 初始化两个字符串的长度
		lstr1, lstr2 = len(str1), len(str2)
		# 生成0矩阵，为方便后续计算，比字符串长度多了一列
		record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]
		# 初始化最长匹配对应在str1中的最后一位
		p = 0
		# 初始化最长匹配长度
		maxNum = 0
		# 遍历两个字符串，寻找最长公共子串
		for i in range(1, lstr1 + 1):
			for j in range(1, lstr2 + 1):
				# 当发现相同字符时
				if str1[i - 1] == str2[j - 1]:
					# 在record矩阵中记录匹配长度
					record[i][j] = record[i - 1][j - 1] + 1
					# 更新最长匹配长度和对应在str1中的最后一位
					if record[i][j] > maxNum:
						maxNum = record[i][j]
						p = i

		# 返回最长公共子串和其长度
		return str1[p - maxNum:p], maxNum

	def hard_mapping(self, sub_label: str):
		"""
		强匹配函数，当模型生成的子label不存在时，通过最大公共子串找到重合度最高的主label。

		Args:
			sub_label (str): 子label。

		Returns:
			str: 主label。
		"""
		# 初始化变量label和max_overlap_str，用于记录最大重叠度的标签和对应的重叠度值
		label, max_overlap_str = '', 0

		# 遍历标签字典，其中main_label是主标签，sub_labels是与主标签相关的子标签列表
		for main_label, sub_labels in self.label_dict.items():
			overlap_num = 0
			# 对于每个子标签，计算它与当前推理标签之间的最长公共子串长度总和
			for s_label in sub_labels:
				# 累加当前主标签下每个子标签与当前推理标签之间的最长公共子串长度
				overlap_num += self.get_common_sub_str(sub_label, s_label)[1]

			# 如果当前的重叠度大于或等于之前的最大重叠度，则更新最大重叠度和对应的主标签, 否则不更新(用上一次主标签)
			if overlap_num >= max_overlap_str:
				max_overlap_str = overlap_num
				label = main_label

		return label

	def find_main_label(self,
						sub_label: Union[list, str],
						hard_mapping=True
						):
		"""
		通过子标签找到父标签。

		Args:
			sub_label (Union[list, str]): 子标签, 文本型 或 id_list, e.g. -> '苹果' or [5741, 3362]
			hard_mapping (bool): 当生成的词语不存在时，是否一定要匹配到一个最相似的label。

		Returns:
			dict -> {
				'label': '水果',
				'token_ids': [3717, 3362]
			}
		"""
		# 如果传入的sub_label为id列表，则通过tokenizer转换回字符串
		if type(sub_label) == list:
			pad_token_id = self.tokenizer.pad_token_id
			# 移除列表中的[PAD]token，避免影响后续处理
			while pad_token_id in sub_label:
				sub_label.remove(pad_token_id)
			# 将id列表转换为对应的字符串
			sub_label = ''.join(self.tokenizer.convert_ids_to_tokens(sub_label))
		# print(f'sub_label-->{sub_label}')
		# 初始化主标签为'无'，作为未找到特定子标签时的默认值
		main_label = '无'

		# 遍历标签字典，寻找与子标签匹配的主标签
		for label, sub_labels in self.label_dict.items():
			# 检查当前子标签是否在字典中对应的子标签列表中
			if sub_label in sub_labels:
				# 当找到匹配时，更新主标签并终止循环
				main_label = label
				break
		# print(f'main_label--》{main_label}')
		# 如果主标签为'无'且启用了强匹配功能，则使用强匹配方法更新主标签
		if main_label == '无' and hard_mapping:
			main_label = self.hard_mapping(sub_label)
		# print('强匹配', main_label)
		ret = {
			'label': main_label,
			'token_ids': self.tokenizer(main_label)['input_ids'][1:-1]
		}
		return ret

	def batch_find_main_label(self,
							  sub_label: List[Union[list, str]],
							  hard_mapping=True
							  ):
		"""
		批量通过子标签找父标签。

		Args:
			sub_label (List[Union[list, str]]): 子标签列表, ['苹果', ...] or [[5741, 3362], ...]

		Returns:
			list: [
					{
					'label': '水果',
					'token_ids': [3717, 3362]
					},
					...
			]
		"""
		return [self.find_main_label(l, hard_mapping) for l in sub_label]


if __name__ == '__main__':
	from rich import print
	from transformers import AutoTokenizer

	tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
	verbalizer = Verbalizer(
		verbalizer_file=pc.verbalizer,
		tokenizer=tokenizer,
		max_label_len=2)
	# 查找单个子标签
	# print('*' * 80)
	# print(f'label_dict--->{verbalizer.label_dict}')
	# label = [4510, 5554]
	# print(tokenizer.convert_ids_to_tokens(label))
	# label = '电脑'
	# ret = verbalizer.find_sub_labels(label)
	# print(f'ret--->{ret}')

	# 查找多个子标签
	# print('*' * 80)
	# labels = ['电脑', '衣服']
	# labels = [[4510, 5554], [6132, 3302]]
	# result = verbalizer.batch_find_sub_labels(labels)
	# print(f'result--》{result}')

	# 查找单个子标签对应的父标签
	# print('*' * 80)
	# sub_label = [4510, 5554]
	# sub_label = "电脑"
	# sub_label = [6132, 4510]
	# print(verbalizer.tokenizer.convert_ids_to_tokens(sub_label))
	# sub_label = '衣电'
	# ret = verbalizer.find_main_label(sub_label)
	# print(f'ret--->{ret}')

	# 查找多个子标签对应的父标签
	sub_label = ['衣服', '牛奶']
	# sub_label = [[6132, 3302], [5885, 4281]]
	# sub_label = ['衣电', '牛奶']
	# sub_label = [[6132, 4510], [5885, 4281]]
	ret = verbalizer.batch_find_main_label(sub_label, hard_mapping=True)
	print(f'ret--->{ret}')