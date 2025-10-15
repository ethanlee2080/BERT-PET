# -*- coding:utf-8 -*-
from rich import print
from transformers import AutoTokenizer
import numpy as np
# 导入sys模块以操作Python解释器相关的设置
import sys

# 在Python解释器的搜索路径列表中添加上一级目录
# 这样做可以让Python解释器在寻找模块时也能找到上一级目录中的模块
# 通常用于在项目中包含外部库或模块，而不必每次都安装这些库或模块到全局环境中
sys.path.append('..')
# print('sys.path--->', sys.path)
from PET.pet_config import *


class HardTemplate(object):
	"""
	硬模板，人工定义句子和[MASK]之间的位置关系。
	"""

	def __init__(self, prompt: str):
		"""
		初始化Prompt对象的构造函数
		Args:
			prompt (str): prompt格式定义字符串, 表示待处理的提示模板 e.g. -> "这是一条{MASK}评论：{textA}。"
		"""
		self.prompt = prompt  # 保存原始的提示模板字符串
		self.inputs_list = []  # 根据文字prompt拆分为各part的列表
		self.custom_tokens = set(['MASK'])  # 从prompt中解析出的自定义token集合
		# self.custom_tokens = {'MASK'}  # 初始化自定义token集合，至少包含'MASK' token
		self.prompt_analysis()  # 解析prompt模板，初始化时即对prompt进行分析处理

	def prompt_analysis(self):
		"""
		将prompt文字模板拆解为可映射的数据结构。

		Examples:
			prompt -> "这是一条{MASK}评论：{textA}。"
			inputs_list -> ['这', '是', '一', '条', 'MASK', '评', '论', '：', 'textA', '。']
			custom_tokens -> {'textA', 'MASK'}
		"""
		# print(f'prompt-->{self.prompt}')
		idx = 0
		# 遍历提示模板字符串中的每个字符
		while idx < len(self.prompt):
			str_part = ''
			# 如果当前字符不是'{', '}'，则直接添加到输入列表中
			if self.prompt[idx] not in ['{', '}']:
				self.inputs_list.append(self.prompt[idx])
			# 如果遇到'{'，表示进入自定义字段部分
			if self.prompt[idx] == '{':  # 进入自定义字段
				idx += 1
				# 继续遍历直到遇到'}'，并将自定义字段的值拼接到str_part中
				while self.prompt[idx] != '}':
					str_part += self.prompt[idx]  # 拼接该自定义字段的值
					idx += 1
			# 如果遇到'}'，但没有对应的'{'，抛出异常提示括号不匹配
			elif self.prompt[idx] == '}':
				raise ValueError("Unmatched bracket '}', check your prompt.")
			# 如果str_part不为空，表示已经完整地获取了一个自定义字段
			if str_part:
				self.inputs_list.append(str_part)  # 将所有自定义字段添加到输入列表中
				self.custom_tokens.add(str_part)  # 将所有自定义字段存储，后续会检测输入信息是否完整
			# 移动到下一个字符
			idx += 1
		# print(f'self.inputs_list-->{self.inputs_list}')
		# print(f'self.custom_tokens-->{self.custom_tokens}')

	def __call__(self,
				 inputs_dict: dict,
				 tokenizer,
				 mask_length,
				 max_seq_len=512):
		"""
		输入一个样本，转换为符合模板的格式。

		Args:
			inputs_dict (dict): prompt中的参数字典, e.g. -> {
															"textA": "这个手机也太卡了",
															"MASK": "[MASK]"
														}
			tokenizer: 用于encoding文本
			mask_length (int): MASK token 的长度

		Returns:
			dict -> {
				'text': '[CLS]这是一条[MASK]评论：这个手机也太卡了。[SEP]',
				'input_ids': [1, 47, 10, 7, 304, 3, 480, 279, 74, 47, 27, 247, 98, 105, 512, 777, 15, 12043, 2],
				'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
				'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
				'mask_position': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
			}
		"""
		# 定义输出格式
		# 初始化一个字典对象以存储处理后的输出数据
		# 该字典包含了文本数据及其对应的编码信息、注意力掩码和掩码位置等关键信息
		outputs = {
			# 存储原始文本数据
			'text': '',
			# 存储文本经过分词和数值化后的输入ID序列
			'input_ids': [],
			# 存储段嵌入（token type embeddings）的ID序列，用于区分不同句子
			'token_type_ids': [],
			# 存储注意力掩码，用于指示每个token是否应该被关注
			'attention_mask': [],
			# 存储掩码位置，即在输入序列中被掩码的token的位置
			'mask_position': []
		}

		# print(f'inputs_dict-->{inputs_dict}')
		# 初始化一个空字符串，用于构建最终的格式化字符串
		str_formated = ''
		# 遍历输入列表中的每个值
		for value in self.inputs_list:
			# 检查当前值是否在custom_tokens中
			if value in self.custom_tokens:
				# 如果当前值是'MASK'，使用mask_length副本的inputs_dict中的对应值
				if value == 'MASK':
					str_formated += inputs_dict[value] * mask_length
				else:
					# 对于其他自定义值，直接添加inputs_dict中的对应值
					str_formated += inputs_dict[value]
			else:
				# 如果当前值不是custom_tokens中的值，直接添加到格式化字符串中
				str_formated += value
		# 打印格式化后的字符串，用于调试和验证
		# print(f'str_formated-->{str_formated}')

		# 使用tokenizer对格式化后的字符串进行编码
		# 编码配置包括截断、最大长度设置和填充，以满足模型输入的要求
		encoded = tokenizer(text=str_formated,
							truncation=True,
							max_length=max_seq_len,
							padding='max_length')
		# print('*'*80)
		# print(f'encoded--->{encoded}')
		# 将编码后的输入ID赋值给输出字典中的'input_ids'键
		outputs['input_ids'] = encoded['input_ids']
		# 将编码后的token类型ID赋值给输出字典中的'token_type_ids'键
		outputs['token_type_ids'] = encoded['token_type_ids']
		# 将编码后的注意力掩码赋值给输出字典中的'attention_mask'键
		outputs['attention_mask'] = encoded['attention_mask']

		# print(tokenizer.convert_ids_to_tokens(encoded['input_ids']))
		# 将编码后的输入ID转换为文本，并存储到输出字典中
		outputs['text'] = ''.join(tokenizer.convert_ids_to_tokens(encoded['input_ids']))
		# print(f'outputs-->{outputs}')
		# print('*' * 80)
		# print(tokenizer.convert_tokens_to_ids(['[MASK]']))
		# 将掩码标记 '[MASK]' 转换为其对应的ID
		mask_token_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
		# print(f'mask_token_id-->{mask_token_id}')
		# print('*' * 80)
		# print(np.array(outputs['input_ids']) == mask_token_id)
		# print('*'*80)
		# print(np.where(np.array(outputs['input_ids']) == mask_token_id))
		# 计算并获取输入ID中'mask'标记的位置，并将其转换为列表
		mask_position = np.where(np.array(outputs['input_ids']) == mask_token_id)[0].tolist()
		# print(f'mask_position--》{mask_position}')
		# 将计算出的mask_position添加到outputs字典中
		outputs['mask_position'] = mask_position
		return outputs


if __name__ == '__main__':
	# 创建ProjectConfig对象以获取项目配置
	pc = ProjectConfig()

	# 根据预训练模型配置，加载分词器
	tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)

	# 定义一个硬模板对象，用于构建特定格式的输入文本
	hard_template = HardTemplate(prompt='这是一条{MASK}评论：{textA}。')
	# 打印硬模板的输入列表和自定义token信息，以便调试
	print(f'inputs_list-->{hard_template.inputs_list}')
	print(f'custom_tokens--》{hard_template.custom_tokens}')

	# 使用硬模板、分词器和指定的输入字典构建一个模板实例
	# 调用模板对象, 自动调用__call__方法
	tep = hard_template(
		inputs_dict={'textA': '包装不错，苹果挺甜的，个头也大。', 'MASK': '[MASK]'},
		tokenizer=tokenizer,
		max_seq_len=30,
		mask_length=2
	)

	print(f'tep--->{tep}')

	# 打印使用模板构建的输入文本和分词后的结果
	print(tokenizer("这是一条[MASK][MASK]评论：包装不错，苹果挺甜的，个头也大。。"))

	# 打印分词器将特定词汇转换为ID的结果
	print(tokenizer.convert_tokens_to_ids(['网', '球']))
