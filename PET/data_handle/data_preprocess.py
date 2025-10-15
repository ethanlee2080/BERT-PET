# 导入必备工具包
from .template import *
from rich import print
from datasets import load_dataset
from functools import partial  # partial是对函数进行再次封装，便于使用
from PET.pet_config import *


def convert_example(
		examples: dict,
		tokenizer,
		max_seq_len: int,
		max_label_len: int,
		hard_template: HardTemplate,
		train_mode=True,
		return_tensor=False) -> dict:
	"""
	将样本数据转换为模型接收的输入数据。

	Args:
		examples (dict): 训练数据样本, e.g. -> {
												"text": [
															'手机	这个手机也太卡了。',
															'体育	世界杯为何迟迟不见宣传',
															...
												]
											}
		max_seq_len (int): 句子的最大长度，若没有达到最大长度，则padding为最大长度
		max_label_len (int): 最大label长度，若没有达到最大长度，则padding为最大长度
		hard_template (HardTemplate): 模板类。
		train_mode (bool): 训练阶段 or 推理阶段。
		return_tensor (bool): 是否返回tensor类型，如不是，则返回numpy类型。

	Returns:
		dict (str: np.array) -> tokenized_output = {
							'input_ids': [[1, 47, 10, 7, 304, 3, 3, 3, 3, 47, 27, 247, 98, 105, 512, 777, 15, 12043, 2], ...],
							'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...],
							'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], ...],
							'mask_positions': [[5, 6, 7, 8], ...],
							'mask_labels': [[2372, 3442, 0, 0], [2643, 4434, 2334, 0], ...]
						}
	"""
	# 初始化一个字典，用于存储token化的输出信息
	# 该字典包含以下键值对：
	# 'input_ids': 输入文本的token ID序列
	# 'token_type_ids': token类型ID序列，用于区分不同句子的token
	# 'attention_mask': 注意力掩码序列，用于标识真实token与padding token
	# 'mask_positions': mask标签在输入序列中的位置
	# 'mask_labels': 需要预测的mask标签的真实值
	tokenized_output = {
		'input_ids': [],
		'token_type_ids': [],
		'attention_mask': [],
		'mask_positions': [],
		'mask_labels': []
	}

	# print(f'examples--》{examples}')
	# 遍历examples中的'text'列表，获取索引和文本内容
	for i, example in enumerate(examples['text']):
		# 判断是否处于训练模式
		if train_mode:
			# print(f'example-->{example}')
			# 将文本内容按制表符分割，获取标签和内容
			label, content = example.strip().split('\t')
			# print(f'label-->{label}')
			# print(f'content-->{content}')

			# 使用tokenizer对标签进行编码，以确保其长度达到预定义的最大长度
			label_encoded = tokenizer(text=[label])  # 将label补到最大长度
			# print(f'label_encoded-->{label_encoded}')

			# 从 label_encoded 字典中提取 'input_ids' 键对应的第一个序列，去除序列首尾的特殊标记
			label_encoded = label_encoded['input_ids'][0][1:-1]
			# print(f'label_encoded-->{label_encoded}')

			# 如果标签长度超过最大标签长度, 将标签编码序列的长度限制在最大标签长度内
			if len(label_encoded) >= max_label_len:
				label_encoded = label_encoded[:max_label_len]
			# 如果标签长度小于最大标签长度, 将标签编码序列进行填充，以确保其长度与max_label_len相等
			else:
				# 这里使用了tokenizer的pad_token_id属性作为填充元素
				# print(f'tokenizer.pad_token_id-->{tokenizer.pad_token_id}')
				label_encoded = label_encoded + [tokenizer.pad_token_id] * (max_label_len - len(label_encoded))

			# 将编码后的标签添加到tokenized_output字典中的'mask_labels'列表中
			tokenized_output['mask_labels'].append(label_encoded)
		else:
			# 如果不是训练模式，直接将文本内容进行修剪并使用
			content = example.strip()

		# 初始化输入字典，用于准备文本数据和特殊标记
		inputs_dict = {
			'textA': content,  # 'textA' 键对应的是后续处理的主要文本内容
			'MASK': '[MASK]'  # 'MASK' 键用于标识特殊的掩码标记，常用于语言模型中
		}
		# print(f'inputs_dict-->{inputs_dict}')

		# 使用硬模板编码方法处理输入数据
		# 该方法将输入数据字典、tokenizer、最大序列长度和最大标签长度作为参数
		# 目的是将输入数据编码成模型所需的格式
		encoded_inputs = hard_template(
			inputs_dict=inputs_dict,
			tokenizer=tokenizer,
			max_seq_len=max_seq_len,
			mask_length=max_label_len)
		# print(f'encoded_inputs--》{encoded_inputs}')

		# print('*' * 80)
		# 将编码后的输入ID添加到输出字典中的input_ids列表
		tokenized_output['input_ids'].append(encoded_inputs["input_ids"])
		# 将编码后的token类型ID添加到输出字典中的token_type_ids列表
		tokenized_output['token_type_ids'].append(encoded_inputs["token_type_ids"])
		# 将编码后的注意力掩码添加到输出字典中的attention_mask列表
		tokenized_output['attention_mask'].append(encoded_inputs["attention_mask"])
		# 将遮罩位置信息添加到输出字典中的mask_positions列表
		tokenized_output['mask_positions'].append(encoded_inputs["mask_position"])
		# print(f'tokenized_output-->{tokenized_output}')

	# 遍历tokenized_output字典，其中k是键，v是值
	for k, v in tokenized_output.items():
		# 如果return_tensor为True，将值转换为torch.LongTensor类型
		if return_tensor:
			tokenized_output[k] = torch.LongTensor(v)
		# 否则，将值转换为numpy数组
		else:
			tokenized_output[k] = np.array(v)

	return tokenized_output


if __name__ == '__main__':
	# 创建ProjectConfig对象以获取项目配置
	pc = ProjectConfig()

	# 加载训练数据集
	# 使用ProjectConfig中定义的训练数据路径
	train_dataset = load_dataset('text', data_files=pc.train_path)
	print(type(train_dataset))
	print(f'train_dataset-->{train_dataset}')
	print('*'*80)
	# print(train_dataset['train'])
	# print('*'*80)
	# print(train_dataset['train']['text'])

	# 使用预训练模型的分词器进行初始化
	tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)

	# 定义一个硬模板，用于将特定的文本结构化到模型输入中
	# {MASK}用于指示模型需要预测的位置，{textA}是输入文本的占位符
	hard_template = HardTemplate(prompt='这是一条{MASK}评论：{textA}')

	# 定义示例输入，包含需要处理的文本数据
	# 每个元素是一个包含类别和评论文本的字符串，用制表符分隔
	# examples = {"text": ['手机	这个手机也太卡了。',
	# 					 '体育	世界杯为何迟迟不见宣传']}
	# print("*" * 80)
	# 将示例转换为模型训练所需的格式
	# tokenized_output = convert_example(examples=examples,
	# 								   tokenizer=tokenizer,
	# 								   max_seq_len=30,
	# 								   max_label_len=2,
	# 								   hard_template=hard_template,
	# 								   train_mode=True,
	# 								   return_tensor=False)
	# print(f'tokenized_output-->{tokenized_output}')

	# 使用functools.partial函数创建一个部分应用函数convert_func
	# 此函数基于convert_example函数，预先设置了一些参数，以便于后续的调用中简化操作
	# 这样做是为了优化样本处理流程，将频繁使用的参数固定下来，提高代码复用性和灵活性
	convert_func = partial(convert_example,
						   tokenizer=tokenizer,
						   hard_template=hard_template,
						   max_seq_len=30,
						   max_label_len=2)

	# 使用map方法对训练数据集进行批量转换
	# batched=True相当于将train_dataset看成一个批次的样本直接对数据进行处理，节省时间
	dataset = train_dataset.map(convert_func, batched=True)
	# 打印整个数据集的概览
	print(f"dataset-->{dataset}")
	print("*" * 80)
	# 打印训练集的概览
	print(f"dataset['train']--》{dataset['train']}")
	print("*" * 80)
	# 打印训练集的长度，即训练样本的数量
	print(len(dataset['train']))
	print("*" * 80)
	# 打印训练集的第一个样本，查看样本的具体结构
	print(dataset["train"][0])

	# # 遍历数据集中的训练数据部分
	# for value in dataset['train']:
	# 	# 打印当前训练数据示例
	# 	print(value)
	# 	# 打印输入ID序列的长度
	# 	print(len(value['input_ids']))
	# 	# 打印输入ID序列的数据类型
	# 	print(type(value['input_ids']))
	# 	# 仅打印第一个训练数据示例后跳出循环
	# 	break
