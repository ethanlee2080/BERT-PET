# coding:utf-8
from torch.utils.data import DataLoader
from transformers import default_data_collator
from .data_preprocess import *


from PET.pet_config import *
from PET.data_handle.template import HardTemplate
# from pet_config import *

# 实例化项目配置文件
pc = ProjectConfig()

# 使用项目配置文件中指定的预训练模型，初始化一个自动分词器
tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)


def get_data():
	"""
	加载训练和验证数据集，并进行预处理以适应模型训练。
	该函数首先读取提示模板文件，然后使用该模板创建一个硬模板对象。
	接着，它加载原始数据集，并将其转换为适合模型训练的格式。
	最后，它将处理后的数据集包装在DataLoader对象中，以便在训练过程中方便地访问数据。
	返回:
		train_dataloader: 训练数据集的DataLoader对象。
		dev_dataloader: 验证数据集的DataLoader对象。
	"""
	# 读取提示模板文件的第一行作为prompt
	prompt = open(pc.prompt_file, 'r', encoding='utf-8').readlines()[0].strip()  # prompt定义
	# print(f'prompt--》{prompt}')

	# 使用读取的prompt创建一个硬模板对象
	hard_template = HardTemplate(prompt=prompt)
	# print(f'hard_template--》{hard_template}')

	# 加载原始文本数据集
	dataset = load_dataset('text',
						   data_files={'train': pc.train_path, 'dev': pc.dev_path})
	# print(f'dataset-->{dataset}')

	# 创建一个新函数，用于将示例转换为模型训练所需的格式
	new_func = partial(convert_example,
					   tokenizer=tokenizer,
					   hard_template=hard_template,
					   max_seq_len=pc.max_seq_len,
					   max_label_len=pc.max_label_len)

	# print("*" * 80)
	# 使用新函数对数据集进行映射，进行批量处理
	dataset = dataset.map(new_func, batched=True)
	# print(f'dataset改变之后的-->{dataset}')

	# 提取训练数据集和验证数据集
	train_dataset = dataset["train"]
	# print(f'train_dataset--》{train_dataset}')
	# print(f'train_dataset[0]--》{train_dataset[0]}')
	dev_dataset = dataset["dev"]
	# print(f'dev_dataset--》{dev_dataset}')
	# print('dev_dataset', dev_dataset[0])
	# print('*'*80)

	# 使用default_data_collator将数据转换为tensor数据类型
	train_dataloader = DataLoader(train_dataset,
								  shuffle=True,
								  collate_fn=default_data_collator,
								  batch_size=pc.batch_size)
	# print(f'train_dataloader--》{train_dataloader}')
	dev_dataloader = DataLoader(dev_dataset,
								collate_fn=default_data_collator,
								batch_size=pc.batch_size)

	# 返回处理后的训练和验证数据集的DataLoader对象
	return train_dataloader, dev_dataloader


if __name__ == '__main__':
	# 获取训练和验证数据集的加载器
	train_dataloader, dev_dataloader = get_data()
	print(len(train_dataloader))
	print(len(dev_dataloader))

	# 遍历训练数据集加载器
	for i, value in enumerate(train_dataloader):
		print(f'i--->{i}')
		print(f'value--->{value}')
		# 打印当前数据项中'input_ids'的数据类型
		print(value['input_ids'].dtype)
		break
