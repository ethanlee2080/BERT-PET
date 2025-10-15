# coding:utf-8
import torch
import sys


# print(sys.path)


class ProjectConfig(object):
	def __init__(self):
		# 初始化设备配置，根据系统环境选择使用GPU或CPU
		self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		# MAC电脑专用配置（如需在MAC上运行，请取消注释）
		# self.device = "mps:0"

		# 预训练模型路径配置
		self.pre_model = 'D:/AI/Projects/PET/bert-base-chinese'

		# 训练、验证数据集路径配置
		self.train_path = 'D:/AI/Projects/PET/PET/data/train.txt'
		self.dev_path = 'D:/AI/Projects/PET/PET/data/dev.txt'

		# 提示词和标签映射文件路径配置
		self.prompt_file = 'D:/AI/Projects/PET/PET/data/prompt.txt'
		self.verbalizer = 'D:/AI/Projects/PET/PET/data/verbalizer.txt'

		# 模型输入序列最大长度配置
		self.max_seq_len = 256

		# 设置训练的超参数
		self.batch_size = 8  # 每个批次的大小，根据显存和模型大小调整
		self.learning_rate = 5e-5  # 学习率，影响模型收敛速度和效果
		self.weight_decay = 0  # 权重衰减，用于防止过拟合，这里不使用权重衰减
		self.warmup_ratio = 0.06  # 学习率预热比例，帮助模型初期更快地学习
		self.max_label_len = 2  # 最大标签长度，限制输出序列的最大长度
		self.epochs = 20  # 训练的轮数，即整个数据集通过模型的次数

		# 日志和验证配置
		self.logging_steps = 2
		self.valid_steps = 20

		# 模型保存路径配置
		self.save_dir = 'D:/AI/Projects/PET/PET/checkpoints'


if __name__ == '__main__':
	pc = ProjectConfig()
	print(pc.prompt_file)
	print(pc.pre_model)
