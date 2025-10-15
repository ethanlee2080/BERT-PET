import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import time
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler
from utils.metric_utils import ClassEvaluator
from utils.common_utils import *
from data_handle.data_loader import *
from utils.verbalizer import Verbalizer
from pet_config import *
from tqdm import tqdm

pc = ProjectConfig()


def evaluate_model(model,
				   metric,
				   data_loader,
				   tokenizer,
				   verbalizer):
	"""
	在测试集上评估当前模型的训练效果。

	Args:
		model: 当前模型
		metric: 评估指标类(metric)
		data_loader: 测试集的dataloader
		global_step: 当前训练步数
	"""
	model.eval()
	metric.reset()

	with torch.no_grad():
		for step, batch in enumerate(tqdm(data_loader)):
			# print(f'batch--》{batch}')
			logits = model(input_ids=batch['input_ids'].to(pc.device),
						   token_type_ids=batch['token_type_ids'].to(pc.device),
						   attention_mask=batch['attention_mask'].to(pc.device)).logits
			# print(f'验证集模型预测的结果————》{logits.shape}')

			mask_labels = batch['mask_labels'].numpy().tolist()  # (batch, label_num)
			# print(f"mask_labels-0-->{mask_labels}")

			for i in range(len(mask_labels)):  # 去掉label中的[PAD] token
				while tokenizer.pad_token_id in mask_labels[i]:
					mask_labels[i].remove(tokenizer.pad_token_id)
			# print(f'mask_labels-1-->{mask_labels}')
			# 将mask_labels id转换为文字
			mask_labels = [''.join(tokenizer.convert_ids_to_tokens(t)) for t in mask_labels]
			# print(f'真实的结果主标签：mask_labels_str-->{mask_labels}')

			# 获取模型预测的子标签
			predictions = convert_logits_to_ids(logits,
												batch['mask_positions']).cpu().numpy().tolist()  # (batch, label_num)
			# print(f'模型预测的子标签的结果--》{predictions}')

			# 根据模型预测的子标签，找到子label属于的主label
			predictions = verbalizer.batch_find_main_label(predictions)  # 找到子label属于的主label
			# print(f"找到模型预测的子标签对应的主标签的结果--》{predictions}')")

			# 获得预测的主标签名
			predictions = [ele['label'] for ele in predictions]
			# print(f"只获得预测的主标签的结果string--》{predictions}')")

			# 调用add_batch方法, 将模型预测的主标签与真实主标签保存到metric属性中
			metric.add_batch(pred_batch=predictions, gold_batch=mask_labels)
	eval_metric = metric.compute()
	model.train()

	return eval_metric['accuracy'], eval_metric['precision'], \
		eval_metric['recall'], eval_metric['f1'], \
		eval_metric['class_metrics']


def model2train():
	# 加载预训练模型
	model = AutoModelForMaskedLM.from_pretrained(pc.pre_model)
	# print(f'预训练模型带MLM头的--》{model}')
	# 加载分词器
	tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
	verbalizer = Verbalizer(verbalizer_file=pc.verbalizer,
							tokenizer=tokenizer,
							max_label_len=pc.max_label_len)
	# print(f'verbalizer--》{verbalizer.label_dict}')

	# 不需要权重衰减的参数
	no_decay = ["bias", "LayerNorm.weight"]
	# print(type(model.parameters()))
	# 定义优化器的参数组，以便对模型的不同部分应用不同的权重衰减
	optimizer_grouped_parameters = [
		# 第一组参数：包含所有适用权重衰减的模型参数
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": pc.weight_decay,
		},
		# 第二组参数：包含所有不适用权重衰减的模型参数
		{
			"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
			"weight_decay": 0.0,
		},
	]
	# 初始化AdamW优化器，用于模型参数的优化
	# AdamW是Adam算法的变体，加入了权重衰减（L2正则化），有助于防止过拟合
	# 参数optimizer_grouped_parameters是分组的模型参数，允许对不同的参数应用不同的学习率或正则化强度
	optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=pc.learning_rate)

	# 将模型移动到指定的设备上，例如CPU、GPU或其他加速器
	model.to(pc.device)

	# 加载训练数据和验证数据
	train_dataloader, dev_dataloader = get_data()
	# 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
	num_update_steps_per_epoch = len(train_dataloader)
	# 指定总的训练步数，它会被学习率调度器用来确定学习率的变化规律，确保学习率在整个训练过程中得以合理地调节
	max_train_steps = pc.epochs * num_update_steps_per_epoch
	# 计算预热阶段的训练步数，用于初始化学习率调度
	warm_steps = int(pc.warmup_ratio * max_train_steps)  # 预热阶段的训练步数
	# 创建学习率调度器，使用线性调度策略，根据训练的进行逐步调整学习率
	lr_scheduler = get_scheduler(
		name='linear',
		optimizer=optimizer,
		num_warmup_steps=warm_steps,
		num_training_steps=max_train_steps)

	# 初始化损失列表，用于记录训练过程中的损失值
	loss_list = []
	# 记录训练开始的时间，用于计算训练时长
	tic_train = time.time()
	# 创建分类评估器，用于评估模型性能
	metric = ClassEvaluator()
	# 定义损失函数，用于计算模型预测值与真实标签之间的差异
	criterion = torch.nn.CrossEntropyLoss()
	# 初始化训练次数和最佳F1分数，用于跟踪训练进度和模型性能
	global_step, best_f1 = 0, 0

	print('开始训练：')
	for epoch in range(pc.epochs):
		# tqdm: 进度条
		for batch in tqdm(train_dataloader):
			# print(f'batch--》{batch}')
			# 将批次数据输入模型，获取logits
			logits = model(input_ids=batch['input_ids'].to(pc.device),
						   token_type_ids=batch['token_type_ids'].to(pc.device),
						   attention_mask=batch['attention_mask'].to(pc.device)).logits

			# print(f'logits->{logits.shape}')
			# print('*'*80)
			# 真实标签
			mask_labels = batch['mask_labels'].numpy().tolist()
			# print(f'mask_labels--->{mask_labels}')
			# 提取子标签
			sub_labels = verbalizer.batch_find_sub_labels(mask_labels)
			# print(f'sub_labels--->{sub_labels}')
			# 获取子标签的token_ids
			sub_labels = [ele['token_ids'] for ele in sub_labels]
			# print(f'sub_labels_token_ids--->{sub_labels}')

			# 计算掩码语言模型的损失值
			loss = mlm_loss(logits,
							batch['mask_positions'].to(pc.device),
							sub_labels,
							criterion,
							pc.device)
			# print(f'计算损失值--》{loss}')
			# 清零优化器的梯度
			optimizer.zero_grad()
			# 反向传播计算梯度
			loss.backward()
			# 更新模型参数
			optimizer.step()
			# 更新学习率调度器
			lr_scheduler.step()
			# loss_list.append(float(loss.cpu().detach()))
			# 将损失值添加到损失列表中
			loss_list.append(loss)
			# 训练次数增加1
			global_step += 1

			# 打印训练日志
			if global_step % pc.logging_steps == 0:
				time_diff = time.time() - tic_train
				loss_avg = sum(loss_list) / len(loss_list)
				print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
					  % (global_step, epoch, loss_avg, pc.logging_steps / time_diff))
				tic_train = time.time()
			# 模型验证
			if global_step % pc.valid_steps == 0:
				# 使用给定的模型、评估指标、数据加载器、分词器和标记化器进行模型评估
				acc, precision, recall, f1, class_metrics = evaluate_model(model,
																		   metric,
																		   dev_dataloader,
																		   tokenizer,
																		   verbalizer)

				# 打印评估结果中的精确度、召回率和F1分数
				print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))
				# 如果当前F1分数高于最佳F1分数，则更新最佳F1分数和相关模型及分词器
				if f1 > best_f1:
					print(
						f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
					)
					print(f'Each Class Metrics are: {class_metrics}')
					# 更新当前最佳的F1分数
					best_f1 = f1
					# 定义当前保存模型和分词器的目录
					cur_save_dir = os.path.join(pc.save_dir, "model_best")
					print(cur_save_dir)
					# 检查并创建保存目录（如果不存在）
					if not os.path.exists(cur_save_dir):
						os.makedirs(cur_save_dir)
					# 保存模型到指定目录
					model.save_pretrained(cur_save_dir)
					# 保存分词器到指定目录
					tokenizer.save_pretrained(cur_save_dir)
				tic_train = time.time()

	print('训练结束')


if __name__ == '__main__':
	model2train()
