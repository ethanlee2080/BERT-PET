"""
加载预训练好的模型并测试效果。
"""
import time
from typing import List
import torch
from rich import print
from transformers import AutoTokenizer, AutoModelForMaskedLM
from data_handle.template import HardTemplate
from utils.verbalizer import Verbalizer
from data_handle.data_preprocess import convert_example
from utils.common_utils import convert_logits_to_ids

# device = 'mps:0' # M1芯片电脑
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_path = 'checkpoints/model_best'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
model.to(device).eval()

max_label_len = 2  # 标签最大长度
verbalizer = Verbalizer(
	verbalizer_file='data/verbalizer.txt',
	tokenizer=tokenizer,
	max_label_len=max_label_len)
prompt = open('data/prompt.txt', 'r', encoding='utf8').readlines()[0].strip()  # prompt定义
hard_template = HardTemplate(prompt=prompt)  # 模板转换器定义
print(f'Prompt is -> {prompt}')


def inference(contents: List[str]):
	"""
	推理函数，输入原始句子，输出mask label的预测值。

	Args:
		contents (List[str]): 描原始句子列表。
	"""
	with torch.no_grad():
		start_time = time.time()

		# 将内容封装为示例字典，准备进行标记化处理
		examples = {'text': contents}

		# 对示例进行标记化处理，返回标记化输出
		tokenized_output = convert_example(
			examples,
			tokenizer,
			hard_template=hard_template,
			max_seq_len=128,
			max_label_len=max_label_len,
			train_mode=False,
			return_tensor=True)

		# 使用模型进行预测，获取logits
		logits = model(input_ids=tokenized_output['input_ids'].to(device),
					   token_type_ids=tokenized_output['token_type_ids'].to(device),
					   attention_mask=tokenized_output['attention_mask'].to(device)).logits

		# 将logits转换为预测标签
		predictions = convert_logits_to_ids(logits, tokenized_output[
			'mask_positions']).cpu().numpy().tolist()  # (batch, label_num)

		# 找到子label属于的主label
		predictions = verbalizer.batch_find_main_label(predictions)

		# 提取预测的标签
		predictions = [ele['label'] for ele in predictions]

		used = time.time() - start_time
		print(f'Used {used}s.')
		return predictions


if __name__ == '__main__':
	contents = [
		'天台很好看，躺在躺椅上很悠闲，因为活动所以我觉得性价比还不错，适合一家出行，特别是去迪士尼也蛮近的，下次有机会肯定还会再来的，值得推荐',
		'环境，设施，很棒，周边配套设施齐全，前台小姐姐超级漂亮！酒店很赞，早餐不错，服务态度很好，前台美眉很漂亮。性价比超高的一家酒店。强烈推荐',
		"物流超快，隔天就到了，还没用，屯着出游的时候用的，听方便的，占地小",
		"福行市来到无早集市，因为是喜欢的面包店，所以跑来集市看看。第一眼就看到了，之前在微店买了小刘，这次买了老刘，还有一直喜欢的巧克力磅蛋糕。好奇老板为啥不做柠檬磅蛋糕了，微店一直都是买不到的状态。因为不爱碱水硬欧之类的，所以期待老板多来点其他小点，饼干一直也是大爱，那天好像也没看到",
		"服务很用心，房型也很舒服，小朋友很喜欢，下次去嘉定还会再选择。床铺柔软舒适，晚上休息很安逸，隔音效果不错赞，下次还会来"
	]
	print("针对下面的文本评论，请分别给出对应所属类别：")
	res = inference(contents)
	print('inference label(s):', res)
	new_dict = {}
	for i in range(len(contents)):
		new_dict[contents[i]] = res[i]
	print(new_dict)
