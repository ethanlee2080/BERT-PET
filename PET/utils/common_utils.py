# 导入必备工具包

import torch
from rich import print


def mlm_loss(logits,
             mask_positions,
             sub_mask_labels,
             cross_entropy_criterion,
             device):
    """
    计算指定位置的mask token的output与label之间的cross entropy loss。

    Args:
        logits (torch.tensor): 模型原始输出 -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token的位置  -> (batch, mask_label_num)
        sub_mask_labels (list): mask token的sub label, 由于每个label的sub_label数目不同，所以这里是个变长的list,
                                    e.g. -> [
                                        [[2398, 3352]],
                                        [[2398, 3352], [3819, 3861]]
                                    ]
        cross_entropy_criterion (CrossEntropyLoss): CE Loss计算器
        device (str): cpu还是gpu

    Returns:
        torch.tensor: CE Loss
    """
    # 获取logits的尺寸信息，为后续计算做准备
    # logits.size()返回一个包含三个维度的元组
    # 第一个维度(batch_size)代表批次大小，即一次处理的数据批次包含的样本数量
    # 第二个维度(seq_len)代表序列长度，即每个样本中包含的序列元素数量
    # 第三个维度(vocab_size)代表词汇表大小，即每个序列元素可能的类别数量
    batch_size, seq_len, vocab_size = logits.size()
    # print(f'模型预测结果logits-->{logits.size()}')
    # print(f'mask_positions-->{mask_positions.shape}')
    # print("*"*80)
    # print(f'sub_mask_labels-->{sub_mask_labels}')
    # print("*" * 80)
    # 初始化loss变量为None，用于后续可能的损失计算
    loss = None
    # for single_logits, single_sub_mask_labels, single_mask_positions in zip(logits, sub_mask_labels, mask_positions):
    # 遍历 logits、sub_mask_labels 和 mask_positions 的元素
    for single_value in zip(logits, sub_mask_labels, mask_positions):
        # 获取当前元素中的 logits
        single_logits = single_value[0]
        # print(f'single_logits-->{single_logits.shape}')
        # 获取当前元素中的 sub_mask_labels
        single_sub_mask_labels = single_value[1]
        # print(f'single_sub_mask_labels-->{single_sub_mask_labels}')
        # 获取当前元素中的 mask_positions
        single_mask_positions = single_value[2]
        # print(f'single_mask_positions-->{single_mask_positions}')
        # print("*"*80)

        # todo:single_logits-->形状[512, 21128],
        # todo:single_mask_positions--》形状size[2]-->具体值([5, 6])
        # 从单个序列的logits中，提取出被掩码位置的logits
        single_mask_logits = single_logits[single_mask_positions]  # (mask_label_num, vocab_size)
        # 打印被掩码位置logits的形状，以验证其是否符合预期
        # print(f'single_mask_logits4--<{single_mask_logits.shape}')
        # single_mask_logits-->水果-》【苹果；香蕉；橘子】

        # repeat重复的倍数
        # (sub_label_num, mask_label_num, vocab_size)
        # 对单个mask logits进行扩展，使其在第一个维度上重复，以匹配sub mask labels的数量
        # 这是为了在后续处理中能够对每个sub mask label应用对应的mask logits
        # 模型训练时主标签对应的所有子标签都有相似的特征值, 所以需要重复
        single_mask_logits = single_mask_logits.repeat(len(single_sub_mask_labels), 1, 1)
        # 打印扩展后的single_mask_logits的形状，以便调试和验证重复操作的效果
        # print(f'single_mask_logits5:{single_mask_logits.shape}')

        # 将三维张量调整为二维，以便计算损失
        single_mask_logits = single_mask_logits.reshape(-1, vocab_size)  # (sub_label_num * mask_label_num, vocab_size)
        # print(f'模型预测的结果：single_mask_logits6:{single_mask_logits.shape}')

        # 将子标签转换为张量，并调整形状以匹配模型预测的结果
        single_sub_mask_labels = torch.LongTensor(single_sub_mask_labels).to(device)  # (sub_label_num, mask_label_num)
        # 计算损失值时真实子标签维度为1维，因此需要将其展平以匹配模型预测的结果
        single_sub_mask_labels = single_sub_mask_labels.reshape(-1, 1).squeeze()  # (sub_label_num * mask_label_num)
        # print(f'真实子标签mask值：single_sub_mask_labels7-->{single_sub_mask_labels.shape}')
        # print(f'真实子标签mask值：single_sub_mask_labels7-->{single_sub_mask_labels}')

        # 计算当前批次所有子标签的损失
        cur_loss = cross_entropy_criterion(single_mask_logits, single_sub_mask_labels)
        # 计算当前批次所有子标签的平均损失
        cur_loss = cur_loss / len(single_sub_mask_labels)

        # 如果当前损失loss未被初始化（即为None），则将其设置为当前批次的损失cur_loss
        if not loss:
            loss = cur_loss
        # 如果当前损失loss已经存在，则将当前批次的损失cur_loss累加到loss中
        else:
            loss += cur_loss

    # 计算平均损失：将累计的损失loss除以批次大小batch_size
    loss = loss / batch_size  # (1,)
    return loss


def convert_logits_to_ids(
        logits: torch.tensor,
        mask_positions: torch.tensor):
    """
    输入Language Model的词表概率分布（LMModel的logits），将mask_position位置的
    token logits转换为token的id。

    Args:
        logits (torch.tensor): model output -> (batch, seq_len, vocab_size)[8, 512, 21128]
        mask_positions (torch.tensor): mask token的位置 -> (batch, mask_label_num)[8, 2]

    Returns:
        torch.LongTensor: 对应mask position上最大概率的推理token -> (batch, mask_label_num)[8, 2]
    """
    # 获取标签的长度，mask_positions.size()返回的是一个包含维度的元组，[1]表示获取第二个维度的大小
    label_length = mask_positions.size()[1]
    # print(f'label_length--》{label_length}')

    # 获取批次大小、序列长度和词汇表大小，logits.size()返回的是一个包含维度的元组
    batch_size, seq_len, vocab_size = logits.size()

    # 初始化一个空列表，用于存储重塑后的mask_positions
    mask_positions_after_reshaped = []

    # print(f'mask_positions.detach().cpu().numpy().tolist()-->{mask_positions.detach().cpu().numpy().tolist()}')
    # 遍历每个批次的mask_positions
    for batch, mask_pos in enumerate(mask_positions.detach().cpu().numpy().tolist()):
        # 遍历每个mask位置
        for pos in mask_pos:
            # 将批次号和序列中的mask位置结合起来，得到重塑后的mask_positions
            mask_positions_after_reshaped.append(batch * seq_len + pos)
    # print(f'mask_positions_after_reshaped-->{mask_positions_after_reshaped}')
    # print(f'原始的logits-->{logits.shape}')

    # 将原始的logits重塑为(batch_size * seq_len, vocab_size)的形状 (8, 256, 21128)
    logits = logits.reshape(batch_size * seq_len, -1)  # (batch_size * seq_len, vocab_size)
    # print('改变原始模型输出的结果形状', logits.shape)

    # 从重塑后的logits中，选择出被掩码位置的logits
    mask_logits = logits[mask_positions_after_reshaped]  # (batch * label_num, vocab_size)
    # print('选择真实掩码位置预测的数据形状',mask_logits.shape)

    # 获取每个样本真实mask位置预测的tokens
    predict_tokens = mask_logits.argmax(dim=-1)  # (batch * label_num)
    # print('求出每个样本真实mask位置预测的tokens', predict_tokens)

    # 将每个样本真实mask位置预测的tokens重塑为(batch, label_num)的形状
    predict_tokens = predict_tokens.reshape(-1, label_length)  # (batch, label_num)
    # print(f'predict_tokens--》{predict_tokens}')
    return predict_tokens


if __name__ == '__main__':
    logits = torch.randn(2, 20, 21128)
    mask_positions = torch.LongTensor([
        [5, 6],
        [5, 6],
    ])
    predict_tokens = convert_logits_to_ids(logits, mask_positions)
    print(predict_tokens)
