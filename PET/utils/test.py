from rich import print


def get_common_sub_str(str1: str, str2: str):
	"""
	寻找最大公共子串：两个字符串中同时出现的最长的子串（连续）
	str1:abcd
	str2:abadbcdba
	"""
	# 生成0矩阵，为方便后续计算，比字符串长度多了一行和一列(第0行和第0列作为边界条件)
	# record[i, j]表示公共子串到最后一个字符的长度(最大长度)，该公共子串以str1的第i-1个字符结尾并且以str2的第j-1个字符结尾
	# record[i-1, j-1]表示公共子串到倒数第二个字符的长度, 对应的值比record[i, j]的值小1
	# record[i, j] = record[i-1,j-1] + 1
	# 完成所有record的计算后，选择最大的record值，即为两个字符串str1与str2的最长公共子串长度；
	# 往前回溯，即可得到最长公共子串
	lstr1, lstr2 = len(str1), len(str2)
	print(f'lstr1--》{lstr1}')
	print(f'lstr2--》{lstr2}')

	# 初始化一个二维列表record，用于记录字符串str1和str2中每个位置的最长公共子串长度
	record = [[0 for i in range(lstr2 + 1)] for _ in range(lstr1 + 1)]
	print(f'record-->{record}')  # 输出初始化后的record列表

	# 初始化结束位置和最长匹配长度
	end_index = 0  # 结束位置
	max_length = 0  # 最长匹配长度

	# 遍历str1和str2中的每个字符，使用动态规划寻找最长公共子串
	for i in range(1, lstr1 + 1):
		for j in range(1, lstr2 + 1):
			# 当前字符相同时，更新record列表中的值
			if str1[i - 1] == str2[j - 1]:
				# 当前值等于上一个位置的值加1
				record[i][j] = record[i - 1][j - 1] + 1
				# 如果当前公共子串长度大于已记录的最大长度，更新最大长度和结束位置
				if record[i][j] > max_length:
					max_length = record[i][j]
					end_index = i  # 记录结束位置

	print(f'record-->{record}')  # 输出更新后的record列表
	print(f'end_index-->{end_index}')  # 输出结束位置
	print(f'max_length-->{max_length}')  # 输出最长匹配长度

	# 根据结束位置和最大长度，提取最长公共子串，并返回该子串和最大长度
	return str1[end_index - max_length:end_index], max_length


str1 = "abcd"
str2 = "adbcd"
a = get_common_sub_str(str1=str1, str2=str2)
print(a)
