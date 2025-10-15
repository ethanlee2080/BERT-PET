import torch
# # 1. 输入一维张量，参数为一个，即表示在列上面进行重复n次
# a = torch.randn(3)
# print(a)
# print(a.repeat(4))
# # #
# print("*"*80)
# #
# # # 2. 输入一维张量，参数为两个(m,n)，即表示先在列上面进行重复n次，再在行上面重复m次，输出张量为二维
# a = torch.randn(3)
# print(a)
# print(a.repeat(4, 2))
# #
# print("*"*80)
# #
# # # 3. 输入一维张量，参数为三个(b,m,n)，即表示先在列上面进行重复n次，再在行上面重复m次，最后在通道上面重复b次，输出张量为三维
# a = torch.randn(3)
# print(a)
# print(a.repeat(3, 4, 2))
# #
# print("*"*80)
#
# # 4. 输入二维张量，参数为两个(m,n)，即表示先在列上面进行重复n次，再在行上面重复m次，输出张量为两维（注意参数个数必须大于等于输入张量维度个数）
a = torch.randn(3, 2)
print(a)
print(a.repeat(4, 2))
#
print("*"*80)
#
# # 5. 输入二维张量，参数为三个(b,m,n)，即表示先在列上面进行重复n次，再在行上面重复m次，最后在通道上面重复b次，输出张量为三维。（注意输出张量维度个数为参数个数）
a = torch.randn(3, 2)
print(a)
print(a.repeat(3, 1, 1))
# print(a.repeat(3, 4, 2))
# #
# # print("*"*80)

