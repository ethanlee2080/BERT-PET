
'''
partial函数解析
partial(fun, *args, **keywords)
'''
from functools import partial

def fun(a, b, c):
    print(f'a-->{a}')
    print(f'b-->{b}')
    return a + b
#
# c = fun(1, 2)
# print(c)
# #
cc = partial(fun,b=2, c=2)
print(cc(1))
# list1 = [[[1, 2]], [[2, 3]]] # -->[2,1, 2]
# list2 = [[2], [3]] # -->[2,1]
# list3 = [[[3, 4]], [[5, 6]], [[3,5]]] #--->[3, 1,2]
#
# print(list(zip(list1, list2, list3)))

# import os
# base_path = os.path.abspath('.')
# print(f'base_path-->{base_path}')
# cur_save_dir = os.path.join(base_path, '%s' % "ai18")
# print(cur_save_dir)
# a = os.path.join(cur_save_dir)
# print(a)
# import torch
# a = torch.tensor([[1, 2, 0, 10, 9,],
#                   [2, 4, 6, 1, 0]])
# b = a.argmax(dim=-1)
# print(b)

# import numpy as np
#
# a = np.array([1.0, 2.0, 3.0])
# # print(a == 3.0)
# print(np.where(a < 3.0))


