import numpy as np
import matplotlib.pyplot as plt

'''
    1. Matplotlib - Basic
'''

# # 基本使用
# plt.plot([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
# plt.xlabel('Step', fontsize=16)
# plt.ylabel('Value')

# # 不同线条、颜色
# plt.plot([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], 'b-.')
# # plt.plot([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], '-.', color='r')
# plt.xlabel('Step')
# plt.ylabel('Value')

# # 画多条直线
# arr1 = np.arange(0, 10, step=0.5)
# plt.plot(arr1, arr1, 'r--')
# plt.plot(arr1, arr1**2, 'y--')

# # 画出函数图
# x = np.linspace(-10, 10)
# y = np.sin(x)
# plt.plot(x, y,linestyle=':', marker='o', linewidth=2.0, markerfacecolor='r')

# # 
# x = np.linspace(-10, 10)
# y = np.sin(x)

# line = plt.plot(x, y)
# plt.setp(line, color='r', linewidth=2.0, alpha=0.5)

# # 绘制多个图
# plt.subplot(211) # 2行1列中的第1个
# plt.plot([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], color='r')

# plt.subplot(212) # 2行1列中的第2个
# plt.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], color='r')

# # 图上加说明
# x = [-2, -1, 0, 1, 2]
# y = [-2, -1, 0, 1, 2]
# plt.plot(x, y, color='r', marker='o', markerfacecolor='b')
# plt.xlabel('step')
# plt.ylabel('value')
# plt.title('graph')
# plt.text(0, 0, 'Origion')
# plt.grid(True)
# plt.annotate(
#     'Key point', xy=(1, 1), xytext=(1.4, 0.8),  # 添加注释
#     arrowprops=dict(facecolor='yellow', shrink=0.05) # 加箭头
# )

# plt.show()

'''
    2. Matplotlib - Style
'''

# plt.style.available —— 查看所有的风格
#   Solarize_Light2, _classic_test_patch, ...

# x = np.linspace(-10, 10)
# y = np.sin(x)

# # plt.style.use('seaborn')
# # plt.style.use('Solarize_Light2')
# # plt.xkcd()
# plt.plot(x, y)


# plt.show()

