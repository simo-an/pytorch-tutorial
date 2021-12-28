import numpy as np

from typing import List

'''
n: 物品总数
c: 背包总质量
w: 每件物品的重量
v: 每件武平的价值
'''
def knapsack(n: int, c: int, w: List[int], v: List[int]):
    m = np.zeros((n + 1, c + 1), dtype=int)

    for i in range(n+1):
        for j in range(c+1):
            if i==0 or j==0:
                m[i, j] = 0
            elif j < w[i]:
                m[i, j] = m[i-1, j]
            else:
                m[i, j] = max(m[i-1, j], m[i-1, j-w[i]]+v[i])

    print(m[5, 10])
    
        
    


n: int = 5
c: int = 10
w: List[int] = [0, 2, 2, 6, 5, 4]
v: List[int] = [0, 6, 3, 5, 4, 6]
knapsack(n, c, w, v)
