'''
Question:
Given n number as a[1], a[2], ..., a[n]
You can take a[i] and a[i+1] and operate like: 
           a[i] = a[i]*a[i+1]+1
After n-1 step, the max result is _____________

Example
    Input: 3,2,4,7
    Output: 204

Schedule:

if i == j
M[i][j] = a[i]

if i != j
M[i][j] = max{ M[i][k] * M[k+1][j] + 1 } i <= k <j

'''

def range_max(M, i, j):
    if i == j: return -1

    MAX = -1
    for k in range(i, j):
        TEMP = M[i][k] * M[k+1][j] + 1
        if TEMP > MAX:
            MAX = TEMP
    
    return MAX



def get_max_result(a):
    n = len(a)
    M = [[0 for i in range(n)] for j in range(n)]

    for i in range(n):
        for j in range(n):
            if j + i >= n: continue

            if i == 0:
                M[j][j] = a[j]
            else:
                M[j][j+i] = range_max(M, j, j+i)  

    return M[0][n-1]

max_result = get_max_result([3, 2, 4, 7])
print(max_result)