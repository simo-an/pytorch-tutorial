from typing import List

def merge(a: List[int], left: int, mid:int, right: int):
    b = []
    left_range = mid-left-1
    right_range = right-mid

    left_cursor=0
    right_cursor=0

    while left_cursor <= left_range and right_cursor <= right_range:
        if a[left + left_cursor] < a[mid + right_cursor]:
            b.append(a[left + left_cursor])
            left_cursor += 1
        else:
            b.append(a[mid + right_cursor])
            right_cursor += 1
    
    while left_cursor <= left_range:
        b.append(a[left + left_cursor])
        left_cursor += 1
    
    
    while right_cursor <= right_range:
        b.append(a[mid + right_cursor])
        right_cursor += 1
    
    for i in range(len(b)):
        a[left+i] = b[i]
        


# [left, right)
def merge_sort(a: List[int], left: int, right: int):
    if left >= right:
        return
    
    mid = (left + right)//2

    merge_sort(a, left, mid)
    merge_sort(a, mid+1, right)
    merge(a, left, mid+1, right)

list = [3, 7, 4, 8, 2, 4, 6, 1]

print(list)
merge_sort(list, 0, len(list)-1)
print(list)