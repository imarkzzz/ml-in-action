import random
import copy
import operator
import math
import time

def bubble_sort(arr):
    for i in range(1, len(arr)):
        for j in range(0, len(arr) - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def selection_sort(arr):
    for i in range(len(arr) - 1):
        # 记录最小数的索引
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        # i 不是最小数时，将i和最小数交换
        if i != min_idx:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


def insertion_sort(arr):
    for i in range(len(arr)):
        pre_idx = i - 1
        curr = arr[i]
        while pre_idx >= 0 and arr[pre_idx] > curr:
            arr[pre_idx + 1] = arr[pre_idx]
            pre_idx -= 1
        arr[pre_idx + 1] = curr
    return arr

def shell_sort(arr):
    gap = 1
    while (gap < len(arr) / 3):
        gap = gap * 3 + 1
    while gap > 0:
        for i in range(gap, len(arr)):
            tmp = arr[i]
            j = i - gap
            while j >= 0 and arr[j] > tmp:
                arr[j + gap] = arr[j]
                j -= gap
            arr[j + gap] = tmp
        gap = math.floor(gap / 3)
    return arr

def merge(left, right):
    result = []
    while left and right:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    while left:
        result.append(left.pop(0))
    while right:
        result.append(right.pop(0))
    return result

def merge_sort(arr):
    if len(arr) < 2:
        return arr
    mid = math.floor(len(arr) / 2)
    left, right = arr[0:mid], arr[mid:]
    return merge(merge_sort(left), merge_sort(right))

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def partition(arr, left, right):
    pivot = left
    idx = pivot + 1
    i = idx
    while i <= right:
        if arr[i] < arr[pivot]:
            swap(arr, i, idx)
            idx += 1
        i += 1
    swap(arr, pivot, idx - 1)
    return idx - 1

def quick_sort(arr, left=None, right=None):
    left = 0 if not isinstance(left, (int, float)) else left
    right = len(arr) - 1 if not isinstance(right, (int, float)) else right
    if left < right:
        partition_idx = partition(arr, left, right)
        quick_sort(arr, left, partition_idx - 1)
        quick_sort(arr, partition_idx + 1, right)
    return arr
            
def gen_print_msg(func, arr, tag, info_type="info", runtime=None):
    if runtime:
        msg = "[%s] %s with Total: %s in %.5fs" %  (tag, func.__name__, len(arr), runtime)
    else:
        type_flag = "-->"
        if info_type in {"info", "result"}:
            arr_print = f" ${arr}"
            if info_type == "info":
                type_flag = "---"
        else:
            arr_print = ""
        msg = "%s [%s]%s, Use %s with Total: %s" % (type_flag, tag, arr_print, func.__name__, len(arr))
    print(msg)

def run_test(func, total=100):
    arr = [i for i in range(-math.floor(total / 2), math.ceil(total / 2))]
    arr_cp = copy.deepcopy(arr)
    while operator.eq(arr, arr_cp):
        random.shuffle(arr_cp)
    gen_print_msg(func, arr, "Origin list")
    gen_print_msg(func, arr_cp, "Random list", info_type="result")
    start_time = time.clock()
    arr_cp = func(arr_cp)
    end_time = time.clock()
    runtime = end_time - start_time
    gen_print_msg(func, arr_cp, "Sorted list", info_type="result")
    if operator.eq(arr, arr_cp):
        gen_print_msg(func, arr_cp, "Sucesss", runtime=runtime)
        return True
    else:
        gen_print_msg(func, arr_cp, "Fail", runtime=runtime)
        return False

def main():
    run_test(bubble_sort)
    run_test(selection_sort)
    run_test(insertion_sort)
    run_test(shell_sort)
    run_test(merge_sort)
    run_test(quick_sort)


if __name__ == '__main__':
    main()