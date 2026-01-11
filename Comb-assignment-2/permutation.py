
#第一题生成全排列的算法

# permutation.py

def trotter_johnson_unrank(n: int, r: int) -> list:
    """n:size  r:rank"""
    perm = [1]
    for j in range(2, n + 1):
        r_prev = r // j
        k = r % j

        if r_prev % 2 == 0:
            pos = j - k - 1
        else:
            pos = k
        perm.insert(pos, j)
        r = r_prev
    return perm



