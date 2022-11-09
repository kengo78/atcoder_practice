import sys
# import collections
from collections import defaultdict, Counter
# Re
# def main(lines):
#     n,m = map(int, lines[0].split(' '))
#     # keys = [i for i in range(0, m)]
#     result = defaultdict(int)
#     for i,v in enumerate(lines[1:]):
#         for j,k in enumerate(v):
#             if k == '#':
#                 result[j] += 1
#             else:
#                 result.setdefault(j, 0)
#     result = sorted(result.items(), key=lambda i: i[0])
#     result = [str(i[1]) for i in result]
#     print(' '.join(result))
    
if __name__ == '__main__':
    n = int(input())
    m = list(map(int, input().split(' ')))
    result = [0] * (2 * n +1)
    for i,v in enumerate(m):
        result[2*i+1]=result[v-1]+1
        result[2*i+2]=result[v-1]+1
        
    print(*result,sep="\n")