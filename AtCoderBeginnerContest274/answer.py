# import sys
# # import collections
# from collections import defaultdict, Counter
# # Re
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
    # lines = []
    # for l in sys.stdin:
    #     lines.append(l.rstrip('\r\n'))
    # main(lines)
    n, m = list(map(int, input().split( )))
    grid = [input() for _ in range(n)]
    print('grid', grid, type(grid))
    print('*grid', *grid)
    res = [''.join(col).count('#') for col in zip(*grid)]
    print(' '.join(str(v) for v in res))