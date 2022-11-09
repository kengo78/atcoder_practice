import sys
import itertools
# Re
def main(lines):
    n = int(lines[0])
    s_list = tuple(lines[1].split(' '))
    tuple_list = sorted(list(itertools.permutations(s_list)))
    K = tuple_list.index(s_list)
    print(' '.join(tuple_list[K-1]))
    
if __name__ == '__main__':
    lines = []
    for l in sys.stdin:
        lines.append(l.rstrip('\r\n'))
    main(lines)