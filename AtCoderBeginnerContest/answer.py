import sys
import itertools
# Re
def main(lines):
    a,b = map(float, lines[0].split(' '))
    print(round(b/a, 3))
    
if __name__ == '__main__':
    lines = []
    for l in sys.stdin:
        lines.append(l.rstrip('\r\n'))
    main(lines)