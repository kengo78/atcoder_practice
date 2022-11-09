import sys

def main(lines):
    S = lines[0]
    rev_sentence = S[::-1]
    try:
        idx = rev_sentence.index('a')
        print(len(S) - idx)
    except ValueError:
        print(-1)
if __name__ == '__main__':
    lines = []
    for l in sys.stdin:
        lines.append(l.rstrip('\r\n'))
    main(lines)