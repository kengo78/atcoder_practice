import sys
def main(lines):
    n= map(int, lines[0].split(' '))
    lines = lines[1:]
    record = {}
    for i in range(m):
        record.setdefault(str(i+1), [])
    for i,v in enumerate(lines):
        a,b = map(int, v.split(' '))
        record[str(a)].append(str(b))
        record[str(b)].append(str(a))
    for i in record.keys():
        record[i].sort()
        # if record[i] == []:
        print(str(len(record[i])),' '.join(record[i]))
if __name__ == '__main__':
    lines = []
    for l in sys.stdin:
        lines.append(l.rstrip('\r\n'))
    main(lines)