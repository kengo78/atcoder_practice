import sys
# Re
def main(lines):
    a,b = map(float, lines[0].split(' '))
    result = round(b/a, 3)
    # if len(str(result)) == 5:
    #     print(result)
    # else:
    #     while len(str(result)) != 5:
    #         result = str(result) + '0'
    #     print(result)
    print("%.3f"%(b/a))
    
if __name__ == '__main__':
    lines = []
    for l in sys.stdin:
        lines.append(l.rstrip('\r\n'))
    main(lines)