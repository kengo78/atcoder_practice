import sys

# Re
def main(n):
    if n == 0:
        return 1
    return n * main(n-1)
    
    
if __name__ == '__main__':
    n = int(input())
    print(main(n))