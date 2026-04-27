# cnt = 0
# def hanoi(n,a,b,c):
#     global cnt
#     if n==1:
#         print(f'Move the tower from {a} to {c}')
#         cnt+=1
#         return
#     hanoi(n-1,a,c,b)
#     print(f'Move the tower from {a} to {c}')
#     cnt+=1
#     hanoi(n-1,b,a,c)
# if __name__=='__main__':
#     n = int(input('Please enter the number of moves: '))
#     hanoi(n,'a','b','c')
#     print(cnt)


#another way
def hanoi(n,a,b,c,count=0):
    if n==1:
        print(f'Move the tower from {a} to {c}')
        return count+1
    count=hanoi(n-1,a,c,b,count)
    print(f'Move the tower from {a} to {c}')
    count=hanoi(n-1,b,a,c,count+1)
    return count
if __name__ == '__main__':
    n = int(input())
    print(hanoi(n,'a','b','c',count=0))