#最小的数
def min_num(args):
    sum = 0
    args.sort()
    if args[0]==0:
        for j in range(len(args)):
            if args[j]!=0:
                args[0],args[j]=args[j],args[0]
                break
    for i in args:
        sum*=10
        sum+=i
    return sum
print(min_num(list(eval(input()))))
