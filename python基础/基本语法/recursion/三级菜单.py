#三级菜单
dic = {
    '北京':{
        '海淀':{
            '东门':{
                '爱奇艺':{},
                '腾讯':{}
            },
            '西门':{
                '地平线':{},
                '学而思':{}
            },
            '北门':{
                '新东方':{},
                '学大尾页':{}
            }
        },
        '朝阳':{
            '南':{},
            '北':{}
        },
        '门头沟':{}
    },
    '上海':{
        '浦东':{},
        '闵行':{},
        '徐汇':{}
    },
    '山东':{}
}

def threelist(dic):
    while(True):
        for k in dic:print(k)
        key = input('>>>')
        if key=='back' or key=='quit':return key
        elif key in dic.keys() and dic[key]:
            ret = threelist(dic[key])
            if ret=='quit':
                return ret
threelist(dic)
#真他妈牛逼


#binarysearch1.0
# def binarysearch(lst,aim):
#     if lst==[]:
#         print('not found')
#         return
#     mid = (len(lst)-1)//2
#     if aim>lst[mid]:
#         return binarysearch(lst[mid+1:],aim)
#     elif aim<lst[mid]:
#         return binarysearch(lst[:mid],aim)
#     else:
#         print(f'Find it! It is {mid}')
#         return
# arr = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
# aim = int(input())
# binarysearch(arr,aim)
#有点问题，没法返回原列表的索引

#binarysearch2.0
# def binarysearch(lst,aim,start=None,end=None):
#     start = start if start!=None else 0
#     end = end if end!=None else len(lst)-1
#     mid = (start+end)//2
#     if start>end:
#         return None
#     elif lst[mid]>aim:
#         return binarysearch(lst,aim,start,mid-1)
#     elif lst[mid]<aim:
#         return binarysearch(lst,aim,mid+1,end)
#     elif lst[mid]==aim:
#         return mid
# lst = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
# aim = int(input('Please input a number'))
# print(binarysearch(lst,aim))