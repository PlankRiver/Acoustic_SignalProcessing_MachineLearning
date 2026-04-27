#searching for the longest word
def findthelongestword(words):
    max_length = len(sorted(words,key=len)[-1])
    res = sorted(set(filter(lambda word:len(word)==max_length,words)),key=words.index)
    return res
s = input()
for i in ['.',',','!','?','...']:
    s = s.replace(i,' ')
s = s.lower().split()
result = findthelongestword(s)
print(','.join(result))