
#liar-island
#Xisland always says the truth,while Yisland always tend to lie
for a in range(2):
    for b in range(2):
        for c in range(2):
            if((a==1 and a+b+c==2) or (a==0 and a+b+c!=2)) and ((b==1 and a+b+c==1) or (b==0 and a+b+c!=1)) and ((c==1 and a+b+c==1) or (c==0 and a+b+c!=1)):
                print(a,b,c,sep=' ')