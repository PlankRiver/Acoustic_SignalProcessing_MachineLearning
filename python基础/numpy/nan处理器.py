import numpy as np

def nan_fill(t1):
    for i in range(t1.shape[1]):
        temp = t1[:,i]
        nan_cnt = np.count_nonzero(temp!=temp)
        if nan_cnt > 0:
            temp_without_nan = temp[temp==temp]
            temp[temp!=temp] = temp_without_nan.mean()
            t1[:,i] = temp
    return t1

if __name__ == '__main__':
    a = np.arange(24).astype(float).reshape(4,6)
    a[1,2:] = np.nan
    print(a)
    print('*'*20)
    print(nan_fill(a))