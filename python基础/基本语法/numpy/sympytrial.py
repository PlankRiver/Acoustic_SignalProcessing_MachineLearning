from sympy import *
#from numpy import * 会与sympy发生冲突
x = Symbol('x')
expr = x*log(1/x+2**0.5)
result = limit(expr,x,oo)
print(result)