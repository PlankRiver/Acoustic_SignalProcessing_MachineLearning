#子类
class dog:
    count = 0
    def __init__(self,name):
        self.count += 1
        self.name=name
    def greet(self):
        print(f'Hi! My name is No.{dog.count} {self.name}')
class barkingdog(dog):
    def bark(self):
        print('barking')
maggie = barkingdog('pussy')
maggie.greet()
maggie.bark()

#重写1
# class dog:
#     count = 0
#     def __init__(self,name):
#         self.count += 1
#         self.name=name
#     def greet(self):
#         print(f'Hi! My name is No.{dog.count} {self.name}')
# class barkingdog(dog):
#     def bark(self):
#         print('barking')
#     def greet(self):
#         print(f'Wolf! My name is No.{dog.count} {self.name}')
# maggie = barkingdog('pussy')
# maggie.greet()
# maggie.bark()
# dog.greet(maggie)

#重写2
# class dog:
#     count = 0
#     def __init__(self,name):
#         self.count += 1
#         self.name=name
#     def greet(self):
#         print(f'Hi! My name is No.{dog.count} {self.name}')
# class barkingdog(dog):
#     def __init__(self,name):
#         self.name = 'Little '+ name
# x = barkingdog('fuck')
# x.greet()
# class barkingdog(dog):
#     def __init__(self,name):
#         self.name = name
#         print('My name is',self.name)
# x = barkingdog('Bob')
# class barkingdog(dog):
#     def __init__(self,name):
#         super().__init__(name)
#         print('My name is',self.name)
# x = barkingdog('Bob')

#super()方法
# class P_father:
#     def __init__(self,a,b):
#         self.a = a
#         self.b = b
#     def pnt(self):
#         print(self.a+self.b)
# class P_child(P_father):
#     def pnt(self):
#         super().pnt()
#         print(self.a*self.b)
# calc = P_child(3,5)
# calc.pnt()