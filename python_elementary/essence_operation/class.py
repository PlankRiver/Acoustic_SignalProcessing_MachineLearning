class Animal:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def eat(self):
        print("eating...")

class Dog(Animal):
    species = 'mammal'
    def __init__(self,name,age,breed):
        super().__init__(name,age)
        self.breed = breed
    def __str__(self):
        return f'My name is {self.name}.\nI am {self.age} years\'old.\nI am a {self.breed}.'
    def bark(self):
        print('Woof! Woof!')
    def fetch(self):
        print('jump!')
    def birthday(self):
        self.age += 1
        print(f'This is {self.name}\'s {self.age} birthday!')

dog1 = Dog('chengao',8,'meiyouba')
dog2 = Dog('jiaping',10,'shabi')
dog1.fetch()
dog2.bark()
dog1.birthday()
dog1.birthday()
print(dog1)