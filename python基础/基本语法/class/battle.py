class person:
    count = 0
    def __init__(self,name,gender,age,fight_value):
        person.count += 1
        self.name = name
        self.gender = gender
        self.age = age
        self.fig = fight_value
    def battle(self):
        self.fig-=100
    def practice(self):
        self.fig+=200
    def eating(self):
        self.fig+=80
    def info(self):
        print(f'Hi! My name is No.{person.count} {self.name},I have {self.fig} fight values.')
player1 = person('John','male',23,100)
player1.info()
player1.battle()
player1.info()
player1.practice()
player1.eating()
player1.info()
player2 = person('Pheobe','female',23,100)
player2.info()
player2.battle()
player2.info()
player2.practice()
player2.eating()
player2.info()