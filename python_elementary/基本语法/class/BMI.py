class BMI:
    def __init__(self,height,weight):
        self.bmi = weight/(height*height)
    def calculate(self):
        print(f'Your BMI is: {self.bmi}')
class chinaBMI(BMI):
    def printBMI(self):
        if(self.bmi < 18.5):
            print('Your BMI is less than 18.5')
        elif(self.bmi < 25):
            print('Your BMI is less than 25')
        elif(self.bmi < 30):
            print('Your BMI is less than 30')
        elif(self.bmi < 35):
            print('Your BMI is less than 35')
        else:
            print('Your BMI is greater than 35')
h = float(input('Please enter your height in m: '))
w = float(input('Please enter your weight in kg: '))
calc = chinaBMI(h,w)
calc.calculate()
calc.printBMI()