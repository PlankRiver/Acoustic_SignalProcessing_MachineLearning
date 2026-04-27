#random
import random

rand_float = random.random()  # [0.0, 1.0)
print(rand_float)

rand_uniform = random.uniform(1.5, 10.5)  # [1.5, 10.5]
print(rand_uniform)

rand_int = random.randint(1, 10)  # [1, 10] 包含两端
print(rand_int)

rand_range = random.randrange(0, 100, 5)  # 从0到100(不包含)，步长为5
print(rand_range)

colors = ['red', 'green', 'blue', 'yellow']
choice = random.choice(colors)
print(choice)

sample = random.sample(colors, 2)  # 从colors中选2个不重复的元素
print(sample)

random.shuffle(colors)  # 直接修改原列表
print(colors)

gauss_num = random.gauss(0, 1)  # 均值为0，标准差为1
print(gauss_num)

normal_num = random.normalvariate(0, 1)  # 同上
print(normal_num)

exp_num = random.expovariate(1.0)  # 参数λ=1.0
print(exp_num)

random.seed(42)  # 设置随机种子为42

# 现在每次运行程序都会得到相同的随机序列
print(random.random())
print(random.randint(1, 100))

#生成随机密码
import string

def generate_password(length=8):
    chars = string.ascii_letters + string.digits + '!@#$%^&*'
    return ''.join(random.choice(chars) for _ in range(length))

print(generate_password(12))

#生成随机验证码
def generate_verification_code(length=6):
    return ''.join(random.choice('0123456789') for _ in range(length))

print(generate_verification_code())

#加权随机选择
def weighted_random_choice(elements, weights):
    return random.choices(elements, weights=weights, k=1)[0]

# 测试
elements = ['A', 'B', 'C']
weights = [0.5, 0.3, 0.2]  # A有50%概率，B30%，C20%
print(weighted_random_choice(elements, weights))



#random库函数伪随机特点：梅森旋转算法
class MersenneTwister:
    def __init__(self, seed=5489):
        # MT19937常数
        self.n = 624
        self.m = 397
        self.matrix_a = 0x9908b0df
        self.upper_mask = 0x80000000
        self.lower_mask = 0x7fffffff
        
        self.mt = [0] * self.n
        self.index = self.n
        self.mt[0] = seed
        
        # 初始化状态数组
        for i in range(1, self.n):
            self.mt[i] = (1812433253 * (self.mt[i-1] ^ (self.mt[i-1] >> 30)) + i) & 0xffffffff
    
    def extract_number(self):
        if self.index >= self.n:
            self._twist()
        
        y = self.mt[self.index]
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= (y >> 18)
        
        self.index += 1
        return y & 0xffffffff
    
    def _twist(self):
        for i in range(self.n):
            x = (self.mt[i] & self.upper_mask) | (self.mt[(i+1) % self.n] & self.lower_mask)
            self.mt[i] = self.mt[(i + self.m) % self.n] ^ (x >> 1)
            if x % 2 != 0:
                self.mt[i] ^= self.matrix_a
        self.index = 0

# 使用示例
mt = MersenneTwister(1234)
for _ in range(5):
    print(mt.extract_number())