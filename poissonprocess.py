import numpy as np

class PoissonProcess:
    def __init__(self):
        self.lam = None
    
    def checkascend(self, data):
        nums = len(data)
        for i in range(nums-1):
            if data[i] > data[i+1]:
                print('昇順に並べ替えます')
                return True
        return False
    
    def fit(self, data):
        check = self.checkascend(data)
        if check:
            data = sorted(data)
        
        lasttime = data[-1]
        datanum  = len(data)
        
        self.lam = datanum/lasttime