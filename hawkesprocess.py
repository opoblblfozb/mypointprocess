import numpy as np

class HawkesProcess:
    ### should be modified
    ### 準ニュートン法あたり勉強しなおして，再実装
    def __init__(self):
        self.a = None
        self.b = None
        self.mu = None

    def poissonfit(self, data):
        lasttime = data[-1]
        datanum  = len(data)
        return datanum/lasttime
    
    def fit(self, data):
        nums = len(data)
        lasttime = data[-1]
        
        tmp = self.poissonfit(data)
        a = tmp
        b = tmp
        mu = tmp
        
        gi = [0 for _ in range(nums)]
        gib = [0 for _ in range(nums)]
        
        for i in range(nums-1):
            k = np.e**(-b*(data[i+1] - data[i]))
            gi[i+1] = (gi[i] + a*b)*k
            gib[i+1] = (gib[i+1] + a)*k - gi[i+1]*(data[i+1] - data[i])
        
        lamb = [gi[i] + mu for i in range(nums)]
        lamba = [gi[i]/a for i in range(nums)]
        lambb = [gib[i] for i in range(nums)]
        
        self.a = 0
        self.b = 0
        self.mu = mu
        for i in range(nums):
            ## add to a
            self.a += (1/lamb[i])*lamba[i]
            self.a -= 1 - np.e**(-b*(lasttime - data[i]))
            
            ## add to b
            self.b += (1/lamb[i])*lambb[i]
            self.b -= a*(lasttime-data[i])*np.e**(-b*(lasttime-data[i]))