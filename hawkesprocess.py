import numpy as np
import math
import time
import functools

class HawkesProcess:
    ## 勾配法が収束しない！
    def __init__(self):
        self.a = None
        self.b = None
        self.mu = None

    def poissonfit(self, data):
        lasttime = data[-1]
        datanum  = len(data)
        return datanum/lasttime
    
    def prepareG(self, a, b, mu, data):
        last = data[-1]
        n = len(data)
        ### prepare gi, gib
        gi = [0 for _ in range(n)]
        gib = [0 for _ in range(n)]
        
        for i in range(n-1):
            ex = np.e**(-b*(data[i+1] - data[i]))
            gi[i+1] = (gi[i] + a*b)*ex
            gib[i+1] = (gib[i] + a)*ex - gi[i+1]*(data[i+1] - data[i])
        return gi, gib
        
    def objectfunc(self, a, b, mu, data):
        # 線形オーダーの実装
        last = data[-1]
        n = len(data)
        gi, gib = self.prepareG(a, b, mu, data)
        
        ### caluculate obj
        p1 = 0
        p2 = mu*last
        
        for i in range(n):
            tmp1 = mu
            tmp1 += gi[i]
            print(tmp1)
            p1 += np.log(tmp1)
            p2 += a*(1-np.e**(-b*(last-data[i])))
        return p1 - p2
    
    def grad(self, a, b, mu, data):
        last = data[-1]
        n = len(data)
        gi, gib = self.prepareG(a, b, mu, data)
        
        nmu = -last
        na = 0
        nb = 0
        for i in range(n):
            nmu += (1/(gi[i]+mu))
            
            na += (1/(gi[i]+mu))*(gi[i]/a)
            na -= 1 - (np.e**(-b*(last-data[i])))
            
            nb += (1/(gi[i]+mu))*gib[i]
            na -= a*(last-data[i])*(np.e**(-b*(last-data[i])))    
        return [na, nb, nmu]
    
    def numerical_diff(self, f, x):
        h = 10**(-4)
        return (f(x+h) - f(x-h))/(2*h)
    
    def serch(self, func):
        x = 1
        epsilon = 0.00001
        if self.numerical_diff(func, x) < 0:
            h = -1
        else:
            h = 1
    
        while(abs(self.numerical_diff(func, x)) > epsilon):
            if numerical_diff(func, x)<0:
                h = - abs(h)
            else:
                h = abs(h)
            maex = x
            nextx = x+h
        
            # step3
            if (func(maex) < func(nextx)):
                while(func(maex) < func(nextx)):
                    h *= 2
                    maex = nextx
                    nextx = maex + h
                x = maex
                h = h/2
            # step4
            else:
                while(func(maex)>=func(nextx)):
                    h = h/2
                    nextx -= h
                x = nextx
                h *= 2
        # step 6
        return x


    def serchline(self, t, a, b, mu, na, nb, nmu, data):
        return self.objectfunc(a=a+t*na, b=b+t*nb, mu=mu+t*nmu, data=data)

    def serchline_pertial(self, ta, tb, tmu, tna, tnb, tnmu, tdata):
        return functools.partial(self.serchline, a=ta, b=tb, mu=tmu, na=tna, \
                                            nb=tnb, nmu=tnmu, data=tdata)


    def fit(self, data):
        epsilon = 10**(-4)
        tmp = self.poissonfit(data)
        a, b, mu = [tmp for _ in range(3)]
        
        while True:
            na, nb, nmu = self.grad(a, b, mu, data)
            t = self.serch(self.serchline_pertial(ta=a, tb=b, tmu=mu, tna=na, \
                                            tnb=nb, tnmu=nmu, tdata=data))
            
            da = t*na
            db = t*nb
            dmu = t*nmu
            a = a + da
            b = b + db
            mu = mu + dmu
            
            if np.sqrt(da**2 + db**2 + dmu**2) < epsilon:
                break
        self.a = a
        self.b = b
        self.mu = mu


        
    '''
    二乗オーダーの実装
    def objectfunc(a, b, mu, data):
        last = data[-1]
        n = len(data)
        p1 = 0
        p2 = last*mu
        
        for i in range(n):
            ### about p1
            tmp1 = mu
            for j in range(0, i):
                tmp1 += a*b*np.e**(-b*(data[i]-data[j]))
            p1 += math.log(tmp1)
            
            ### about p2
            p2 += a*(1-np.e**(-b*(last-data[i])))
        return p1 - p2 
    '''