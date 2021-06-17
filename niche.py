from population import *
#最终的小生境,由多个cluster组成。
class Niche:

    #clusters 由一个big和若干个small构成, cluster[0]:big  剩下的是small
    '''
    def __init__(self,match_list,func_name,lb,ub,budget):  #函数的lb ,ub
        self.func_name=func_name
        #self.clusters=clusters
        self.bc=match_list[0]
        self.scs=match_list[1:]
        self.set_range(lb,ub)
        self.d=len(lb)
        self.budget=budget
        self.used_budget=0
    '''

    def __init__(self, bc, cec, lb, ub, budget):  # 函数的lb ,ub
        self.cec = cec
        # self.clusters=clusters
        self.bc = bc
        self.set_range(lb, ub)
        self.d = len(lb)
        self.budget = budget
        self.used_budget = 0
    '''
    def set_range1(self,lb,ub):  #设定niche的管辖范围,矩形 不可以超过函数的定义域
        d=len(lb)
        solutions=self.bc.solutions
        #把所有的solution放一起
        solutions=solutions.reshape(-1,1) if solutions.ndim==1 else solutions
        m = np.min(solutions, axis=0)
        M = np.max(solutions, axis=0)
        std = np.std(solutions,axis=0)
        self.lb=[0]*d
        self.ub=[0]*d
        for i in range(d):
            self.lb[i]=max(lb[i],m[i]-5*std[i])
            self.ub[i] = min(ub[i], M[i]+5*std[i])
    '''
    def set_range(self,lb,ub):
        self.lb=lb
        self.ub=ub

    def printme(self):
        print(self.lb,self.ub)

    def evaluate(self,solutions):
        assert self.budget>=1
        fitness=np.array([])
        for x in solutions:
            self.budget-=1
            self.used_budget+=1
            fitness=np.append(fitness,self.cec.evaluate(x))
        return fitness
    def EA(self): #
        mu=self.bc.size
        tmp=mu
        solutions=self.bc.solutions
        fitness=self.bc.fitness
        ##print(fitness)
        offspring=np.zeros((mu,self.d))
        count = 0 #连续count代最佳fitness没有变化
        lr = 0.0001  # 对前三个很友好
        while(self.budget>0): #结束条件为连续5代 最佳fitness没有变化或者budget使用完
            ##print(f'used:{self.used_budget}')
            if(self.budget<mu):
                #print('budget not enough')
                tmp=self.budget #仅生成tmp个offspring
            ##print(f'f={fitness}')
            M=np.max(fitness) #用于与下一代种群max比较
            idx = np.argsort(-fitness)
            for c,i in enumerate(idx):
                if(c>=tmp):
                    break

                s=solutions[i]
                p_prime=solutions[idx[np.random.randint(c,len(idx),1)]]
                direction=s-p_prime
                direction+=0.1*np.random.randn(self.d)
                assert direction.ndim==2
                '''
                if(direction.ndim==1):
                    direction=direction/(abs(direction))
                '''
                #else:
                direction=direction/((direction@direction.T)**(1/2))
                t=s+lr*direction
                if(t.shape[1]>1):
                    t=t.squeeze()
                for j in range(self.d):
                    t[j] = min(t[j], self.ub[j])
                    t[j] = max(t[j], self.lb[j])
                offspring[i]=t
            total_solutions=np.append(solutions,offspring[0:tmp],axis=0)
            total_fitness=np.append(fitness,self.evaluate(offspring[0:tmp]))
            idx=np.argsort(-total_fitness)
            solutions=total_solutions[idx[0:mu]]
            fitness=total_fitness[idx[0:mu]]
            if(fitness[0]>M):
                lr*=2
                count=0
            else:
                lr=max(lr/10,0.0001)
                count+=1
                if(count==10):
                    return [solutions,fitness]
        ##print('budget exhausted for this niche')
        return [solutions,fitness]