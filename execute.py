from population import *
from niche import *
from functions import *
from cec2013 import *
import os
class Execute:
    def __init__(self, lb , ub, mu, d, cec, max_eva):
        self.lb=lb
        self.ub=ub
        self.d=d
        self.mu=mu
        self.cec=cec
        self.max_eva=max_eva
    def __call__(self,lb,ub):
        pop=Population(lb, ub, self.mu, self.d, self.cec)
        return pop

    #为每一个bc 匹配2d个 sc
    #return list[list] 内部的每一个list包含一个bc和2d个sc


    #将搜索空间划分区域
    def split_region(self):
        lb=self.lb
        ub=self.ub
        mid=(np.asarray(lb)+np.asarray(ub))/2
        d=len(lb)
        t=[lb,mid,ub]
        lbs=[]
        ubs=[]
        for i in range(2**d):
            lb=[]
            ub=[]
            tmp=i
            bin_of_i=[0]*d
            pos=d-1
            while(not tmp==0):
                bin_of_i[pos]=tmp%2
                pos-=1
                tmp//=2
            for m,n in enumerate(bin_of_i): #第m个维度来自于t[n]的第m个维度
                lb.append(t[n][m])
                ub.append(t[n+1][m])
            lbs.append(lb.copy())
            ubs.append(ub.copy())
        return lbs,ubs


    def run_on_one_region(self,lb,ub,max_eva): #区域内每一个Niche进行搜索
        #print(f'for this region budget={max_eva}')

        #分n^d份 即每个维度均分为n份
        #每个子问题的种群大小可以不一样的哦  自适应 目的是把budgety用完 但是就先固定吧
        #每个部分的评估次数也应该是不一样的，怎么弄还没想好

        archive=[]
        fitness=np.array([])
        pop=self(lb,ub) #init pop
        budget_letf=max_eva-len(pop.solutions) #该region剩下的评估次数
        #print(f'after init pop{budget_letf} left')
        #pop.match()

        #生成若干niche
        for bc in pop.clusters_of_big:
            ##print('??')
            niche=Niche(bc,self.cec,lb,ub,budget_letf)
            [arc,fit]=niche.EA()
            #print('EA end')
            budget_letf-=niche.used_budget
            #print(f'{budget_letf} budget left')
            assert niche.budget>=0
            budget_from_last=niche.budget #
            archive.append(arc)
            ##print(f'this regions arc={archive}')
            fitness=np.append(fitness,fit)
        #print(f'for this region {budget_letf} budgets left')
        return archive,fitness,budget_letf

    def run(self):
        archive=[]
        fitness=np.array([])
        lbs,ubs=self.split_region()
        ave_eva=self.max_eva//len(lbs) #每一个区域先平分budget
        extra_eva=0 #之前的区域剩下的budget
        for i in range(len(lbs)):
            #print(f'run on region{i}\nfor this region lb={lbs[i]} ub={ubs[i]}')
            arc,fit,budget_left=self.run_on_one_region(lbs[i],ubs[i],ave_eva+extra_eva)
            #print(f'region {i} end')
            #print('#'*100)
            assert budget_left>=0
            extra_eva=budget_left
            ##print(arc)
            archive.append(arc.copy())
            fitness=np.append(fitness,fit.copy())
        archive=np.array(archive,dtype=object)
        return archive,fitness

'''
把fitness最高的取出来 可以往下浮动1e-1
'''
def proess(archive,fitness): #对run的结果进行处理
    ##print(archive)
    ##print(fitness)
    solutions=[]
    best=[]
    for niches in archive:
        for niche in niches:
            for s in niche:
                solutions.append(s)
    idx=np.argsort(-fitness)
    M=fitness[idx[0]]
    ##print(M)
    for i in idx:
        if(M-fitness[i]<=1e-1):
            best.append(list(solutions[i]))
    return best,M


def cal_pr_sr(cec:CEC2013,nr,epsilon): #nr number run
    found=[0]*len(epsilon) #对每一个精度 找到的最优个数
    nsr=[0]*len(epsilon) #对每一个精度 , 找到所有的次数
    nkp=cec.get_no_goptima()
    lb = cec.get_lbound()
    ub = cec.get_ubound()
    # lb=[-10,-10]
    # ub=[0,0]
    max_eva = cec.budget[cec.func_idx - 1]
    # #print(func_name,lb,ub)
    d = len(lb)
    if(d==1):
        mu=500
    elif(d==10 or d==20):
        mu=100
    elif(cec.func_idx in [4,5]):
        mu=1000
    else:
        mu=5000
    for i in range(nr):
        #===================================找到解================
        # #print(func_name)

        exec = Execute(lb, ub,mu, d, cec, max_eva)
        arc, fit = exec.run()
        best_ind, best_fitness = proess(arc, fit)
        X = np.array(best_ind)
        fits = np.zeros(len(X))
        for k in range(len(X)):
            fits[k] = cec.evaluate(X[k])

        # Descenting sorting
        order = np.argsort(-fits)
        sorted_X = X[order, :]
        fitness=fits[order]
        directory = f'C:\\Users\\xy\\Desktop\\result_of_hw3\\f{cec.func_idx}'
        if(not os.path.exists(directory)):
            os.mkdir(directory)
        ##print(f'd={directory}')
        file_name1 = f'run_{i+1}_acc'
        file_name2 = 'pr'
        file_name3 = 'sr'
        # ===================================找到解================
        for j,e in enumerate(epsilon):
            count, seeds = how_many_goptima(sorted_X,fitness, cec, e)
            ##print(type(directory+'\\'+file_name1+f'acc_{j}.txt'))
            path1=directory+'\\'+file_name1+f'{j}.txt'
            if(e==1e-1):
                if (not os.path.exists(path1)):
                    #print('cre')
                    fp = open(path1,'w')
                    fp.close()
                np.savetxt(path1,seeds)
            if(count==nkp):
                nsr[j]+=1
            found[j]+=count
    pr=np.array(found)/(nkp*nr)
    sr=np.array(nsr)/nr
    path2=directory + '\\' + file_name2+ '.txt'
    f2=open(path2,'w')
    f2.close()
    np.savetxt(path2, pr)
    path3 = directory + '\\' + file_name3 + '.txt'
    f3 = open(path3, 'w')
    f3.close()
    np.savetxt( path3,sr)
    return pr,sr





'''
F8 很垃圾

'''
epsilon=[1e-1,1e-2,1e-3,1e-4,1e-5]
'''

for func_idx in range(1,2,1):
    cec=CEC2013(func_idx+1)
    pr,sr=cal_pr_sr(cec,1,epsilon)
    #print(f'{func_idx + 1} end')
#print(f'pr={pr}\nsr={sr}')
'''
cec=CEC2013(1)
pr,sr=cal_pr_sr(cec,1,epsilon)