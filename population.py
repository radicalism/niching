import numpy as np
from functions import *
import matplotlib.pyplot as plt
import time
from cfunction import *
class Population:

    class Cluster:
        # population中的某一块区域
        def __init__(self, solutions,fitness):
            assert fitness.ndim==1
            self.solutions = solutions
            self.fitness=fitness.squeeze()
            if(self.fitness.ndim==0):
                self.fitness=self.fitness.reshape(1,)
            assert isinstance(fitness,np.ndarray)
            self.size = solutions.shape[0]
            self.center = np.mean(self.solutions, axis=0)
            ###print(f'new clister:{self.solutions}')

        def merge(self, another):
            assert isinstance(another, Population.Cluster)
            ###print(f'{self.solutions}\n{another.solutions}')
            self.solutions=np.append(self.solutions,another.solutions,axis=0)
            if(self.solutions.ndim==1):
                self.solutions=self.solutions.reshape(-1,1)
            self.fitness=np.append(self.fitness,another.fitness)
            ###print(self.solutions)
            self.size += another.size
            ###print(self.solutions)
            assert (self.size==len(self.solutions))
            self.center = np.mean(self.solutions, axis=0)



        def printme(self):
            print(f'solotions={self.solutions}\ncenter={self.center}')


    def __init__(self,lb,ub,mu,d,cec):
        self.cec=cec
        self.init_solutions(lb,ub,mu,d) #初始化种群
        self.evaluate() #计算种群的适应度
        big, small = self.devided_into_big_small()
        dis_mat=self.cal_distance() #计算距离（解空间）
        self.init_clusters(dis_mat,big,small) #对种群按适应度大小分为big,small  并对big做聚类
        self.showCluster()
        #self.merge_among_big()
        #self.merge_among_small()
        #不必对small做merge


    def init_solutions(self,lb:list,ub:list,mu:int,d:int):
        t0 = time.time()
        self.lb=lb
        self.ub=ub
        self.mu=mu #种群大小 重要
        self.d=d
        solutions = np.zeros((self.mu, self.d))
        ##print(lb,ub)
        ##print(f'd={self.d}')
        for c in range(self.d):
            ##print(f'c={c}')
            t=np.random.uniform(lb[c], ub[c], self.mu)
            ##print(solutions[:,c],t)
            solutions[:, c] = t
        self.solutions = solutions
        ##print(f'init solutions end {time.time() - t0}used')
        ###print(f'init_pop={self.solutions}')
    #计算解之间的距离,聚类时会用到
    def cal_distance(self):
        t0 = time.time()
        G=np.dot(self.solutions,self.solutions.T)
        H=np.tile(np.diag(G),(self.mu,1))
        ##print(H.shape)
        ##print(f'cal_distance end {time.time() - t0}used')
        return H+H.T-2*G


    def evaluate(self):
        fitness=np.array([])
        for x in self.solutions:
            fitness=np.append(fitness,float(self.cec.evaluate(x)))
        assert fitness.ndim==1
        self.fitness=fitness

    def devided_into_big_small(self):
        sorted_fitness=sorted(self.fitness,reverse=True)
        threshold=sorted_fitness[len(self.fitness)//20] #前1/25进入big
        threshold1=sorted_fitness[len(self.fitness)//20]
        #print(threshold,threshold1)
        t0=time.time()
        #threshold=100
        ###print(f'fit={self.fitness}')
        ###print(f'th={threshold}')
        big = [] #idx of big solution in pop
        small = [] #idx of small solution in pop
        for i, f in enumerate(self.fitness):
            if (f >= threshold):
                big.append(i)
            else:
                small.append(i)
        big=np.array(big)
        #print(f'before enlarge,big.size={len(big)}')
        #对big内的点进行繁殖，增加个数（有助于提升聚类效果）
        new_big=big.copy()
        k=50 #增殖倍数
        self.mu += k*len(big)
        offspring=np.zeros((k*len(big),self.d))
        for i,b in enumerate(big):
            for t in range(k):
                p=self.solutions[b]
                ind=p+np.random.randn(self.d)
                for z in range(self.d):
                    ind[z]=min(self.ub[z],ind[z])
                    ind[z]=max(self.lb[z],ind[z])
                offspring[i*k+t]=ind
                fit=self.cec.evaluate(ind)
                self.fitness=np.append(self.fitness,fit)
                if(fit>threshold1):
                    ##print('yes')
                    new_big=np.append(new_big,len(self.solutions)+i*k+t)
        big=new_big
        small=np.array(small)
        self.solutions=np.append(self.solutions,offspring,axis=0)
        ###print(f'big={big}\nsmall={small}')
        #self.small=small
        ##print(f'big small get,{time.time()-t0}used')
        return big,small
    '''
    clustering
    '''
    #input:#idx of big(small) solution in pop
    #对big或者small初步聚类
    #返回clusters:list
    #clusters=[cluster_i] i=1,2,...,n
    #cluster_i:list<idx>

    '''
    big: indices of big ind
    '''

    def clustering_0(self,big,small,dis_mat):  # 新版聚类
        t0=time.time()
        ##print('clustering begin')
        clusters = []  # 记录所有的簇 这里的每一个簇为solution的集合，并非一个类
        visited = np.array([False] * self.mu)
        while(not all(visited[big])):
            solutions = []  # 记录某一个cluster中的个体的下标
            q=[]
            M=-1e4
            M_idx=-1
            for b in big:
                if(not visited[b] and self.fitness[b]>M):
                    M=self.fitness[b]
                    M_idx=b
            q.append(M_idx)
            #每次从fitness最大的个体出发
            while(not len(q)==0):
                t=q[0]
                visited[t]=True
                ##print(f'vis={sum(visited)}')
                solutions.append(t)
                q.remove(t)
                idx=np.argsort(dis_mat[t])
                for i in idx:
                    if(i in big and visited[i]==False):
                        q.append(i)
                        ##print(f'q={q}')
                        visited[i]=True
                    elif(i in small):
                        break
                    else:
                        continue
            clusters.append(solutions.copy())
        ##print('clustering end')
        ##print(f'{time.time()-t0}used')
        return clusters

    def init_clusters(self,dis_mat,big,small):

        clusters_of_big = self.clustering_0(big,small,dis_mat)  # list<idx>
        clusters_of_big.sort(key=lambda x:np.max(self.fitness[x]),reverse=True)#按MAX适应度大小降序排列

        '''
        for x in clusters_of_big:
            #print(np.mean(self.fitness[x]))
        '''

        #clusters_of_small = self.clustering_0(small, big,dis_mat)  # list<idx>
        ###print(clusters_of_big)
        #clusters_of_small = self.clustering(small,dis_mat)
        self.clusters_of_big=[]  #里面存的是cluster类
        for c in clusters_of_big:
            if(len(c)<3): continue
            cluster=Population.Cluster(self.solutions[c],self.fitness[c])
            self.clusters_of_big.append(cluster)
        '''
        self.clusters_of_small = []
        
        
        for c in clusters_of_small:
            cluster = Population.Cluster(self.solutions[c],self.fitness[c])
            self.clusters_of_small.append(cluster)
        '''
    '''
    
    def match(self):  #small 也聚类时使用
        match_lists=[]
        for bc in self.clusters_of_big:
            ##print(f'bc.size={len(bc.solutions)}')
            match_list=[]
            match_list.append(bc)
            dis=[0]*len(self.clusters_of_small)
            for i,sc in enumerate(self.clusters_of_small):
                dis[i]=((bc.center-sc.center)@(bc.center-sc.center))**(1/2)
            idx=np.argsort(dis) #升序
            for j in range(min(2*self.d,len(self.clusters_of_small))):
                match_list.append(self.clusters_of_small[idx[j]])
            match_lists.append(match_list)
        self.match_lists=match_lists
        '''


    def showCluster(self):
        if(self.d==1):
            ##print('p')
            fig=plt.figure()
            ax=fig.add_subplot(1,1,1)
            for clus in self.clusters_of_big:
                ax.scatter(clus.solutions, clus.fitness)
            #ax.scatter(self.solutions[self.small], self.fitness[self.small])
            plt.show()

        elif(self.d==2):
            fig=plt.figure()
            ax=fig.add_subplot(111,projection='3d')
            for clus in self.clusters_of_big:
                x = clus.solutions[:, 0]
                y = clus.solutions[:, 1]
                ax.scatter3D(x,y,clus.fitness)
            #ax.scatter3D(self.solutions[self.small][:,0],self.solutions[self.small][:,1], self.fitness[self.small])
            plt.show()
            pass

        else:
            #print('dimension>2d')
            pass


    def printme(self):
        print(f'lb=\n{self.lb}\nub=\n{self.ub}\nindividuals=\n{self.solutions}\nfitness=\n{self.fitness} ')
'''
#聚类效果可视化
'''




