import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class EventData:
    # 初始化EventData类，传入core、theta、data三个参数
    def __init__(self, core: np.array, theta: float, E: float, data: pd.DataFrame):
        self.__core = core
        self.__theta = theta
        self.__data = data
        self.__E = E

    # 显示EventData类的core和theta属性
    def display(self):
        print(f"Core= {self.__core}; Theta= {self.__theta}; E= {self.__E}")
        print(f"Detector Data: \n {self.__data}")

    # 返回EventData类的core属性
    def get_core(self):
        return self.__core

    # 返回EventData类的theta属性
    def get_theta(self):
        return self.__theta
    
    # 返回EventData类的E属性
    def get_E(self):
        return self.__E

    # 返回EventData类的data属性
    def get_data(self):
        return self.__data


def loadData_1_event(data_path,id,):  # 读取txt文件中的一个 event_id = id  的数据
    data=[]
    core=[]
    theta=100.
    end=0
    data_path=data_path.strip()
    find_content=f"EVENT"
    find=False
    with open(data_path, 'r') as file:
        event_id=0
        for line in file:
            if line.startswith(find_content):
                content=line.strip().split()
                event_id=int(content[1])
                
                if event_id==id:
                    find=True
                    E=float(content[-6])  # 能量 (GeV)
                    theta=float(content[-4]) # 角度 (度)
                    core=[content[-2],content[-1]] # 中心(距离探测阵列中心的距离,单位米)
                    continue
                elif event_id > id:
                    #print(f"No EVENT with id = {id}")
                    return None, end

            if find:
                row=line.strip().split()
                if len(row)==0:
                    break               
                data.append(row)
        
        if event_id < id :
            print('end of file')
            return  None, 1
        
   
    core=np.float32(core)
    data=np.float32(data)
    E=np.float32(E)
#    HitsE.pe:     number of estimated pe photons from wave
#    HitsE.np:     number of pe photons, -1 for noise hit  
    list_columns=['EDid' ,   'EDx',    'EDy' ,   'EDz'   , 'EDpe'  ,  'EDnp' ,   'tED' ]
    data=pd.DataFrame(data,columns=list_columns)
    data = data[data['EDnp'] != -1]  # 去除噪声点

    data=data.drop(columns=['EDnp','EDz'])
    #统计重复激发的探测器
    data = data.groupby('EDid').agg({
                            'EDpe': 'sum',
                            'EDx': 'first',  # 保留分组中的第一个值
                            'EDy': 'first',
                            'tED': 'mean',
                            
                        }).reset_index()
    
    return EventData(core,theta,E,data),end
    

def loadAll_Detectors(ED_pos_path,MD_pos_path,df=False): 
    if df:
        list_columns="id x y z"
        list_columns=list_columns.strip().split()
        ED_pos=pd.read_csv(ED_pos_path,skiprows=[0],sep=' ')
        MD_pos=pd.read_csv(MD_pos_path,skiprows=[0],sep=' ')
        ED_pos=ED_pos.astype(float)
        MD_pos=MD_pos.astype(float)
        ED_pos["pe"]=None
        MD_pos["pe"]=None
    else:
        ED_pos=np.genfromtxt(ED_pos_path, delimiter=' ', skip_header=2)
        MD_pos=np.genfromtxt(MD_pos_path, delimiter=' ', skip_header=2)
        
    return ED_pos,MD_pos

def path_creator(dir,num_list,): # 用于批量读取数据
    path_ED,path_MD= [],[]
    for n in num_list:
        if n < 10:
            path_ED.append(f'{dir}DAT00000{n}_ED.txt')
            path_MD.append(f'{dir}DAT00000{n}_MD.txt')
        elif (n >= 10) and (n < 100):
            path_ED.append(f'{dir}DAT0000{n}_ED.txt')
            path_MD.append(f'{dir}DAT0000{n}_MD.txt')
        elif (n >= 100) and (n < 1000):
            path_ED.append(f'{dir}DAT000{n}_ED.txt')
            path_MD.append(f'{dir}DAT000{n}_MD.txt')
        else:
            print(f'num > 1000, please modify this code ! ')
            raise ValueError

    
    return path_ED,path_MD

def showerPlot(core,df_ED,df_MD,): # 点图,展示一个event
    
    plt.figure(figsize=(12, 4))
    size=1.5
    core_new=core#[-core[1],core[0]]
    
    plt.subplot(1, 2, 1)
    EDmarker = 'o'  
    EDscatter = plt.scatter(
        df_ED['EDx'], df_ED['EDy'],
        s=10,  # 根据粒子数设置标记大小
        c=np.log10(df_ED['EDpe']),  # 根据粒子数设置颜色
        cmap='viridis',  # 选择一个颜色映射
        alpha=1,  # 设置透明度
        marker=EDmarker  
    )
    plt.colorbar(EDscatter).set_label(' log10: Photo-Electron Count')
    plt.title('ED:pe')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid(linestyle='dashed')
    
    plt.plot(core_new[0], core_new[1], marker='x', markersize=20, color='red', linewidth=6)

    plt.subplot(1, 2, 2)
    MDmarker = 'o'  
    MDscatter = plt.scatter(
        df_MD['EDx'], df_MD['EDy'],
        s=20,  # 根据粒子数设置标记大小
        c=np.log10(df_MD['EDpe']),  # 根据粒子数设置颜色
        cmap='viridis',  # 选择一个颜色映射
        alpha=1,  # 设置透明度
        marker=MDmarker  
    )
    plt.colorbar(MDscatter).set_label('log10: Photo-Electron Count')
    plt.title('MD:pe')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid(linestyle='dashed')
    
    plt.plot(core_new[0], core_new[1], marker='x', markersize=20, color='red', linewidth=6)
    
    plt.suptitle(f'core={core_new}')
    #plt.savefig(pngfile)
    plt.show()

# 用于机器学习  生成:  特征列表  + label
colum=['label',
        'R_ED', 'N_EM','a','b','c',      #  ED
        'R_MD', 'N_Muon'                         #  MD
                                         # 纵向
       ]

def read_data(path_ED,path_MD,label):    #特征列表  + label
    data=pd.DataFrame(columns=colum)

    for n in range(len(path_ED)):
        print(f'In: {path_ED[n]}')
        nevent=0
        id=1
        while True:
            EventData_ED,end = loadData_1_event(path_ED[n],id=id,)
            EventData_MD,_ = loadData_1_event(path_MD[n],id=id,)

            if (EventData_ED is None) and (end==0):
                #print(f'No EVENT with id ={nevent}')
                id += 1
                continue
            elif (EventData_ED is None) and (end==1):
                break

            event_core=EventData_ED.get_core()
            event_data_ED=EventData_ED.get_data()
            event_theta=EventData_ED.get_theta()
            event_data_MD=EventData_MD.get_data()

            if event_data_MD['EDpe'].max() > 1e3 :
                id += 1
                continue
            
            # 计算event的特征
            bins=51  # 多一个
            rmax=600 # 统计的范围
            x=np.linspace(0,rmax,bins)

            # ED
            # R参数
            R_ED = Rparameter(event_core,event_data_ED)
            if R_ED > 800:
                id += 1
                continue

            # pe 指数拟合参数 
            r_bin_ED=pe_radi_dis(event_core,event_data_ED,bins=bins)
            try:
                fitparams_ED=ED_pe_fit(x,r_bin_ED,)  # 指数函数拟合参数a,b,c
            except Exception as e:
                print(f'id = {id} raise {e} : please check it ')
                id += 1
                continue
            if fitparams_ED is None:
                id += 1
                continue

            # N 数量
            N_EM = event_data_ED['EDpe'].sum()


            # MD
            # R参数
            R_MD = Rparameter(event_core,event_data_MD)
            if R_MD > 800:
                id += 1
                continue
            # N_Muon
            
            N_muon=cal_N_muon(event_core,event_data_MD)


            # 前面几个bin里面的pe与后面的比值
            # r_bin_MD=pe_radi_dis(event_core,event_data_MD,bins=bins)
            # tmp0 = len(x[x <= 10])
            # tmp1 = len(x[x <=  40])
            # tmp2 = len(x[x <=  300])
            #ratio_MD=np.mean(r_bin_MD[tmp0-1:tmp1]) / np.mean(r_bin_MD[tmp2-1:])

            # 纵向发展(有待添加)

            # 添加一个event的特征
            data.loc[len(data)] = [label,
                                   
                                   R_ED,N_EM,fitparams_ED[0],fitparams_ED[1],fitparams_ED[2],
                                   R_MD,N_muon
                                   
                                   ]  # event_theta,ratio_MD
            
            


            # 输出
            if nevent < 100:
                print(f'nevent = {nevent}, id = {id}')
            elif nevent % 100 == 0:
                print(f'nevent = {nevent}, id = {id}')


            nevent += 1  
            id += 1
            
            

        print(f'total nevent = {nevent}')
        print()
    return data

def cal_N_muon(core,data,):
    data["r"]=np.sqrt((data['EDx']-core[0])**2 + (data['EDy']-core[1])**2)
    data=data[data['r']>12.5]
    return data['EDpe'].sum()

def pe_radi_dis(core,data,bins=100): # pe径向分布
    rmax=600
    data["r"]=np.sqrt((data['EDx']-core[0])**2 + (data['EDy']-core[1])**2)
    #rmax=1200 #米
    #rmax=data['r'].max()
    bin_list=np.linspace(0,rmax,bins+1) #bin=bins-1
    #print(bin_list)
    data['r_bin'] = pd.cut(data['r'], bins=bin_list,)
    #print(data)
    bin_counts = data.groupby('r_bin', observed=False)['EDpe'].sum()
    r_bin=bin_counts.values
    return r_bin

def Rparameter(core, data):   # 计算 R = rho*r / rho
    data["r"]=np.sqrt((data['EDx']-core[0])**2 + (data['EDy']-core[1])**2)
    # 计算整个阵列的R
    rho_sum=data['EDpe'].sum()
    if rho_sum == 0:
        return 600
    R_i=np.asarray(data['EDpe']*data['r']/rho_sum)
    #R_i=np.sort(R_i)[::-1]
    R=np.sum(R_i)
    return R

def ED_pe_fit(x,r_bin,): # pe径向分布拟合
    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    x=x/100
    #r_bin[r_bin == 0]=1
    r_bin=np.log(r_bin+1)
    #r_bin= r_bin/np.max(r_bin)
    params, pcov = curve_fit(exp_func, x, r_bin,maxfev=10000)
    err=np.sqrt(np.diag(pcov))
    if np.max(err) > 3:
        return None
        #return [0,0,0]
        
    # perr = np.sqrt(np.diag(pcov)) 

    return params



def read_long(path_long,id=1,mode='number'):  # 读取event_id = id 的long数据 ,返回纵向发展(数量或能量沉积)和GH函数拟合参数 
    if id < 1:
        print(f'id must >= 1')
        raise ValueError
    if mode not in ['number', 'deposit']:
        print(f'mode wrong')
        raise ValueError
    
    distribute_header=" LONGITUDINAL DISTRIBUTION"
    deposit_header=" LONGITUDINAL ENERGY DEPOSIT"
    para_header=' PARAMETERS         ='
    chidof_header=" CHI**2/DOF"
    deposit_tail=' FIT OF THE HILLAS CURVE'
    if mode=='number':
        header=distribute_header
        colums='DEPTH       GAMMAS   POSITRONS   ELECTRONS         MU+         MU-     HADRONS     CHARGED      NUCLEI   CHERENKOV'
        colums_list=['DEPTH', 'GAMMAS', 'POSITRONS', 'ELECTRONS', 'MU+', 'MU-', 'HADRONS', 'CHARGED', 'NUCLEI', 'CHERENKOV']
    elif mode=='deposit':
        header=deposit_header
        colums='DEPTH         GAMMA    EM IONIZ     EM CUT    MU IONIZ      MU CUT  HADR IONIZ    HADR CUT   NEUTRINO        SUM'
        colums_list=['DEPTH', 'GAMMA', 'EM_IONIZ', 'EM_CUT', 'MU_IONIZ', 'MU_CUT', 'HADR_IONIZ', 'HADR_CUT', 'NEUTRINO', 'SUM']
    
    find_1=False
    find_2=False
    find_3=False
    long_data=[]
    with open(path_long,'r') as file:
        event_id = 0
        for line in file:

            if line.startswith(header):
                content=line.strip().split()
                event_id = int(content[-1])
                if event_id==id:
                    find_1=True
                    continue
            if (line.strip()==colums) and (find_1==True):
                    find_2=True
                    continue
            
            if find_1 and find_2 and (not find_3) and (mode=='number'):
                try:
                    content=np.float32(line.strip().split())
                except ValueError:
                    if line.startswith(deposit_header):
                        find_3=True
            
                    long_data=pd.DataFrame(long_data,columns=colums_list)
                    long_data=long_data.drop(columns=['HADRONS', 'NUCLEI', 'CHERENKOV'])
                    long_data.drop([len(long_data)-2,len(long_data)-1],inplace=True)
                    continue
                    
                if find_1 and find_2 and (not find_3):            
                    long_data.append(content)
                    continue
            
            if find_1 and find_2 and (not find_3) and (mode=='deposit'):
                try:
                    content=np.float32(line.strip().split())
                except ValueError:
                    if line.startswith(deposit_tail):
                        print(1)
                        find_3=True
            
                    long_data=pd.DataFrame(long_data,columns=colums_list)
                    long_data=long_data[['DEPTH','GAMMA','EM_IONIZ','SUM']]
                    long_data.drop([len(long_data)-1],inplace=True)
                    continue

                if find_1 and find_2 and (not find_3):            
                    long_data.append(content)
                    continue


            if  find_2 and (line.startswith(para_header)):
                content=line.strip().split()
                para_list=np.float32(content[2:])

            elif find_2 and (line.startswith(chidof_header)):
                content=line.strip().split()
                para_list=np.append(para_list,np.float32(content[2]))
                
                return long_data,para_list

            
        print(f'not find ')
        return None,None

def GH(depth,para):  
    P=para
    num = P[0]*(np.abs((depth-P[1])/(P[2]-P[1])))**((P[2]-P[1])/(P[3]+P[4]*depth+P[5]*depth**2)) * np.exp((P[2]-depth)/(P[3]+P[4]*depth+P[5]*depth**2))
    return num


