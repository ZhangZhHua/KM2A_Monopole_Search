# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os


ED_pos_file = "./config/ED_all.txt"
MD_pos_file = "./config/MD_all.txt"
ED_pos = pd.read_csv(ED_pos_file, sep=" ", skiprows=1)
MD_pos = pd.read_csv(MD_pos_file, sep=" ", skiprows=1)

ED_exp_pos_file = './config/ED_pos_5216up_20220705.txt'
MD_exp_pos_file = './config/MD_pos_1188.txt'
ED_exp_pos = pd.read_csv(ED_exp_pos_file, sep=" ", skiprows=1)
MD_exp_pos = pd.read_csv(MD_exp_pos_file, sep=" ", skiprows=1)


class LHFiltedEvent:
    ED_pos_map_x = None
    ED_pos_map_y = None
    MD_pos_map_x = None
    MD_pos_map_y = None
    @classmethod
    def load_station_pos(cls, ED_pos_df, MD_pos_df):
        cls.ED_pos_map_x = dict(zip(ED_pos_df["id"], ED_pos_df["x"]))
        cls.ED_pos_map_y = dict(zip(ED_pos_df["id"], ED_pos_df["y"]))
        cls.MD_pos_map_x = dict(zip(MD_pos_df["id"], MD_pos_df["x"]))
        cls.MD_pos_map_y = dict(zip(MD_pos_df["id"], MD_pos_df["y"]))
   
    def __init__(self, max_hits_E=200000, max_hits_M=200000):
        # 原始属性
        self.ev_n = -1
        self.mjd = -1
        self.dt = -1

        self.trueE = -1
        self.Id = -1
        self.NpE1 = -1
        self.NpE2 = -1
        self.NpE3 = -1
        self.NuW1 = -1
        self.NuW2 = -1
        self.NuW3 = -1
        self.theta = -1
        self.phi = -1
        self.x = -1
        self.y = -1

        self.rec_Ex = -1
        self.rec_Ey = -1
        self.rec_Ez = -1

        self.NhitE = -1
        self.NhitM = -1
        self.NtrigE = -1
        self.NtrigE2 = -1
        self.NfiltE = -1
        self.NfiltM = -1

        self.rec_theta = -1
        self.rec_phi = -1
        self.rec_x = -1
        self.rec_y = -1
        self.rec_r = -1

        self.rec_Eage = -1
        self.rec_Mage = -1
        self.rec_Echi = -100
        self.rec_Endf = -1
        self.rec_Mchi = -100
        self.rec_Mndf = -1

        self.recE = -1
        self.log10E_limit_low = -1
        self.log10E_limit_high = -1

        # 存 hits 用 list，而不是 DataFrame
        self._hitsE_list = []
        self._hitsM_list = []

    # 添加 hit（直接存到 list）
    def add_hitE(self, p_id, p_time, p_pe, p_np):
        self._hitsE_list.append((p_id, p_time, p_pe, p_np))

    def add_hitM(self, p_id, p_time, p_pe, p_np):
        self._hitsM_list.append((p_id, p_time, p_pe, p_np))

    # 清空 hits
    def clear_hit(self):
        self._hitsE_list.clear()
        self._hitsM_list.clear()

    # ====== 获取 Hits ======
    def get_hitE(self):
        if not self._hitsE_list:
            return pd.DataFrame(columns=["id", "x", "y", "time", "pe", "np"])

        arr = np.array(self._hitsE_list, dtype=object)
        ids = arr[:, 0]

        # 用 map 代替 merge
        xs = np.vectorize(self.ED_pos_map_x.get)(ids)
        ys = np.vectorize(self.ED_pos_map_y.get)(ids)

        df = pd.DataFrame({
            "id": ids,
            "x": xs,
            "y": ys,
            "time": arr[:, 1].astype(float),
            "pe": arr[:, 2].astype(float),
            "np": arr[:, 3].astype(float)
        }).dropna()

        # 聚合（id 相同的合并）
        return df.groupby("id", as_index=False).agg({
            "x": "first",
            "y": "first",
            "time": "mean",
            "pe": "sum",
            "np": "sum"
        })

    def get_hitM(self):
        if not self._hitsM_list:
            return pd.DataFrame(columns=["id", "x", "y", "time", "pe", "np"])

        arr = np.array(self._hitsM_list, dtype=object)
        ids = arr[:, 0]

        xs = np.vectorize(self.MD_pos_map_x.get)(ids)
        ys = np.vectorize(self.MD_pos_map_y.get)(ids)

        df = pd.DataFrame({
            "id": ids,
            "x": xs,
            "y": ys,
            "time": arr[:, 1].astype(float),
            "pe": arr[:, 2].astype(float),
            "np": arr[:, 3].astype(float)
        }).dropna()

        return df.groupby("id", as_index=False).agg({
            "x": "first",
            "y": "first",
            "time": "mean",
            "pe": "sum",
            "np": "sum"
        })

    # ====== 其他功能保持原样 ======
    def trans_core(self):
        self.x, self.y = -self.y, self.x
        self.rec_x, self.rec_y = -self.rec_y, self.rec_x

    def get_core(self):
        self.trans_core()
        try:
            rec_x = float(self.rec_x)
            rec_y = float(self.rec_y)
            return [rec_x, rec_y]
        except (ValueError, TypeError) as e:
            print(f"转换坐标值时出错: {e}")
            return None


    def summary(self):
        print(f"ev_n = {self.ev_n}, trueE = {self.trueE}, Id = {self.Id}")
        print(f"recE = {self.recE}, core = ({self.rec_x:.2f}, {self.rec_y:.2f})")
        print(f"Number of E hits: {self.NhitE}")
        print(f"Number of M hits: {self.NhitM}")

def read_filted_event(event,FiltedEvent,num,nentries=100):
   
    if num>=nentries:
        print("num is too large")
        return None
    
    ev_n = event.GetEvN()
    Id= event.GetId()
    trueE= event.GetTrueE()
    recE= event.GetRecE()
    rec_theta= event.GetRecTheta()
    rec_x= event.GetRecX()
    rec_y= event.GetRecY()
    rec_r= event.GetRecR()
    NpE1= event.GetNpE1()
    NpE2= event.GetNpE2()
    NpE3= event.GetNpE3()
    Eage= event.GetRecEage()
    Mage= event.GetRecMage()
    NhitE= event.GetNhitE()
    NhitM= event.GetNhitM()
    NtrigE= event.GetNtrigE()
    NfiltE= event.GetNfiltE()
    RecEchi= event.GetRecEchi()
    RecMchi= event.GetRecMchi()
    RecEndf= event.GetRecEndf()
    RecMdf= event.GetRecMndf()

    log10E_limit_low=event.GetLog10E_limit_low()
    log10E_limit_high=event.GetLog10E_limit_high()

    FiltedEvent.ev_n=ev_n
    FiltedEvent.Id=Id
    FiltedEvent.trueE=trueE
    FiltedEvent.recE=recE
    FiltedEvent.rec_theta=rec_theta
    FiltedEvent.rec_x=rec_x
    FiltedEvent.rec_y=rec_y
    FiltedEvent.rec_r=rec_r
    FiltedEvent.NpE1=NpE1
    FiltedEvent.NpE2=NpE2
    FiltedEvent.NpE3=NpE3
    FiltedEvent.rec_Eage=Eage
    FiltedEvent.rec_Mage=Mage
    FiltedEvent.NhitE=NhitE
    FiltedEvent.NhitM=NhitM
    FiltedEvent.NtrigE=NtrigE
    FiltedEvent.NfiltE=NfiltE
    FiltedEvent.rec_Echi=RecEchi
    FiltedEvent.rec_Mchi=RecMchi
    FiltedEvent.rec_Endf=RecEndf
    FiltedEvent.rec_Mndf=RecMdf

    FiltedEvent.log10E_limit_low=log10E_limit_low
    FiltedEvent.log10E_limit_high=log10E_limit_high

    hitsE = event.GetHitsE()
    hitsM = event.GetHitsM()
    FiltedEvent.clear_hit()
    for i in range(hitsE.GetEntries()):
        hit = hitsE.At(i)
        if hit.GetPe() <= 0:
            continue
        FiltedEvent.add_hitE(hit.GetId(), hit.GetTime(), hit.GetPe(), hit.GetNp())
    for i in range(hitsM.GetEntries()):
        hit = hitsM.At(i)
        if hit.GetPe() <= 0:
            continue
        FiltedEvent.add_hitM(hit.GetId(), hit.GetTime(), hit.GetPe(), hit.GetNp())

    return FiltedEvent

def event_plot(n,file_path,IS):
    ROOT.gSystem.Load("/Users/macbook/PycharmProjects/pythonProject/searchMonopole/paper/LHAASO_Monopole/Filt_Event/bin/libLHEvent.so")   # 打开 ROOT 文件
    ED_pos_file = "/Users/macbook/PycharmProjects/pythonProject/searchMonopole/paper/LHAASO_Monopole/Filt_Event/config/ED_all.txt"
    MD_pos_file = "/Users/macbook/PycharmProjects/pythonProject/searchMonopole/paper/LHAASO_Monopole/Filt_Event/config/MD_all.txt"
    ED_pos = pd.read_csv(ED_pos_file, sep=" ", skiprows=1)
    MD_pos = pd.read_csv(MD_pos_file, sep=" ", skiprows=1)

    f = ROOT.TFile.Open(file_path)
    event = ROOT.LHFiltedEvent()
    tree = f.Get("filted_tree") 
    tree.SetBranchAddress("FiltedEvent", event)
    nentries=tree.GetEntries()
    tree.GetEntry(n)
    FiltedEvent=LHFiltedEvent()
    if ISMC==1:
        FiltedEvent.load_station_pos(ED_pos_df=ED_pos, MD_pos_df=MD_pos)
    elif ISMC==0:
        FiltedEvent.load_station_pos(ED_pos_df=ED_exp_pos, MD_pos_df=MD_exp_pos)
    else:
        print('wrong ISMC')
        raise ValueError
    FiltedEvent=read_filted_event(event,FiltedEvent=FiltedEvent,num=n,nentries=nentries)
    hitE=FiltedEvent.get_hitE()
    hitM=FiltedEvent.get_hitM()
    core=FiltedEvent.get_core()
    print(core)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    scatter1 = ax1.scatter(hitE['x'], hitE['y'], c=hitE['pe'], s=10)
    ax1.scatter(core[0], core[1], c='r', s=100, marker='x')
    ax1.set_title('E Hits')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    fig.colorbar(scatter1, ax=ax1, label='PE (Photoelectrons)')

    scatter2 = ax2.scatter(hitM['x'], hitM['y'], c=hitM['pe'], s=10)
    ax2.scatter(core[0], core[1], c='r', s=100, marker='x')
    ax2.set_title('M Hits')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    fig.colorbar(scatter2, ax=ax2, label='PE (Photoelectrons)')
    plt.title(f"Event: {n}: core: {core}")
    plt.tight_layout()
    plt.show()
    plt.close()



class EventHitParameters:
    """
    封装所有参数计算的类，方便统一管理和访问
    """
    def __init__(self, core, data_ED, data_MD=None):
        """
        初始化参数计算类
        
        参数:
        - core: 簇射核心坐标 (x, y)
        - data_ED: 电子探测器数据 (DataFrame)
        - data_MD: 缪子探测器数据 (DataFrame, 可选)
        """
        # self.core = core
        
        self.data_ED = data_ED.copy()
        self.data_MD = data_MD.copy() if data_MD is not None else None
        # print(type(self.data_ED['x']),type(self.data_ED['y']))
        # self.data_ED["r"] = np.sqrt((self.data_ED["x"] - self.core[0])**2 + (self.data_ED["y"] - self.core[1])**2)
        self.data_ED["r"] = np.hypot(self.data_ED["x"] - core[0],
                             self.data_ED["y"] - core[1])
        # 计算所有参数
        self.compactness = self._calculate_compactness()
        self.pincness = self._calculate_pincness()
        self.rho40, self.R_mean = self._calculate_rho40_and_Rmean()
        
        if self.data_MD is not None:
            self.N_ED_pe, self.N_MD_pe, self.R_ue = self._calculate_R_ue()
        else:
            self.N_ED_pe = np.sum(self.data_ED['pe'])
            self.N_MD_pe = -1
            self.R_ue = -10
    
    def _calculate_compactness(self):
        """计算紧凑度参数 C = nFit/CxPE45"""
        nFit = len(self.data_ED)
        CxPE45 = np.max(self.data_ED[self.data_ED["r"] > 45]["pe"])
        return nFit / CxPE45 if CxPE45 > 0 else -1
    
    def _calculate_pincness(self):
        """计算针状度参数"""
        data = self.data_ED.copy()
        data['pe'] = data['pe'].apply(lambda x: x if x > 0 else 0)
        data['zeta'] = np.log10(data['pe']+1)
        # data['r'] = np.sqrt((data['x'] - self.core[0])**2 + (data['y'] - self.core[1])**2)
        
        width = 10
        bins = np.arange(0, data['r'].max() + width, width)
        groups = data.groupby(np.digitize(data['r'], bins=bins))
        
        data['zeta_avg'] = groups['zeta'].transform('mean')
        data['zeta_std'] = groups['zeta'].transform('std')
        
        # 只考虑 zeta_std 不为零的行
        data_nonzero_std = data[data['zeta_std'] != 0]
        
        if len(data_nonzero_std) == 0:
            return -1
        
        return np.mean(((data_nonzero_std['zeta'] - data_nonzero_std['zeta_avg'])**2) / 
                       (data_nonzero_std['zeta_std']**2))
    
    def _calculate_rho40_and_Rmean(self):
        """计算 rho40 和 R_mean 参数"""
        data = self.data_ED.copy()
        # data['r'] = np.sqrt((data['x'] - self.core[0])**2 + 
        #                    (data['y'] - self.core[1])**2)
        
        # 计算 rho40
        data_40 = data[data['r'] > 40]
        PE40_sum = data_40['pe'].sum()
        PMT40_count = data_40.shape[0]
        rho40 = PE40_sum / PMT40_count if PMT40_count > 0 else 0
        
        # 计算 R_mean
        R_weight_sum = (data['pe'] * data['r']).sum()
        R_mean = R_weight_sum / data['pe'].sum() if data['pe'].sum() > 0 else 0
        
        return rho40, R_mean
    
    def _calculate_R_ue(self):
        """计算 R_ue 参数 (需要 MD 数据)"""
        N_MD_pe = np.sum(self.data_MD['pe'])
        N_ED_pe = np.sum(self.data_ED['pe'])
        
        if (N_ED_pe <= 0) or (N_MD_pe < 0):
            return N_ED_pe, N_MD_pe, -10
        else:
            return N_ED_pe, N_MD_pe, np.log10((N_MD_pe + 1e-4) / N_ED_pe)
    
    def get_all_parameters(self):
    
        return {
            'compactness': self.compactness,
            'pincness': self.pincness,
            'rho40': self.rho40,
            'R_mean': self.R_mean,
            'N_ED_pe': self.N_ED_pe,
            'N_MD_pe': self.N_MD_pe,
            'R_ue': self.R_ue
        }
    
    def __str__(self):
        """打印所有参数"""
        params = self.get_all_parameters()
        return "\n".join([f"{k}: {v:.4f}" for k, v in params.items()])



def extract_sim_paras(sim_event, additional_params=None):

    if additional_params is None:
        N_ED_pe = 0
        N_MD_pe = 0
        R_ue = 0 
        compactness = 0 
        pincness = 0 
        rho40 = 0 
        R_mean = 0
    else:
        N_ED_pe = additional_params['N_ED_pe']
        N_MD_pe = additional_params['N_MD_pe']
        R_ue = additional_params["R_ue"] 
        compactness = additional_params["compactness"] 
        pincness = additional_params["pincness"] 
        rho40 = additional_params["rho40"] 
        R_mean = additional_params["R_mean"]

    sim_parasDF = pd.DataFrame([{
        "trueE": sim_event.trueE,
        "recE": sim_event.recE,
        "rec_x": sim_event.rec_x,
        "rec_y": sim_event.rec_y,
        "rec_r": sim_event.rec_r,
        "NhitE": sim_event.NhitE,
        "NhitM": sim_event.NhitM,
        "NtrigE": sim_event.NtrigE,
        "NfiltE": sim_event.NfiltE,
        "Eage": sim_event.rec_Eage,
        "Echi": sim_event.rec_Echi,
        "Endf": sim_event.rec_Endf,
        "Mage": sim_event.rec_Mage,
        "Mchi": sim_event.rec_Mchi,
        "Mndf": sim_event.rec_Mndf,
        'NpE1': sim_event.NpE1,
        'NpE2': sim_event.NpE2,
        'NpE3': sim_event.NpE3,

        'N_ED_pe': N_ED_pe,
        'N_MD_pe': N_MD_pe,
        'R_ue': R_ue,
        'compactness': compactness,
        'pincness': pincness,
        'rho40': rho40,
        'R_mean': R_mean,
        
        
    }])
    
    return sim_parasDF

def get_parameters(infile, outfile,ISMC=1):

    
    AllEventParamsList=[]
    file_path=infile
    f = ROOT.TFile.Open(file_path)
    event = ROOT.LHFiltedEvent()
    FiltedEvent=LHFiltedEvent()
    if ISMC==1:
        FiltedEvent.load_station_pos(ED_pos_df=ED_pos, MD_pos_df=MD_pos)
    elif ISMC==0:
        FiltedEvent.load_station_pos(ED_pos_df=ED_exp_pos, MD_pos_df=MD_exp_pos)
    else:
        print('wrong ISMC')
        raise ValueError
    tree = f.Get("filted_tree") 
    tree.SetBranchAddress("FiltedEvent", event)
    nentries=tree.GetEntries()
    n=0
    while True:
        if n%1000==0:
            print(f"Processing event {n}/{nentries}")
        tree.GetEntry(n)
        sim_event=read_filted_event(event, FiltedEvent, num=n, nentries=nentries)
        if sim_event is None:
            print("No more events")
            break
        if n> 1000:
            break
        sim_additional_params=EventHitParameters(sim_event.get_core(), sim_event.get_hitE(), sim_event.get_hitM())
        event_params=extract_sim_paras(sim_event, sim_additional_params.get_all_parameters())
        AllEventParamsList.append(event_params)
        n+=1
    
    sim_all_parasm=pd.concat(AllEventParamsList)
    # print(sim_all_parasm.head(100))
    sim_all_parasm.to_csv(outfile, index=False)

def save_events_npz(file_path, ED_pos, MD_pos, output_npz,ISMC=1):
    f = ROOT.TFile.Open(file_path)
    event = ROOT.LHFiltedEvent()
    FiltedEvent = LHFiltedEvent()
    if ISMC==1:
        FiltedEvent.load_station_pos(ED_pos_df=ED_pos, MD_pos_df=MD_pos)
    elif ISMC==0:
        FiltedEvent.load_station_pos(ED_pos_df=ED_exp_pos, MD_pos_df=MD_exp_pos)
    else:
        print('wrong ISMC')
        raise ValueError

    tree = f.Get("filted_tree") 
    tree.SetBranchAddress("FiltedEvent", event)
    nentries = tree.GetEntries()

    hitsE_list = []
    hitsM_list = []
    labels = []
    # nentries=1000
    for n in range(nentries):
        if n % 1000 == 0:
            print(f"Processing event {n}/{nentries}")

        tree.GetEntry(n)
        sim_event = read_filted_event(event, FiltedEvent, num=n, nentries=nentries)
        if sim_event is None:
            print("No more events")
            break

        # 取 HitsE
        hitsE_df = sim_event.get_hitE()[["x", "y", "pe"]]
        hitsE = hitsE_df.to_numpy(dtype=np.float32)
        # 取 HitsM
        hitsM_df = sim_event.get_hitM()[["x", "y", "pe"]]
        hitsM = hitsM_df.to_numpy(dtype=np.float32)
        # 取标签
        label = sim_event.Id if int(sim_event.Id)>=1 else -1 
        # 添加到列表
        hitsE_list.append(hitsE)
        hitsM_list.append(hitsM)
        labels.append(label)
        # sim_event.clear_hit()
    
    # print(labels[:100])
    # 保存到 npz
    np.savez_compressed(output_npz,
                        hitsE=np.array(hitsE_list, dtype=object),
                        hitsM=np.array(hitsM_list, dtype=object),
                        labels=np.array(labels, dtype=np.int8))

    print(f"数据集已保存到 {output_npz}")

def check_df(df):
    issues = {}
    issues["NaN"] = df[df.isna().any(axis=1)]
    issues["Non-numeric"] = df[~df.applymap(lambda v: isinstance(v, (int, float)))]
    issues["Inf"] = df[np.isinf(df).any(axis=1)]
    return issues

def params_and_npzdataset(file_path, ED_pos, MD_pos, output_csv, output_npz,ISMC=1):
    f = ROOT.TFile.Open(file_path)
    event = ROOT.LHFiltedEvent()
    FiltedEvent = LHFiltedEvent()
    if ISMC==1:
        FiltedEvent.load_station_pos(ED_pos_df=ED_pos, MD_pos_df=MD_pos)
    elif ISMC==0:
        FiltedEvent.load_station_pos(ED_pos_df=ED_exp_pos, MD_pos_df=MD_exp_pos)
    else:
        print('wrong ISMC')
        raise ValueError

    tree = f.Get("filted_tree") 
    tree.SetBranchAddress("FiltedEvent", event)
    nentries = tree.GetEntries()
    AllEventParamsList=[]
    hitsE_list = []
    hitsM_list = []
    labels = []
    # nentries=1000
    for n in range(nentries):
        # print(n)
        if n % 1000 == 0:
            print(f"Processing event {n}/{nentries}")

        tree.GetEntry(n)
        sim_event = read_filted_event(event, FiltedEvent, num=n, nentries=nentries)
        if sim_event is None:
            print(f"total {n} events")
            break

        # 取 HitsE
        hitsE_df = sim_event.get_hitE()[["x", "y", "pe"]]
        hitsE = hitsE_df.to_numpy(dtype=np.float32)
        # 取 HitsM
        hitsM_df = sim_event.get_hitM()[["x", "y", "pe"]]
        hitsM = hitsM_df.to_numpy(dtype=np.float32)
        # 取标签
        label = sim_event.Id if int(sim_event.Id)>=1 else -1 
        # 添加到列表
        hitsE_list.append(hitsE)
        hitsM_list.append(hitsM)
        labels.append(label)
        # print(hitsE_df, hitsM_df)
        # problems=check_df(hitsE_df)
        # for k, v in problems.items():
        #     if not v.empty:
        #         print(f"=== {k} ===")
        #         print(v)
        hitsE_df['x'] = pd.to_numeric(hitsE_df["x"], errors='coerce')
        hitsE_df['y'] = pd.to_numeric(hitsE_df["y"], errors='coerce')
        sim_additional_params=EventHitParameters(sim_event.get_core(), hitsE_df, hitsM_df)
        event_params=extract_sim_paras(sim_event, sim_additional_params.get_all_parameters())
        AllEventParamsList.append(event_params)
        # sim_event.clear_hit()

    sim_all_params=pd.concat(AllEventParamsList)
    # print(sim_all_params.head(10))
    # print(labels[:10])
    sim_all_params.to_csv(output_csv, index=False)
    # 保存到 npz
    np.savez_compressed(output_npz,
                        hitsE=np.array(hitsE_list, dtype=object),
                        hitsM=np.array(hitsM_list, dtype=object),
                        labels=np.array(labels, dtype=np.int8))

def merge_npzdataset(infile_list,sample_num,outfile=None,):
    if len(infile_list)!=len(sample_num):
        print("The length of infile_list and sample_num must be the same")
        raise ValueError
    data_list = [np.load(file, allow_pickle=True) for file in infile_list]
    merged_hitsE = []
    merged_hitsM = []
    merged_labels = []
    for n,data in enumerate(data_list):
        hitsE = data['hitsE']
        hitsM = data['hitsM']
        labels = data['labels']
        if sample_num[n] == -1: # if sample_num is -1, use all events
            merged_hitsE.extend(hitsE)
            merged_hitsM.extend(hitsM)
            merged_labels.extend(labels)
            print(f"Using all events (size={len(labels)}) from file {infile_list[n]}")
        elif sample_num[n] > 0:
            sample_size= min(len(labels), sample_num[n])
            sample_indices = np.random.choice(len(labels), sample_size, replace=False)
            merged_hitsE.extend(hitsE[sample_indices])
            merged_hitsM.extend(hitsM[sample_indices])
            merged_labels.extend(labels[sample_indices])
            print(f"Using {sample_size} events from file {infile_list[n]}")
        else:
            print(f"Invalid sample_num {sample_num[n]} for file {infile_list[n]}")
            raise ValueError
        
    
    merged_hitsE = np.array(merged_hitsE, dtype=object)  # dtype=object to handle variable-length events
    merged_hitsM = np.array(merged_hitsM, dtype=object)
    merged_labels = np.array(merged_labels, dtype=np.int8)  # Assuming labels are integers (e.g., 0=background, 1=signal)
    if outfile is not None:
        np.savez(outfile, hitsE=merged_hitsE, hitsM=merged_hitsM, labels=merged_labels)
        print(f"Merged {len(merged_labels)} events from {len(infile_list)} files into {outfile}.")
    else:
        print(f"Merged {len(merged_labels)} events from {len(infile_list)} files.")
        return merged_hitsE, merged_hitsM, merged_labels


if __name__ == "__main__":
    import ROOT
    ROOT.gSystem.Load("/Users/macbook/PycharmProjects/pythonProject/searchMonopole/paper/LHAASO_Monopole/Filt_Event/bin/libLHEvent.so")   # 打开 ROOT 文件

    # file_DIR=f"./Dataset_Filted/Simulation/gamma/1e3_1e4"
    # file=f"combined_gamma_1e3_1e4_run000.root"
    # csvfile=f"combined_paras_gamma_1e3_1e4_run000.csv"
    # npzfile=f"train_dataset_gamma_1e3_1e4_run000.npz"

    # file_DIR=f"./Dataset_Filted/Simulation/gamma/1e4_1e5"
    # file=f"combined_gamma_1e4_1e5_run000.root"
    # csvfile=f"combined_paras_gamma_1e4_1e5_run000.csv"
    # npzfile=f"train_dataset_gamma_1e4_1e5_run000.npz"

    # proton
    file_DIR=f"./Dataset_Filted/Simulation/proton/1e3_1e4"
    file=f"combined_proton_1e3_1e4_run000.root"
    csvfile=f"combined_paras_proton_1e3_1e4_run000.csv"
    npzfile=f"train_dataset_proton_1e3_1e4_run000.npz"
    
    file_DIR=f"./Dataset_Filted/Simulation/proton/1e4_1e5"
    file=f"combined_proton_1e4_1e5_run000.root"
    csvfile=f"combined_paras_proton_1e4_1e5_run000.csv"
    npzfile=f"train_dataset_proton_1e4_1e5_run000.npz"

    # file_DIR=f"./Dataset_Filted/Simulation/monopole/E1e9"
    # file=f"combined_monopole_E1e9.root"
    # csvfile=f"combined_paras_monopole_E1e9.csv"
    # npzfile=f"train_dataset_monopole_E1e9.npz"
    

    # file_DIR="./Dataset_Filted/Experiment/2024/1231"
    # file="combined_Exp_20241231_1e9.root"
    # csvfile="combined_paras_Exp_20241231_1e9.csv"
    # npzfile="train_dataset_Exp_20241231_1e9.npz"

    file_DIR="./Dataset_Filted/Experiment/2022/0510"
    file="combined_Exp_20220510_1e9_V01.root"
    csvfile="combined_paras_Exp_20220510_1e9_V01.csv"
    npzfile="train_dataset_Exp_20220510_1e9_V01.npz"

    infile=os.path.join(file_DIR,file)
    csvfile=os.path.join(file_DIR,csvfile)
    npzfile=os.path.join(file_DIR,npzfile)
    print(infile)

    ISMC=0
    # get_parameters(infile, csvfile)
    # save_events_npz(infile, ED_pos, MD_pos, npzfile)
    params_and_npzdataset(infile, ED_pos, MD_pos, csvfile, npzfile)
    # infile_list = ['./Dataset_Filted/Simulation/gamma/1e3_1e4/train_dataset_gamma_1e3_1e4_run000.npz',
    #                './Dataset_Filted/Simulation/gamma/1e3_1e4/train_dataset_gamma_1e3_1e4_run000.npz',]
    # outfile = './Dataset_Filted/Simulation/gamma/1e3_1e4/merged.npz'
    # merge_npzdataset(infile_list, outfile=outfile)
    print(f"Done")


    # year="2022"
    # monthday="0510"
    # label="1e9_V01"
    # EOSopen=""
    # file_DIR=f"{EOSopen}/eos/user/z/zhangzhonghua/experiment_rec/{year}/{monthday}"
    # file=f"combined_Exp_{year}{monthday}_{label}"
    # rootfile=f"{file}.root"
    # csvfile=f"{file}_params.csv"
    # npzfile=f"{file}_dataset.npz"

    # # file_DIR="/eos/user/z/zhangzhonghua/experiment_rec/2022/0510"
    # # file="combined_Exp_20220510_1e9_V01.root"
    # # csvfile="combined_paras_Exp_20220510_1e9_V01.csv"
    # # npzfile="train_dataset_Exp_20220510_1e9_V01.npz"

    # # infile="./test.root"#
    # infile=os.path.join(file_DIR,rootfile)
    # csvfile=os.path.join(file_DIR,csvfile)
    # npzfile=os.path.join(file_DIR,npzfile)
    # print(infile)
# %%
# import numpy as np
# data=np.load("./Dataset_Filted/Simulation/gamma/1e3_1e4/merged.npz",allow_pickle=True)
# print(data.files)
# print(data['hitsE'].shape)
# print(data['hitsM'].shape)
# print(data['labels'].shape)

# %%

#  ROOT_BASE="/Users/macbook/PycharmProjects/pythonProject/searchMonopole/paper/LHAASO_Monopole/Filt_Event/Dataset_Filted"
#     mode="Simulation"
#     particle="gamma" # "gamma" "monopole" "proton"
#     erange_list=["1e3_1e4","1e4_1e5","1e5_1e6",] # "1e6_1e7","1e7_1e8","1e8_1e9"
#     runxx="run000"
#     # rootfile_list=[f"combined_{particle}_{erange}_{runxx}.root" for erange in erange_list]
#     for n in range(len(erange_list)):
#         file_DIR=f"{ROOT_BASE}/{mode}/{particle}/{erange_list[n]}"
#         file=f"combined_{particle}_{erange_list[n]}_{runxx}.root"
#         csvfile=f"combined_paras_{particle}_{erange_list[n]}_{runxx}.csv"
#         npzfile=f"train_dataset_{particle}_{erange_list[n]}_{runxx}.npz"
#         print(f"Processing file: {file}")

