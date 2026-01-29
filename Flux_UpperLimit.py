import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
def CLs(data, sig_num, bkg_num, mu_scan=np.linspace(0.01, 0.3, 10), R_t=-2.36):
    def poisson_likelihood(mu, n_obs, s_counts, b_counts):
        lam = mu * s_counts + b_counts
        lam = np.clip(lam, 1e-12, None)
        logL = np.sum(n_obs * np.log(lam) - lam - gammaln(n_obs + 1))
        return logL

    # test statistic: q_mu
    def q_mu(mu, n_obs, s_counts, b_counts):
        logL_mu = poisson_likelihood(mu, n_obs, s_counts, b_counts)
        # 找到 best fit (mu_hat>=0)
        mu_hat_grid = np.linspace(0, 0.2, 50)
        logL_vals = [poisson_likelihood(muh, n_obs, s_counts, b_counts) for muh in mu_hat_grid]
        mu_hat = mu_hat_grid[np.argmax(logL_vals)]
        logL_max = np.max(logL_vals)
        return -2 * (logL_mu - logL_max), mu_hat

    # 生成 toy 数据
    def generate_toys(mu, s_counts, b_counts, n_toys=1000):
        lam = mu * s_counts + b_counts
        return np.random.poisson(lam, size=(n_toys, len(lam)))

    # CLs 计算
    def compute_CLs(mu, n_obs, s_counts, b_counts, n_toys=2000):
        q_obs, _ = q_mu(mu, n_obs, s_counts, b_counts)

        # toys under signal+background
        toys_sb = generate_toys(mu, s_counts, b_counts, n_toys)
        q_sb = [q_mu(mu, toy, s_counts, b_counts)[0] for toy in toys_sb]
        p_mu = np.mean(np.array(q_sb) >= q_obs)

        # toys under background only
        toys_b = generate_toys(0, s_counts, b_counts, n_toys)
        q_b = [q_mu(mu, toy, s_counts, b_counts)[0] for toy in toys_b]
        p_b = np.mean(np.array(q_b) >= q_obs)

        CLs = p_mu / (1 - p_b + 1e-12)
        return CLs, q_obs

    sig_hist=data['sig_hist']
    bkg_hist=data['bkg_hist']
    # sig_num*=5
    # bkg_num*=5
    threshold=0.4
    bins=50

    s_pdf,bin_edges=np.histogram(sig_hist, bins=bins, density=True, range=(0,1))
    b_pdf,_=np.histogram(bkg_hist, bins=bins, density=True, range=(0,1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = bin_centers > threshold
    bin_widths = np.diff(bin_edges)[mask]
    s_pdf_cut=s_pdf[mask]
    b_pdf_cut=b_pdf[mask]
    s_counts = s_pdf_cut*sig_num*bin_widths
    b_counts = b_pdf_cut*bkg_num*bin_widths

    counts,_ = np.histogram(data['bkg_hist'], bins=bins, density=True, range=(0,1))
    n_obs_pdf_cut = counts[mask]
    n_obs = n_obs_pdf_cut * bkg_num * bin_widths

    # mu_scan = np.linspace(0.15, 0.4, 10)
    CLs_list = []
    for mu in mu_scan:
        CLs_val, q_obs = compute_CLs(mu, n_obs, s_counts, b_counts, n_toys=10000)
        # print(f"    mu={mu:.3f}, CLs={CLs_val:.3f}, q_obs={q_obs:.2f}")
        CLs_list.append(CLs_val)

    # 找到90% CL 上限
    mask = np.array(CLs_list) < 0.1
    if np.any(mask):
        mu_upper = mu_scan[np.where(mask)[0][0]]
        print("     90% CL upper limit on mu =", mu_upper)
    else:
        mu_upper = np.nan
        print("     没有找到小于0.1的 CLs, 请扩大 mu 范围")

    plt.plot(mu_scan, CLs_list, label=f"R_t={R_t:.2f}, mu upper={mu_upper:.2e}")
    return mu_upper

ga_data_list=[  
                "/data/zhonghua/Dataset_Filted/Simulation/1e10_V03/Gamma_combined_filted_1e10_V03_13_14_params.csv",
                "/data/zhonghua/Dataset_Filted/Simulation/1e10_V03/Gamma_combined_filted_1e10_V03_14_15_params.csv",
                "/data/zhonghua/Dataset_Filted/Simulation/1e10_V03/Gamma_combined_filted_1e10_V03_15_16_params.csv",
                ]
pr_data_list=[
                "/data/zhonghua/Dataset_Filted/Simulation/1e10_V03/Proton_combined_filted_1e10_V03_13_14_params.csv",
                "/data/zhonghua/Dataset_Filted/Simulation/1e10_V03/Proton_combined_filted_1e10_V03_14_15_params.csv",
                "/data/zhonghua/Dataset_Filted/Simulation/1e10_V03/Proton_combined_filted_1e10_V03_15_16_params.csv",
                ]
mn_data_list=['/data/zhonghua/Dataset_Filted/Simulation/1e10_V03/Monopole_combined_filted_1e10_V03_params.csv']
seconds= 365*24*3600 # 1年 的秒数
eff_areas = 600*600*np.pi*1e4 #  ~单位cm^2
eff_sr=2.243 # 河外立体角, km2a视场

data=np.load('/home/zhonghua/Filt_Event/figures/GNN_Val_hist_1e10_V03_csv.npz')
# data=np.load('/home/zhonghua/Filt_Event/figures/GNN_Val_hist_1e10_V03_csv-proton.npz')
bkg=data['bkg_hist']
sig=data['sig_hist']

sig_num=800*0.95
ga_nums=[7.93E+05*0.025296213, 2.20E+04*0.189300592, 2.25E+02*0.000142012]
pr_nums=[2.18E+10*0.004891248, 1.23E+09*0.18012071, 1.78E+07*0.003455621]
mn_nums=sig_num
R_significancs=[]
GNN_significs=[]
mu_up_list=[]
thresholds=np.arange(-2.8,-2.4, 0.01)
# thresholds=np.arange(-3,-2, 0.5)
for threshold in thresholds:
    ga_ratios=[]
    pr_ratios=[]
    mn_ratio=1
    for data_file in ga_data_list:
        df = pd.read_csv(data_file)
        num = len(df[df['R_ue'] < threshold])
        ga_ratios.append(num/len(df))
    for data_file in pr_data_list:
        df = pd.read_csv(data_file)
        num = len(df[df['R_ue'] < threshold])
        pr_ratios.append(num/len(df))
    for data_file in mn_data_list:
        df = pd.read_csv(data_file)
        num = len(df[df['R_ue'] < threshold])
        mn_ratio=num/len(df)

    significance=(mn_nums*mn_ratio)/np.sqrt(np.sum(np.array(pr_nums)*np.array(pr_ratios))+np.sum(np.array(ga_nums)*np.array(ga_ratios)))
    # print(f"Thresholds: {threshold}: \n mn={mn_ratio:.2e}\n pr={pr_ratios}\n ga={ga_ratios}\n Significance: {significance:.2e}\n")
    R_significancs.append(significance)
    print(f"Threshold={threshold}")
    mu_scan = np.linspace(0.01, 0.3, 29)
    mu_up=CLs(data, sig_num*mn_ratio, np.sum(np.array(pr_nums)*np.array(pr_ratios))+np.sum(np.array(ga_nums)*np.array(ga_ratios)), mu_scan=mu_scan, R_t=threshold)
    mu_up_list.append(mu_up)

plt.axhline(0.1, color='r', linestyle='--', label="90% CL")
plt.xlabel("mu")
plt.ylabel("CLs")
plt.title(f"CLs scan: min mu_up={np.nanmin(mu_up_list):.2e}")
plt.legend()
plt.savefig(f'./PPT_figs/CLs_scan_1e10_V03_Rt.png')
plt.show()
plt.close()

plt.plot(thresholds,mu_up_list,marker='o')
plt.xlabel('R_ue Thresholds')
plt.ylabel('90% CL upper limit on mu')
plt.title('90% CL upper limit on mu vs R_ue Thresholds')
plt.savefig(f'./PPT_figs/mu_up_1e10_V03_Rt.png')
plt.show()
plt.close()
# print(f"Optimal R_ue Thresholds: {thresholds[np.argmax(R_significancs)]:.4f}, with Significance: {max(R_significancs):.4f}, at mn_ratio={mn_ratio:.2e}, pr_ratios={[f'{r:.2e}' for r in pr_ratios]}, ga_ratios={[f'{r:.2e}' for r in ga_ratios]}")
# print(f"Left: mn_num={mn_nums*mn_ratio:.2e}, pr_num={np.sum(np.array(pr_nums)*np.array(pr_ratios)):.2e}, ga_num={np.sum(np.array(ga_nums)*np.array(ga_ratios)):.2e}")
plt.plot(thresholds,R_significancs,marker='o')
plt.xlabel('R_ue Thresholds')
plt.ylabel('R_Significance')
plt.title('R_Significance vs R_ue Thresholds')
plt.savefig(f'./PPT_figs/R_Significance_1e10_V03_Rt.png')
plt.show()

