// ==========================================================
// SingleBinLimit_WithSyst.C
//
// 单 bin 上限计算，含系统误差：
//   - spectrum        : 非对称 [-80%, +20%]
//   - HadronicModel   : 对称   [-20%, +20%]
//   - MCData_Variation: 注释保留，待启用
//
// 提供两个入口函数：
//   RunFrequentist()  — Toy MC 频率学派（适合低统计量）
//   RunAsymptotic()   — 渐近公式（快速，大统计量下与前者一致）
//
// 使用方法：
//   root -l -q 'SingleBinLimit_WithSyst.C("freq")'
//   root -l -q 'SingleBinLimit_WithSyst.C("asymp")'
// ==========================================================

#include "RooWorkspace.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooAbsPdf.h"
#include "RooAbsReal.h"
#include "RooFitResult.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/FrequentistCalculator.h"
#include "RooStats/AsymptoticCalculator.h"
#include "RooStats/ProfileLikelihoodTestStat.h"
#include "RooStats/ToyMCSampler.h"
#include "RooStats/HypoTestInverter.h"
#include "RooStats/HypoTestInverterResult.h"
#include "RooStats/HypoTestInverterPlot.h"
#include "RooMsgService.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TLatex.h"
#include <iostream>
#include <cmath>

using namespace RooFit;
using namespace RooStats;
using namespace std;

// ==========================================================
//  ██████╗  ██████╗  ███╗   ██╗███████╗██╗ ██████╗
//  ██╔════╝██╔═══██╗ ████╗  ██║██╔════╝██║██╔════╝
//  ██║     ██║   ██║ ██╔██╗ ██║█████╗  ██║██║  ███╗
//  ██║     ██║   ██║ ██║╚██╗██║██╔══╝  ██║██║   ██║
//  ╚██████╗╚██████╔╝ ██║ ╚████║██║     ██║╚██████╔╝
//   ╚═════╝ ╚═════╝  ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝
//
//  修改这里的数字，其余代码无需改动
// ==========================================================

// ---------- 观测数据 ----------
const int    N_OBS         = 1;       // 观测事例数 (整数)

// ---------- 信号 ----------
const double SIG_NOM       = 3081.1;  // mu=1 时的名义信号事例数
// 注：上限结果 mu_up 对应排除信号事例数 = mu_up * SIG_NOM

// ---------- 本底 ----------
const double BKG_NOM       = 3.1989;     // 本底名义值（来自 MC 或 sideband）
const double BKG_STAT_ERR  = 1.0;     // 本底统计误差（MC 统计 / sideband 统计）
// 本底约束项使用 Gaussian(bkg_meas | bkg_yield, BKG_STAT_ERR)

// ---------- 系统误差（相对值，以名义信号产额为基准）----------
// spectrum: 非对称 [-80%, +20%]  → alpha=+1 时信号 ×1.20，alpha=-1 时 ×0.20
const double SYST_SPEC_UP  =  0.61;   // +20%
const double SYST_SPEC_DN  = -0.61;   // -80%  (输入负数)

// HadronicModel: 对称 [-20%, +20%]
const double SYST_HADR_UP  =  0.43;   // +20%
const double SYST_HADR_DN  = -0.43;   // -20%

// MCData_Variation: 暂时注释 (启用时取消注释对应行)
// const double SYST_MCVAR_UP = ...;
// const double SYST_MCVAR_DN = ...;

// ---------- 扫描参数 ----------
const double MU_SCAN_MIN   = 0.0;
const double MU_SCAN_MAX   = 0.02;   // 根据 SIG_NOM 调整：预期上限 ~ 2.3/SIG_NOM
const int    N_SCAN_POINTS = 60;

// ---------- Frequentist Toy 数量 ----------
const int    N_TOYS_NULL   = 10000;   // null (b-only) toys
const int    N_TOYS_ALT    =  5000;   // alt  (s+b)    toys（少一点省时间）

// ==========================================================
// 辅助：构建 Workspace（两种计算器共用）
// ==========================================================
RooWorkspace* BuildWorkspace() {
    RooWorkspace* w = new RooWorkspace("w");

    // ---- POI ----
    // mu 允许略微为负，让 Minuit 找到真实抛物线谷底
    // 下界 = max(-0.5/SIG_NOM, -0.01) 确保 N_exp 永远 > 0
    double mu_lo = -min(0.5 / SIG_NOM, 0.01);
    double mu_hi = MU_SCAN_MAX * 3.0;
    w->factory(Form("mu[0, %g, %g]", mu_lo, mu_hi));

    // ---- 固定常数 ----
    w->factory(Form("sig_nom[%g]",        SIG_NOM));
    w->factory(Form("bkg_meas[%g]",       BKG_NOM));      // 全局观测量：本底测量值
    w->factory(Form("bkg_sigma[%g]",      BKG_STAT_ERR)); // 本底统计误差

    // ---- Nuisance: 本底归一化（由统计误差约束）----
    // bkg_yield ~ Gaussian(bkg_meas, bkg_sigma)，浮动范围取 ±5σ
    double bkg_lo = max(0.01, BKG_NOM - 5*BKG_STAT_ERR);
    double bkg_hi =           BKG_NOM + 5*BKG_STAT_ERR;
    w->factory(Form("bkg_yield[%g, %g, %g]", BKG_NOM, bkg_lo, bkg_hi));

    // ---- Nuisance: 系统误差参数，均为标准正态 N(0,1) ----
    w->factory("alpha_spectrum[0, -5, 5]");
    w->factory("alpha_hadronic[0, -5, 5]");
    // w->factory("alpha_mcvar[0, -5, 5]");   // MCData_Variation: 暂时注释

    // ====================================================
    //  系统误差插值因子：使用指数插值（AsymPow 风格）
    //
    //  定义：
    //    f(alpha) = exp( ln(1+δ_up) * max(alpha,0)
    //                  + ln(1+δ_dn) * max(-alpha,0) )
    //
    //  性质：
    //    f(0)  = 1          （名义值）
    //    f(+1) = 1 + δ_up   （上浮 δ_up）
    //    f(-1) = 1 + δ_dn   （下移 |δ_dn|）
    //    f 在 alpha=0 处连续可微，且 f > 0（不会变负）
    //
    //  这等价于 HistFactory 的 kappa 参数化
    // ====================================================

    // spectrum: f(+1)=1.20, f(-1)=0.20
    double log_spec_up = log(1.0 + SYST_SPEC_UP);  //  ln(1.20)
    double log_spec_dn = log(1.0 + SYST_SPEC_DN);  //  ln(0.20)  < 0
    w->factory(Form(
        "expr::factor_spectrum("
        "'exp(%g * TMath::Max(alpha_spectrum, 0.) + %g * TMath::Max(-alpha_spectrum, 0.))',"
        "alpha_spectrum)",
        log_spec_up, log_spec_dn));

    // hadronic: f(+1)=1.20, f(-1)=0.80
    double log_hadr_up = log(1.0 + SYST_HADR_UP);  //  ln(1.20)
    double log_hadr_dn = log(1.0 + SYST_HADR_DN);  //  ln(0.80)
    w->factory(Form(
        "expr::factor_hadronic("
        "'exp(%g * TMath::Max(alpha_hadronic, 0.) + %g * TMath::Max(-alpha_hadronic, 0.))',"
        "alpha_hadronic)",
        log_hadr_up, log_hadr_dn));

    // MCData_Variation (注释状态):
    // double log_mcvar_up = log(1.0 + SYST_MCVAR_UP);
    // double log_mcvar_dn = log(1.0 + SYST_MCVAR_DN);
    // w->factory(Form("expr::factor_mcvar('exp(%g*TMath::Max(alpha_mcvar,0.)+%g*TMath::Max(-alpha_mcvar,0.))',alpha_mcvar)", ...));

    // ---- 预期事例数 ----
    // N_sig = mu * sig_nom * f_spectrum * f_hadronic (* f_mcvar 待启用)
    // N_exp = N_sig + bkg_yield
    w->factory(
        "expr::N_sig("
        "'mu * sig_nom * factor_spectrum * factor_hadronic',"
        "mu, sig_nom, factor_spectrum, factor_hadronic)");
    // 若启用 MCData_Variation:
    // w->factory("expr::N_sig('mu*sig_nom*factor_spectrum*factor_hadronic*factor_mcvar',...)");

    w->factory("expr::N_exp('N_sig + bkg_yield', N_sig, bkg_yield)");

    // ---- PDF ----
    // 观测量
    w->factory(Form("n[%d, 0, 100]", N_OBS));
    w->var("n")->setVal(N_OBS);

    w->factory("Poisson::pdf_stat(n, N_exp)");
    w->factory("Gaussian::pdf_bkg_syst(bkg_meas, bkg_yield, bkg_sigma)");
    w->factory("Gaussian::pdf_alpha_spectrum(0., alpha_spectrum, 1.)");
    w->factory("Gaussian::pdf_alpha_hadronic(0., alpha_hadronic, 1.)");
    // w->factory("Gaussian::pdf_alpha_mcvar(0., alpha_mcvar, 1.)");

    w->factory("PROD::model("
               "pdf_stat,"
               "pdf_bkg_syst,"
               "pdf_alpha_spectrum,"
               "pdf_alpha_hadronic"
               // ",pdf_alpha_mcvar"   // 待启用
               ")");

    // ---- 数据集 ----
    RooDataSet* data = new RooDataSet("obsData", "obsData", RooArgSet(*w->var("n")));
    data->add(RooArgSet(*w->var("n")));
    w->import(*data);
    delete data;

    return w;
}

// ==========================================================
// 辅助：打印配置摘要
// ==========================================================
void PrintConfig() {
    cout << "\n=============================================" << endl;
    cout << "  Input Configuration" << endl;
    cout << "=============================================" << endl;
    cout << Form("  N_obs         = %d",     N_OBS)         << endl;
    cout << Form("  Signal (mu=1) = %.1f",   SIG_NOM)       << endl;
    cout << Form("  Bkg nominal   = %.1f",   BKG_NOM)       << endl;
    cout << Form("  Bkg stat err  = %.1f",   BKG_STAT_ERR)  << endl;
    cout << Form("  Bkg/Sig ratio = %.4f",   BKG_NOM/SIG_NOM) << endl;
    cout << "  Systematics:" << endl;
    cout << Form("    spectrum     : [%+.0f%%, %+.0f%%]",
                 SYST_SPEC_DN*100, SYST_SPEC_UP*100) << endl;
    cout << Form("    HadronicModel: [%+.0f%%, %+.0f%%]",
                 SYST_HADR_DN*100, SYST_HADR_UP*100) << endl;
    cout << "    MCData_Var   : [commented out]"       << endl;
    cout << Form("  Naive Poisson 90%% CL (no syst): %.5f",
                 2.3 / SIG_NOM) << endl;
    cout << "=============================================" << endl;
}

// ==========================================================
// 辅助：构建 ModelConfig（B-only 和 S+B 快照分开设置）
// ==========================================================
void BuildModelConfigs(RooWorkspace* w,
                       ModelConfig*& bModel,
                       ModelConfig*& sbModel) {
    RooArgSet nuisParams;
    nuisParams.add(*w->var("bkg_yield"));
    nuisParams.add(*w->var("alpha_spectrum"));
    nuisParams.add(*w->var("alpha_hadronic"));
    // nuisParams.add(*w->var("alpha_mcvar"));  // 待启用

    RooArgSet globalObs;
    globalObs.add(*w->var("bkg_meas"));
    // 全局观测量 alpha 约束项的 mean 固定在 0，不需要单独的 global obs
    // 因为 pdf_alpha_xxx 的 mean 已经硬编码为 0.

    // ---- B-only model ----
    bModel = new ModelConfig("B_model", w);
    bModel->SetPdf(*w->pdf("model"));
    bModel->SetObservables(*w->var("n"));
    bModel->SetParametersOfInterest(*w->var("mu"));
    bModel->SetNuisanceParameters(nuisParams);
    bModel->SetGlobalObservables(globalObs);
    w->var("mu")->setVal(0.0);
    bModel->SetSnapshot(*w->var("mu"));

    // ---- S+B model ----
    sbModel = new ModelConfig("SB_model", w);
    sbModel->SetPdf(*w->pdf("model"));
    sbModel->SetObservables(*w->var("n"));
    sbModel->SetParametersOfInterest(*w->var("mu"));
    sbModel->SetNuisanceParameters(nuisParams);
    sbModel->SetGlobalObservables(globalObs);
    w->var("mu")->setVal(MU_SCAN_MAX / 2.0);
    sbModel->SetSnapshot(*w->var("mu"));
}

// ==========================================================
// 辅助：打印每个扫描点的 CLs（用于诊断）
// ==========================================================
void PrintCLsScan(HypoTestInverterResult* r) {
    cout << "\n  CLs Scan Points:" << endl;
    cout << Form("  %-10s %-10s %-10s %-10s %-10s",
                 "mu", "CLs", "err", "CLb", "CLs+b") << endl;
    cout << "  " << string(55,'-') << endl;
    for (int i = 0; i < r->ArraySize(); i++) {
        RooStats::HypoTestResult* htr = r->GetResult(i);
        cout << Form("  %-10.5f %-10.4f %-10.4f %-10.4f %-10.4f",
                     r->GetXValue(i),
                     r->GetYValue(i),
                     r->GetYError(i),
                     htr ? htr->CLb()      : -1.,
                     htr ? htr->CLsplusb() : -1.) << endl;
    }
}

// ==========================================================
// 辅助：打印最终结果
// ==========================================================
void PrintResults(HypoTestInverterResult* r, const char* method) {
    auto Safe = [&](int s) -> double {
        try { return r->GetExpectedUpperLimit(s); } catch(...) { return -999.; }
    };
    double obs = r->UpperLimit();
    double em2 = Safe(-2), em1 = Safe(-1), e0 = Safe(0),
           ep1 = Safe(1),  ep2 = Safe(2);

    cout << "\n=============================================" << endl;
    cout << Form("  90%% CLs Upper Limits  [%s]", method)    << endl;
    cout << "=============================================" << endl;
    cout << Form("  Observed              : %.6f", obs) << endl;
    cout << Form("  Expected -2sigma      : %.6f", em2) << endl;
    cout << Form("  Expected -1sigma      : %.6f", em1) << endl;
    cout << Form("  Expected  median      : %.6f", e0 ) << endl;
    cout << Form("  Expected +1sigma      : %.6f", ep1) << endl;
    cout << Form("  Expected +2sigma      : %.6f", ep2) << endl;
    cout << "---------------------------------------------" << endl;
    cout << Form("  Excluded signal events (obs)    : < %.1f", obs * SIG_NOM) << endl;
    cout << Form("  Excluded signal events (exp)    : < %.1f", e0  * SIG_NOM) << endl;
    cout << "=============================================" << endl;
}

// ==========================================================
// 辅助：画 CLs 扫描图
// ==========================================================
void DrawAndSave(HypoTestInverterResult* r,
                 double obs, double exp,
                 const char* method,
                 const char* outpng) {


    // ==========================================================
    // 画图
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    gStyle->SetPadTickX(1);
    gStyle->SetPadTickY(1);
    gStyle->SetLineWidth(2);
    
    // gStyle->SetPadLeftMargin(0.15); 
    // gStyle->SetPadBottomMargin(0.14);
    // gStyle->SetPadRightMargin(0.08);
    // gStyle->SetPadTopMargin(0.06);

    int font = 132; 
    gStyle->SetTextFont(font);
    gStyle->SetLabelFont(font, "xyz");
    gStyle->SetTitleFont(font, "xyz");
    gStyle->SetLabelSize(0.05, "xyz");
    gStyle->SetTitleSize(0.06, "xyz");
    gStyle->SetTitleOffset(1.1, "y");
    gStyle->SetTitleOffset(1.1, "x");
    gStyle->SetLegendFont(font);                

    // ==========================================================

    

    TCanvas* c = new TCanvas("c_limit", "Limit", 800, 600);
    c->SetLeftMargin(0.14);
    c->SetBottomMargin(0.15); 
    c->SetRightMargin(0.1);

    HypoTestInverterPlot* plot = new HypoTestInverterPlot("HTI_plot", "", r);
    TGaxis::SetMaxDigits(3);
    plot->Draw();//"CLb 2CL"
    c->Update();

    TLine* l = new TLine(gPad->GetUxmin(), 0.1, gPad->GetUxmax(), 0.1);
    l->SetLineColor(kRed); l->SetLineWidth(2); l->Draw("same");
    

    double txtX = 0.23; 
    double txtY = 0.85; 

    double expLimit = r->GetExpectedUpperLimit(0);
    double expLimitPlus1 = r->GetExpectedUpperLimit(1);
    double expLimitMinus1 = r->GetExpectedUpperLimit(-1);      
            
    TLatex lat; 
    lat.SetNDC();
    lat.SetTextFont(132);
    lat.SetTextSize(0.04);

    lat.DrawLatex(txtX, txtY, "LHAASO Preliminary");
    lat.DrawLatex(txtX, txtY - 0.06, "Monopole Analysis, #gamma = 10^{7}");
    lat.DrawLatex(txtX, txtY - 0.11, "90% C.L. Upper Limit");
    lat.DrawLatex(txtX, txtY - 0.17, Form("Observed #mu < %.4f", obs));
    lat.DrawLatex(txtX, txtY - 0.22, Form("Expected #mu < %.4f", expLimit));
    lat.DrawLatex(txtX, txtY - 0.27, Form("Expected #pm 1#sigma: [%.4f, %.4f]", expLimitMinus1, expLimitPlus1));
    // if (obs > 0 && obs < 1e5)
    //     lat.DrawLatex(0.25, 0.68, Form("Obs: #mu < %.5f", obs));
    // if (exp > 0 && exp < 1e5)
    //     lat.DrawLatex(0.25, 0.61, Form("Exp: #mu < %.5f", exp));

    c->SaveAs(outpng);
    cout << "[INFO] Plot saved → " << outpng << endl;
    delete c;
}

// ==========================================================
//   入口 1: Frequentist Toy MC
// ==========================================================
void RunFrequentist() {
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
    RooRealVar::enableSilentClipping();

    PrintConfig();
    cout << "\n>>> [FREQUENTIST] Building workspace..." << endl;

    RooWorkspace* w = BuildWorkspace();
    RooAbsData*   data = w->data("obsData");

    ModelConfig *bModel = nullptr, *sbModel = nullptr;
    BuildModelConfigs(w, bModel, sbModel);

    // ---- 先跑一次无约束拟合（诊断用）----
    cout << "\n>>> Unconditional fit check:" << endl;
    w->var("mu")->setConstant(false);
    RooFitResult* fitCheck = w->pdf("model")->fitTo(
        *data, Save(), PrintLevel(-1), Warnings(false),
        Minimizer("Minuit2","migrad"), Strategy(1), Offset(kTRUE));
    if (fitCheck)
        cout << Form("  mu_hat = %.5f +/- %.5f   status=%d",
                     w->var("mu")->getVal(),
                     w->var("mu")->getError(),
                     fitCheck->status()) << endl;
    delete fitCheck;

    // ---- 配置检验统计量 ----
    ProfileLikelihoodTestStat profLL(*w->pdf("model"));
    profLL.SetOneSided(true);
    profLL.SetOneSidedDiscovery(false);
    profLL.SetPrintLevel(-1);
    profLL.SetMinimizer("Minuit2");
    profLL.SetStrategy(1);
    profLL.SetReuseNLL(true);

    // ---- 配置 ToyMCSampler ----
    // 注意：FrequentistCalculator 构造函数签名
    //   (data, altModel=S+B, nullModel=B-only)
    // 其中 null 对应 mu=mu_test (S+B)，alt 对应 mu=0 (B)
    // RooStats 的命名惯例与物理惯例相反，但 HypoTestInverter 内部会正确处理
    FrequentistCalculator fc(*data,*bModel, *sbModel );
    fc.SetToys(N_TOYS_NULL, N_TOYS_ALT);

    ToyMCSampler* sampler = (ToyMCSampler*)fc.GetTestStatSampler();
    sampler->SetTestStatistic(&profLL);
    sampler->SetUseMultiGen(false); // 单线程，避免 ROOT 内存竞争

    // ---- 运行扫描 ----
    HypoTestInverter calc(fc);
    calc.SetConfidenceLevel(0.90);
    calc.UseCLs(true);
    calc.SetVerbose(false);
    calc.SetFixedScan(N_SCAN_POINTS, MU_SCAN_MIN, MU_SCAN_MAX);

    cout << Form("\n>>> Running Frequentist scan: %d points in [%.4f, %.4f], %d toys/point...",
                 N_SCAN_POINTS, MU_SCAN_MIN, MU_SCAN_MAX, N_TOYS_NULL) << endl;

    HypoTestInverterResult* r = nullptr;
    try { r = calc.GetInterval(); }
    catch (std::exception& e) {
        cout << "[ERROR] " << e.what() << endl; return;
    }
    if (!r || r->ArraySize() == 0) {
        cout << "[ERROR] Empty result!" << endl; return;
    }

    r->ExclusionCleanup();
    PrintCLsScan(r);

    double obs = r->UpperLimit();
    auto Safe = [&](int s)->double{ try{return r->GetExpectedUpperLimit(s);}catch(...){return -999.;}};
    PrintResults(r, "Frequentist");

    // 保存
    TFile* fout = TFile::Open("results/FreqResult_WithSyst.root","RECREATE");
    if (fout && !fout->IsZombie()) { r->Write("HTI_Result"); fout->Close(); }

    DrawAndSave(r, obs, Safe(0), "Frequentist",
                "./figures/SingleBin_Freq_WithSyst.png");

    delete bModel; delete sbModel; delete w;
}

// ==========================================================
//   入口 2: Asymptotic (渐近公式)
// ==========================================================
void RunAsymptotic() {
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

    PrintConfig();
    cout << "\n>>> [ASYMPTOTIC] Building workspace..." << endl;

    RooWorkspace* w = BuildWorkspace();
    RooAbsData*   data = w->data("obsData");

    ModelConfig *bModel = nullptr, *sbModel = nullptr;
    BuildModelConfigs(w, bModel, sbModel);

    // 渐近公式只需要一个 ModelConfig（内部用 S+B 做剖面似然）
    // 传给 AsymptoticCalculator: (data, sbModel, bModel)
    AsymptoticCalculator ac(*data, *bModel,  *sbModel);
    ac.SetOneSided(true);    // q_mu 单侧统计量（上限用）
    ac.SetPrintLevel(-1);

    HypoTestInverter calc(ac);
    calc.SetConfidenceLevel(0.90);
    calc.UseCLs(true);
    calc.SetVerbose(false);
    calc.SetFixedScan(N_SCAN_POINTS, MU_SCAN_MIN, MU_SCAN_MAX);

    cout << Form("\n>>> Running Asymptotic scan: %d points in [%.4f, %.4f]...",
                 N_SCAN_POINTS, MU_SCAN_MIN, MU_SCAN_MAX) << endl;

    HypoTestInverterResult* r = nullptr;
    try { r = calc.GetInterval(); }
    catch (std::exception& e) {
        cout << "[ERROR] " << e.what() << endl; return;
    }
    if (!r || r->ArraySize() == 0) {
        cout << "[ERROR] Empty result!" << endl; return;
    }

    PrintCLsScan(r);

    double obs = r->UpperLimit();
    auto Safe = [&](int s)->double{ try{return r->GetExpectedUpperLimit(s);}catch(...){return -999.;}};
    PrintResults(r, "Asymptotic");

    TFile* fout = TFile::Open("results/AsympResult_WithSyst.root","RECREATE");
    if (fout && !fout->IsZombie()) { r->Write("HTI_Result"); fout->Close(); }

    DrawAndSave(r, obs, Safe(0), "Asymptotic",
                "./figures/SingleBin_Asymp_WithSyst.png");

    delete bModel; delete sbModel; delete w;
}

// ==========================================================
//   统一入口：默认跑 Frequentist
// ==========================================================
void SingleBinLimit_WithSyst(const char* method = "freq") {
    TString m(method);
    if (m == "asymp" || m == "asym" || m == "asymptotic")
        RunAsymptotic();  
        // root -l 'SingleBinLimit_WithSyst.C("asym")'
    else
        RunFrequentist();
        // root -l 'SingleBinLimit_WithSyst.C()' > ./log/SingleBin_Freq_WithSyst.log
}

// cd /data/home/zzh/Filt_Event/UpperLimit/1e12/upperlimit
// root -l SingleBinLimit_WithSyst.C