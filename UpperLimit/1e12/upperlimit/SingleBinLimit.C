// #include "RooWorkspace.h"
// #include "RooRealVar.h"
// #include "RooDataSet.h"
// #include "RooPoisson.h"
// #include "RooGaussian.h"
// #include "RooProdPdf.h"
// #include "RooFitResult.h"
// #include "RooStats/ModelConfig.h"
// #include "RooStats/AsymptoticCalculator.h"
// #include "RooStats/ProfileLikelihoodTestStat.h"
// #include "RooStats/HypoTestInverter.h"
// #include "RooStats/HypoTestInverterResult.h"
// #include "RooStats/HypoTestInverterPlot.h"
// #include "TCanvas.h"
// #include <iostream>

// using namespace RooFit;
// using namespace RooStats;

// void SingleBinLimit() {
//     // 1. 创建工作区
//     RooWorkspace w("w");

//     // 【关键修复 1】：给 mu 开放负数空间，让 Minuit 能找到真实的抛物线谷底！
//     w.factory("mu[0, -0.05, 0.05]");  // 初始值为0，允许范围 [-0.05, 0.05]
    
//     // 【关键修复 2】：给本底留足浮动空间，防止撞墙
//     w.factory("bkg_yield[5, 0.1, 20]"); 
    
//     w.factory("n[1, 0, 50]");         // Data = 1
//     w.factory("sig_nom[500]");        // Signal = 500
    
//     w.factory("bkg_meas[5]");         // 全局观测量
//     w.factory("bkg_sigma[1.0]");      // 本底误差

//     // 2. 构建模型
//     w.factory("expr::N_exp('mu*sig_nom + bkg_yield', mu, sig_nom, bkg_yield)");
//     w.factory("Poisson::pdf_stat(n, N_exp)");
//     w.factory("Gaussian::pdf_syst(bkg_meas, bkg_yield, bkg_sigma)");
//     w.factory("PROD::model(pdf_stat, pdf_syst)");

//     // 3. 构造数据
//     w.var("n")->setVal(1); 
//     RooDataSet data("obsData", "obsData", RooArgSet(*w.var("n")));
//     data.add(RooArgSet(*w.var("n")));

//     // ------------------------------------------------------------------
//     // 【DEBUG 区域】：做一次无条件全局拟合，看 Minuit 到底算出了什么
//     // ------------------------------------------------------------------
//     std::cout << "\n=============================================\n";
//     std::cout << ">>> DEBUG: Running Unconditional Global Fit...\n";
//     std::cout << "=============================================\n";
    
//     RooAbsPdf* model = w.pdf("model");
//     RooFitResult* fitRes = model->fitTo(data, 
//                                         Save(true), 
//                                         PrintLevel(1),  // 打印详细拟合信息
//                                         Minimizer("Minuit2", "migrad"));
    
//     std::cout << "\n>>> Best Fit mu        : " << w.var("mu")->getVal() << " +/- " << w.var("mu")->getError() << "\n";
//     std::cout << ">>> Best Fit bkg_yield : " << w.var("bkg_yield")->getVal() << " +/- " << w.var("bkg_yield")->getError() << "\n";
//     std::cout << ">>> Fit Status (0 is Good): " << fitRes->status() << "\n";
//     std::cout << "=============================================\n\n";

//     // 如果拟合状态不是 0，说明模型构建依然有问题，需要立刻停止。
//     if (fitRes->status() != 0) {
//         std::cout << "ERROR: Unconditional fit failed! Limit calculation will be garbage.\n";
//         return;
//     }

//     // 4. 配置 ModelConfig
//     ModelConfig mc("ModelConfig", &w);
//     mc.SetPdf(*w.pdf("model"));
//     mc.SetParametersOfInterest(*w.var("mu"));
//     mc.SetObservables(*w.var("n"));
//     mc.SetNuisanceParameters(*w.var("bkg_yield"));
//     mc.SetGlobalObservables(*w.var("bkg_meas")); 

//     w.import(mc);

//     // B-only Snapshot
//     w.var("mu")->setVal(0.0);
//     mc.SetSnapshot(*w.var("mu"));

//     // ------------------------------------------------------------------
//     // 5. 使用 Asymptotic Calculator 计算极限
//     // ------------------------------------------------------------------
//     std::cout << ">>> Running Asymptotic Limit Calculation...\n";
//     AsymptoticCalculator ac(data, mc, mc);
//     ac.SetOneSided(true); // 使用单侧上限的 q_mu 统计量
//     ac.SetPrintLevel(-1);

//     HypoTestInverter calc(ac);
//     calc.SetConfidenceLevel(0.90);
//     calc.UseCLs(true);
//     calc.SetVerbose(false);

//     // 恢复扫描区间 (只扫正数区间，因为这是计算物理上限)
//     calc.SetFixedScan(30, 0.000, 0.015); 

//     HypoTestInverterResult *r = calc.GetInterval();

//     // 6. 打印并画图
//     double obsLimit = r->UpperLimit();
//     double expLimit = r->GetExpectedUpperLimit(0);

//     std::cout << "\n=============================================\n";
//     std::cout << "  Observed Limit on mu     : " << obsLimit << "\n";
//     std::cout << "  Expected Limit on mu     : " << expLimit << "\n";
//     std::cout << "=============================================\n";

//     TCanvas* c1 = new TCanvas("c1", "Limit", 800, 600);
//     HypoTestInverterPlot *plot = new HypoTestInverterPlot("HTI_Result_Plot", "CLs Scan", r);
//     plot->Draw("CLb 2CL");
//     c1->SaveAs("./figures/SingleBinLimit.png");
// }


#include "RooWorkspace.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooPoisson.h"
#include "RooGaussian.h"
#include "RooProdPdf.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/FrequentistCalculator.h"
#include "RooStats/ProfileLikelihoodTestStat.h"
#include "RooStats/ToyMCSampler.h"
#include "RooStats/HypoTestInverter.h"
#include "RooStats/HypoTestInverterResult.h"
#include "RooStats/HypoTestInverterPlot.h"
#include "TCanvas.h"
#include <iostream>

using namespace RooFit;
using namespace RooStats;

void SingleBinLimit() {
    // ====================================================================
    // 1. 【用户配置区】：随时修改这里的数据量和误差
    // ====================================================================
    double val_data = 3.0;      // 观测 Data
    double val_bkg  = 5.0;      // 预期本底
    double val_sig  = 1000.0;   // 预期名义信号

    // 系统误差配置 (表示为因数：+20% -> 1.20, -80% -> 0.20)
    double syst_spec_up = 1.20; // Spectrum 上限 (+20%)
    double syst_spec_dn = 0.20; // Spectrum 下限 (-80%)

    double syst_had_up  = 1.20; // Hadronic 上限 (+20%)
    double syst_had_dn  = 0.80; // Hadronic 下限 (-20%)

    double syst_bkg_up  = 1.20; // 本底总体误差上限 (+20%)
    double syst_bkg_dn  = 0.80; // 本底总体误差下限 (-20%)

    // MC 统计误差 (预留位，暂不启用)
    // double syst_mcstat_up = 1.10;
    // double syst_mcstat_dn = 0.90;

    // 扫描范围配置
    double scan_mu_min = 0.00;
    double scan_mu_max = 0.015; 
    int    scan_points = 20;

    // ====================================================================
    // 2. 构建 RooWorkspace 与参数
    // ====================================================================
    RooWorkspace w("w");

    w.factory(Form("n[%f, 0, 100]", val_data));
    w.factory(Form("sig_nom[%f]", val_sig));
    w.factory(Form("bkg_nom[%f]", val_bkg));

    // mu 的下限设为 0 就行，因为系统误差全是乘法，绝不会变负
    w.factory("mu[0, 0, 0.05]"); 

    // --- 步骤 2.1: 建立 Nuisance Parameters (标准高斯参数 N(0,1)) ---
    w.factory("theta_spec[0, -5, 5]");
    w.factory("theta_had[0, -5, 5]");
    w.factory("theta_bkg[0, -5, 5]");
    // w.factory("theta_mcstat[0, -5, 5]"); // [预留]

    // --- 步骤 2.2: 建立 Global Observables (全局约束观测值) ---
    w.factory("theta_spec_obs[0]");
    w.factory("theta_had_obs[0]");
    w.factory("theta_bkg_obs[0]");
    // w.factory("theta_mcstat_obs[0]"); // [预留]

    // --- 步骤 2.3: 建立高斯约束项 ---
    w.factory("Gaussian::constr_spec(theta_spec_obs, theta_spec, 1)");
    w.factory("Gaussian::constr_had(theta_had_obs, theta_had, 1)");
    w.factory("Gaussian::constr_bkg(theta_bkg_obs, theta_bkg, 1)");
    // w.factory("Gaussian::constr_mcstat(theta_mcstat_obs, theta_mcstat, 1)"); // [预留]

    // ====================================================================
    // 3. 构建非对称误差响应函数 (Log-Normal Interpolation)
    // ====================================================================
    // 核心数学：当 theta > 0 时，因数为 (up)^theta；当 theta < 0 时，因数为 (dn)^(-theta)
    w.factory(Form("expr::resp_spec('(theta_spec>=0)*pow(%f, theta_spec) + (theta_spec<0)*pow(%f, -theta_spec)', theta_spec)", syst_spec_up, syst_spec_dn));
    
    w.factory(Form("expr::resp_had('(theta_had>=0)*pow(%f, theta_had) + (theta_had<0)*pow(%f, -theta_had)', theta_had)", syst_had_up, syst_had_dn));
    
    w.factory(Form("expr::resp_bkg('(theta_bkg>=0)*pow(%f, theta_bkg) + (theta_bkg<0)*pow(%f, -theta_bkg)', theta_bkg)", syst_bkg_up, syst_bkg_dn));

    // [预留 MC 统计误差]
    // w.factory(Form("expr::resp_mcstat('(theta_mcstat>=0)*pow(%f, theta_mcstat) + (theta_mcstat<0)*pow(%f, -theta_mcstat)', theta_mcstat)", syst_mcstat_up, syst_mcstat_dn));

    // ====================================================================
    // 4. 构建总模型 (Signal + Bkg)
    // ====================================================================
    // S = mu * Nominal * Spec_Syst * Had_Syst
    w.factory("expr::S_yield('mu * sig_nom * resp_spec * resp_had', mu, sig_nom, resp_spec, resp_had)");
    
    // B = Nominal * Bkg_Syst [* MCStat_Syst]
    w.factory("expr::B_yield('bkg_nom * resp_bkg', bkg_nom, resp_bkg)");
    
    w.factory("expr::N_exp('S_yield + B_yield', S_yield, B_yield)");

    // 主概率密度函数 (泊松) * 所有的约束项
    w.factory("Poisson::pdf_stat(n, N_exp)");
    w.factory("PROD::model(pdf_stat, constr_spec, constr_had, constr_bkg)"); 
    // 若开启 mcstat，则改为: PROD::model(pdf_stat, constr_spec, constr_had, constr_bkg, constr_mcstat)

    // ====================================================================
    // 5. 导入数据并构建独立快照 (B-only 和 S+B)
    // ====================================================================
    RooDataSet data("obsData", "obsData", RooArgSet(*w.var("n")));
    data.add(RooArgSet(*w.var("n")));

    // 建立需要管理的参数集合
    RooArgSet nuisParams(*w.var("theta_spec"), *w.var("theta_had"), *w.var("theta_bkg"));
    RooArgSet globObs(*w.var("theta_spec_obs"), *w.var("theta_had_obs"), *w.var("theta_bkg_obs"));

    // --- B-only 模型 ---
    ModelConfig bModel("B_model", &w);
    bModel.SetPdf(*w.pdf("model"));
    bModel.SetObservables(*w.var("n"));
    bModel.SetParametersOfInterest(*w.var("mu"));
    bModel.SetNuisanceParameters(nuisParams);
    bModel.SetGlobalObservables(globObs);
    w.var("mu")->setVal(0.0);
    bModel.SetSnapshot(*w.var("mu"));

    // --- S+B 模型 ---
    ModelConfig sbModel("SB_model", &w);
    sbModel.SetPdf(*w.pdf("model"));
    sbModel.SetObservables(*w.var("n"));
    sbModel.SetParametersOfInterest(*w.var("mu"));
    sbModel.SetNuisanceParameters(nuisParams);
    sbModel.SetGlobalObservables(globObs);
    w.var("mu")->setVal(0.01);
    sbModel.SetSnapshot(*w.var("mu"));

    // ====================================================================
    // 6. 运行 Frequentist Toy MC
    // ====================================================================
    std::cout << "\n>>> Running Frequentist CLs with Asymmetric Systematics...\n";
    // 顺序至关重要：Data, B-only (Alt), S+B (Null)
    FrequentistCalculator fc(data, bModel, sbModel);
    
    fc.SetToys(3000, 1500); 

    ProfileLikelihoodTestStat profll(*w.pdf("model"));
    profll.SetOneSided(true);
    profll.SetMinimizer("Minuit2");
    profll.SetStrategy(0); 
    profll.SetPrintLevel(-1);

    ToyMCSampler* sampler = (ToyMCSampler*)fc.GetTestStatSampler();
    sampler->SetTestStatistic(&profll);
    sampler->SetUseMultiGen(false);

    HypoTestInverter calc(fc);
    calc.SetConfidenceLevel(0.90);
    calc.UseCLs(true);
    calc.SetVerbose(true);
    calc.SetFixedScan(scan_points, scan_mu_min, scan_mu_max); 

    HypoTestInverterResult *r = calc.GetInterval();
    r->ExclusionCleanup(); 

    // ====================================================================
    // 7. 打印与绘图
    // ====================================================================
    double obsLimit = r->UpperLimit();
    double expLimit = r->GetExpectedUpperLimit(0);

    std::cout << "\n=============================================\n";
    std::cout << "  [FREQUENTIST] Single Bin with Systematics\n";
    std::cout << "=============================================\n";
    std::cout << "  Observed Limit on mu     : " << obsLimit << "\n";
    std::cout << "  Expected Limit on mu     : " << expLimit << "\n";
    std::cout << "  -------------------------------------------\n";
    std::cout << "  Excluded Signal Events:\n";
    std::cout << "  -> OBSERVED : < " << obsLimit * val_sig << " events\n";
    std::cout << "  -> EXPECTED : < " << expLimit * val_sig << " events\n";
    std::cout << "=============================================\n";

    TCanvas* c1 = new TCanvas("c1", "Limit", 800, 600);
    HypoTestInverterPlot *plot = new HypoTestInverterPlot("HTI_Result_Plot", "Frequentist CLs Scan", r);
    plot->Draw();  // "CLb 2CL"
    c1->SaveAs("/data/home/zzh/Filt_Event/UpperLimit/1e12/upperlimit/figures/ToySingleBinLimit.png");
}