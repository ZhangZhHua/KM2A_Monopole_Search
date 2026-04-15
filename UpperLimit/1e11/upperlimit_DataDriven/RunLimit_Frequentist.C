// ==========================================================
// RunLimit_Frequentist.C  (修复版)
//
// 修复内容:
//   1. 移除 Eps() (无效命令) → 改用 Precision()
//   2. alpha 范围策略重写:
//      - Toy MC 阶段不限制范围（允许 Gaussian 自由采样）
//      - 仅在 Post-Fit 画图阶段收窄到 [-2,2]
//   3. 加入 RooRealVar::enableSilentClipping() 作为保底
// ==========================================================

#include "TFile.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TLine.h"
#include "TGraph.h"
#include "TPad.h"

#include "RooWorkspace.h"
#include "RooAbsPdf.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooFitResult.h"
#include "RooPlot.h"
#include "RooHist.h"
#include "RooCurve.h"
#include "RooMsgService.h"

#include "RooStats/ModelConfig.h"
#include "RooStats/FrequentistCalculator.h"
#include "RooStats/HypoTestInverter.h"
#include "RooStats/HypoTestInverterResult.h"
#include "RooStats/HypoTestInverterPlot.h"
#include "RooStats/ToyMCSampler.h"
#include "RooStats/ProfileLikelihoodTestStat.h"

using namespace RooFit;
using namespace RooStats;
using namespace std;

// ==========================================================
// 辅助函数：Data - Bkg 残差直方图
// ==========================================================
RooHist* MakeDiffHist(RooHist* data, RooCurve* curve) {
    if (!data || !curve) return nullptr;
    RooHist* diff = new RooHist(*data);
    diff->SetName("h_diff");
    for (int i = 0; i < diff->GetN(); i++) {
        double x, y;
        data->GetPoint(i, x, y);
        double modelVal = curve->interpolate(x);
        diff->SetPoint(i, x, y - modelVal);
        diff->SetPointError(i, 0, 0,
                            data->GetErrorYlow(i),
                            data->GetErrorYhigh(i));
    }
    return diff;
}

// ==========================================================
// 辅助函数：误差带 → 残差带（围绕 0）
// ==========================================================
TGraph* MakeDiffBand(RooCurve* band, RooCurve* central) {
    if (!band || !central || band->GetN() <= 0) return nullptr;
    TGraph* diffBand = new TGraph(band->GetN());
    diffBand->SetName(Form("%s_diff", band->GetName()));
    for (int i = 0; i < band->GetN(); i++) {
        double x, y;
        band->GetPoint(i, x, y);
        diffBand->SetPoint(i, x, y - central->interpolate(x));
    }
    diffBand->SetFillColor(band->GetFillColor());
    diffBand->SetFillStyle(band->GetFillStyle());
    diffBand->SetLineColor(band->GetLineColor());
    diffBand->SetLineWidth(band->GetLineWidth());
    return diffBand;
}

// ==========================================================
// 画图风格
// ==========================================================
void SetAcademicStyle() {
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);
    gStyle->SetPadTickX(1);
    gStyle->SetPadTickY(1);
    gStyle->SetLineWidth(2);
    gStyle->SetPadLeftMargin(0.15);
    gStyle->SetPadBottomMargin(0.14);
    gStyle->SetPadRightMargin(0.05);
    gStyle->SetPadTopMargin(0.06);
    int font = 132;
    gStyle->SetTextFont(font);
    gStyle->SetLabelFont(font, "xyz");
    gStyle->SetTitleFont(font, "xyz");
    gStyle->SetLabelSize(0.05, "xyz");
    gStyle->SetTitleSize(0.06, "xyz");
    gStyle->SetTitleOffset(1.1, "y");
    gStyle->SetTitleOffset(1.1, "x");
    gStyle->SetLegendFont(font);
}

// ==========================================================
// 健壮拟合：三级容错策略
//   Migrad(S1) → Simplex+Migrad(S2) → Migrad(S1, 放宽精度)
// ==========================================================
RooFitResult* RobustFit(RooAbsPdf* pdf, RooAbsData* dat) {
    RooFitResult* res = nullptr;

    // 第一次：标准 Migrad Strategy 1
    res = pdf->fitTo(*dat,
                     Save(), PrintLevel(-1), Warnings(false),
                     Minimizer("Minuit2", "migrad"),
                     Strategy(1),
                     Offset(kTRUE), Hesse(kTRUE));
    if (res && res->status() == 0) return res;

    // 第二次：Simplex 暖启动 + Migrad Strategy 2
    if (res) { delete res; res = nullptr; }
    pdf->fitTo(*dat, PrintLevel(-1), Warnings(false),
               Minimizer("Minuit2", "simplex"),
               Strategy(0), Offset(kTRUE));
    res = pdf->fitTo(*dat,
                     Save(), PrintLevel(-1), Warnings(false),
                     Minimizer("Minuit2", "migrad"),
                     Strategy(2),
                     Offset(kTRUE), Hesse(kTRUE));
    if (res && res->status() == 0) return res;

    // 第三次：放宽收敛精度（Precision 替代已废弃的 Eps）
    if (res) { delete res; res = nullptr; }
    res = pdf->fitTo(*dat,
                     Save(), PrintLevel(-1), Warnings(false),
                     Minimizer("Minuit2", "migrad"),
                     Strategy(1),
                     Precision(1e-3),   // ← 正确的精度控制命令
                     Offset(kTRUE), Hesse(kFALSE));

    return res;
}

// ==========================================================
// 主函数
// ==========================================================
void RunLimit_Frequentist() {
    SetAcademicStyle();

    // 抑制 RooFit 冗余输出
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

    // ----------------------------------------------------------
    // 关键修复 ①：开启静默裁剪，防止 Toy 采样时变量恰好到达边界而崩溃
    // 这是 ROOT >= 6.28 新增的安全阀，若编译报错说明版本不支持可注释掉
    // ----------------------------------------------------------
    RooRealVar::enableSilentClipping();

    // ----------------------------------------------------------
    // 1. 读取 Workspace
    // ----------------------------------------------------------
    const char* filename = "results/Monopole_Search_combined_Monopole_Limit_Setting_model.root";
    TFile* file = TFile::Open(filename);
    if (!file || file->IsZombie()) {
        cout << "[ERROR] Cannot open: " << filename << endl; return;
    }
    RooWorkspace* w = (RooWorkspace*)file->Get("combined");
    if (!w) { cout << "[ERROR] Workspace 'combined' not found!" << endl; return; }

    ModelConfig*  mc   = (ModelConfig*)w->obj("ModelConfig");
    RooAbsData*   data = w->data("obsData");
    if (!mc || !data) { cout << "[ERROR] ModelConfig or data missing!" << endl; return; }

    RooRealVar* mu     = (RooRealVar*)w->var("mu");
    RooRealVar* mu_bkg = (RooRealVar*)w->var("mu_bkg");
    RooRealVar *x = w->var("obs_x_LHAASO_Monopole_Channel"); 
    RooAbsPdf *model = w->pdf("model_LHAASO_Monopole_Channel");
    if (!mu || !x || !model) { cout << "[ERROR] Key vars/pdf missing!" << endl; return; }

    // ----------------------------------------------------------
    // 关键修复 ②：Toy MC 阶段 ——「不」限制 alpha 范围
    //
    // 原理：FrequentistCalculator 产生 toy 时，nuisance 参数从
    // Gaussian 约束先验中采样，σ=1 的 Gaussian 有 ~0.27% 概率
    // 超出 ±3。若将变量范围设为 [-3,3]，采样值恰好落在边界就
    // 会触发 "Value outside range" 异常。
    //
    // 正确做法：Toy 阶段保留变量的「物理范围」（通常已在
    // workspace 里设好，如 [-5,5]），不额外收窄；
    // 仅在最终画图拟合时才收窄到 [-2,2] 防止外推。
    // ----------------------------------------------------------
    RooArgSet allVars = w->allVars();

    // 打印一下当前所有 alpha 的范围（便于调试）
    cout << "[INFO] Alpha parameter ranges in workspace:" << endl;
    for (auto* arg : allVars) {
        RooRealVar* var = dynamic_cast<RooRealVar*>(arg);
        if (var && TString(var->GetName()).Contains("alpha_")) {
            cout << "  " << var->GetName()
                 << "  range=[" << var->getMin() << ", " << var->getMax() << "]" << endl;
        }
    }

    // ----------------------------------------------------------
    // 2. 配置 FrequentistCalculator
    // ----------------------------------------------------------
    const int    nToys       = 20000;  // 每个扫描点的 toy 数
    const int    nScanPoints = 10;    // CLs 曲线扫描点数
    const double muMin       = 0.0;
    const double muMax       = 0.005; // ← 根据物理预期调整
    mu->setVal(0);
    model->fitTo(*data);

    mu->setVal(1);
    model->fitTo(*data);

    mu->setVal(0);
    double n0 = model->expectedEvents(*x);

    mu->setVal(1);
    double n1 = model->expectedEvents(*x);

    cout << "[DEBUG] yield(mu=0) = " << n0 << endl;
    cout << "[DEBUG] yield(mu=1) = " << n1 << endl;

    mu->setVal(1.0);
    mu->setConstant(false);
    if (mu_bkg) { mu_bkg->setVal(1.0); mu_bkg->setConstant(false); }

    // 单侧剖面似然比检验统计量 q_mu
    ProfileLikelihoodTestStat profLL(*mc->GetPdf());
    profLL.SetOneSided(true);           // mu_hat<0 时 q_mu=0
    profLL.SetOneSidedDiscovery(false); // 上限模式
    profLL.SetPrintLevel(-1);
    profLL.SetReuseNLL(true);           // 重用 NLL 加速
    
    // ToyMCSampler
    ToyMCSampler sampler(profLL, nToys);
    sampler.SetPdf(*mc->GetPdf());
    sampler.SetObservables(*mc->GetObservables());
    sampler.SetGlobalObservables(*mc->GetGlobalObservables());

    if (!mc->GetPdf()->canBeExtended()) {
        int nEv = data->numEntries();
        sampler.SetNEventsPerToy(nEv);
        cout << "[INFO] Non-extended PDF: fixing toy events = " << nEv << endl;
    }

    // FrequentistCalculator
    FrequentistCalculator fc(*data, *mc, *mc, &sampler);
    fc.SetToys(nToys, nToys / 2); // (null toys, alt toys)

    // HypoTestInverter
    HypoTestInverter calc(fc);
    calc.SetConfidenceLevel(0.90);
    calc.UseCLs(true);
    calc.SetVerbose(false);
    mu->setRange(muMin, muMax);
    calc.SetFixedScan(nScanPoints, muMin, muMax);

    // ----------------------------------------------------------
    // 在创建计算器和统计量后添加：
    profLL.SetPrintLevel(0); // 关掉 Minuit 内部的细节
    calc.SetVerbose(true);    // 保留扫描点的进度
    // ----------------------------------------------------------
    // 3. 运行上限计算
    // ----------------------------------------------------------
    cout << "\n>>> Starting Frequentist Limit Calculation..." << endl;
    cout << "    nToys=" << nToys << "  nScanPoints=" << nScanPoints
         << "  muRange=[" << muMin << "," << muMax << "]" << endl;

    HypoTestInverterResult* r = nullptr;
    try {
        r = calc.GetInterval();
    } catch (std::exception& e) {
        cout << "[ERROR] GetInterval() exception: " << e.what() << endl;
        return;
    }

    if (!r || r->ArraySize() == 0) {
        cout << "[ERROR] Empty result! Check toys and PDF validity." << endl;
        return;
    }

    // ----------------------------------------------------------
    // 4. 安全提取期望/观测上限
    // ----------------------------------------------------------
    auto SafeLimit = [&](int sigma) -> double {
        try { return r->GetExpectedUpperLimit(sigma); }
        catch (...) { return -999.; }
    };

    double upperLimit = r->UpperLimit();
    double expLimitM2 = SafeLimit(-2);
    double expLimitM1 = SafeLimit(-1);
    double expLimit   = SafeLimit(0);
    double expLimitP1 = SafeLimit(1);
    double expLimitP2 = SafeLimit(2);

    cout << "\n===== 90% CLs Upper Limits (Frequentist) =====" << endl;
    cout << Form("  Observed               : %.5f", upperLimit)  << endl;
    cout << Form("  Expected -2sigma       : %.5f", expLimitM2)  << endl;
    cout << Form("  Expected -1sigma       : %.5f", expLimitM1)  << endl;
    cout << Form("  Expected  median       : %.5f", expLimit)    << endl;
    cout << Form("  Expected +1sigma       : %.5f", expLimitP1)  << endl;
    cout << Form("  Expected +2sigma       : %.5f", expLimitP2)  << endl;
    cout << "===============================================" << endl;

    // 保存结果（断点续传用）
    {
        TFile* out = TFile::Open("results/FrequentistResult.root", "RECREATE");
        if (out && !out->IsZombie()) {
            r->Write("HTI_Result");
            out->Close();
            cout << "[INFO] Result saved → results/FrequentistResult.root" << endl;
        }
    }

    // ==========================================================
    // Part A: CLs 扫描图 (Brazil Plot)
    // ==========================================================
    TCanvas* c1 = new TCanvas("c1", "Limit Scan", 1000, 800);
    c1->SetLeftMargin(0.14);
    c1->SetBottomMargin(0.15);
    c1->SetRightMargin(0.08);

    HypoTestInverterPlot* htplot =
        new HypoTestInverterPlot("HTI_Result_Plot", "", r);
    htplot->Draw("CLb 2CL");
    c1->Update();

    double xMin = gPad->GetUxmin(), xMax = gPad->GetUxmax();
    TLine* line90 = new TLine(xMin, 0.1, xMax, 0.1);
    line90->SetLineColor(kRed); line90->SetLineWidth(2);
    line90->Draw("same");

    TLatex lat;
    lat.SetNDC(); lat.SetTextFont(132); lat.SetTextSize(0.04);
    double tx = 0.23, ty = 0.88;
    lat.DrawLatex(tx, ty,        "LHAASO Preliminary");
    lat.DrawLatex(tx, ty - 0.06, "Monopole Analysis, #gamma = 10^{6}");
    lat.DrawLatex(tx, ty - 0.11, "90% C.L. Upper Limit (Frequentist CLs)");
    if (upperLimit > 0)
        lat.DrawLatex(tx, ty-0.17, Form("Observed #mu < %.4f", upperLimit));
    if (expLimit > 0)
        lat.DrawLatex(tx, ty-0.22, Form("Expected #mu < %.4f", expLimit));
    if (expLimitM1 > 0 && expLimitP1 > 0)
        lat.DrawLatex(tx, ty-0.27,
            Form("Expected #pm 1#sigma: [%.4f, %.4f]", expLimitM1, expLimitP1));

    c1->SaveAs("./figures/1e11_CLs_scan_freq.png");
    c1->SaveAs("/data/home/zzh/Filt_Event/Note_Writing/figures/1e11_CLs_scan_freq.pdf");
    cout << "[INFO] CLs scan plot saved." << endl;

    // ==========================================================
    // Part B: Post-Fit 分布图
    // 这里才收窄 alpha 范围，防止画图拟合外推到非物理区域
    // ==========================================================
    cout << "\n>>> Plotting Post-Fit Distributions..." << endl;

    // 关键修复 ③：仅在此处收窄 alpha，且用 setRange 而非直接赋值
    for (auto* arg : allVars) {
        RooRealVar* var = dynamic_cast<RooRealVar*>(arg);
        if (var && TString(var->GetName()).Contains("alpha_")) {
            // 先检查当前范围，只在原范围内收窄，不扩大
            double lo = TMath::Max(var->getMin(), -2.0);
            double hi = TMath::Min(var->getMax(),  2.0);
            var->setRange(lo, hi);
            // 如果当前值在新范围外，重置为 0
            if (var->getVal() < lo || var->getVal() > hi)
                var->setVal(0.0);
        }
    }

    mu->setVal(0);
    mu->setConstant(true);

    RooFitResult* fitRes = RobustFit(model, data);
    if (!fitRes)
        cout << "[WARNING] All fit attempts failed for post-fit plot!" << endl;
    else
        cout << "[INFO] Post-fit status=" << fitRes->status()
             << "  edm=" << fitRes->edm() << endl;

    RooPlot* frame = x->frame(Title(" "));

    data->plotOn(frame, Name("Data"),
                 DataError(RooAbsData::Poisson), MarkerStyle(20), MarkerSize(2));

    if (fitRes) {
        model->plotOn(frame, Name("ModelBand2"),
                      VisualizeError(*fitRes, 2, kTRUE),
                      FillColor(kYellow), LineColor(kNone));
        model->plotOn(frame, Name("ModelBand1"),
                      VisualizeError(*fitRes, 1, kTRUE),
                      FillColor(kGreen), LineColor(kNone));
    }
    model->plotOn(frame, Name("BkgLine"), LineColor(kBlue), LineWidth(2));

    mu->setConstant(false);
    double signalScale = 1.0;
    mu->setMax(upperLimit * signalScale * 2.0);
    mu->setVal(upperLimit * signalScale);
    model->plotOn(frame, Name("SignalLine"),
                  Components("Signal*"),
                  LineColor(kRed), LineStyle(kDashed), LineWidth(3));

    data->plotOn(frame, Name("Data"),
                 DataError(RooAbsData::Poisson),
                 MarkerStyle(20), MarkerSize(1), LineWidth(1));

    // --- Canvas ---
    TCanvas* c2 = new TCanvas("c2", "Fit Result", 1600, 1600);

    TPad* pad1 = new TPad("pad1", "pad1", 0, 0.3, 1, 1.0);
    pad1->SetBottomMargin(0.02); pad1->SetLeftMargin(0.15);
    pad1->SetRightMargin(0.05); pad1->SetTopMargin(0.05);
    pad1->Draw(); pad1->cd();

    frame->GetYaxis()->SetTitle("Events / Bin");
    frame->GetYaxis()->SetTitleOffset(1.3);
    frame->GetXaxis()->SetLabelSize(0);
    frame->Draw();

    TLegend* leg = new TLegend(0.6, 0.6, 0.90, 0.88);
    leg->SetBorderSize(0); leg->SetFillStyle(0); leg->SetTextSize(0.04);
    leg->AddEntry(frame->findObject("Data"), "Data", "pe");
    leg->AddEntry(frame->findObject("BkgLine"), "Background Fit", "l");
    if (frame->findObject("ModelBand1"))
        leg->AddEntry(frame->findObject("ModelBand1"), "Bkg. Unc. #pm 1#sigma", "f");
    if (frame->findObject("ModelBand2"))
        leg->AddEntry(frame->findObject("ModelBand2"), "Bkg. Unc. #pm 2#sigma", "f");
    leg->AddEntry(frame->findObject("SignalLine"),
                  Form("Signal (#mu_{up} #times %.0f)", signalScale), "l");
    leg->Draw();

    lat.DrawLatex(0.25, 0.85, "LHAASO Preliminary");
    lat.DrawLatex(0.25, 0.78, "Monopole Analysis, #gamma = 10^{6}");
    lat.DrawLatex(0.25, 0.71, "Background Only Fit");

    // 下方残差图
    c2->cd();
    TPad* pad2 = new TPad("pad2", "pad2", 0, 0.0, 1, 0.3);
    pad2->SetTopMargin(0.02); pad2->SetBottomMargin(0.35);
    pad2->SetLeftMargin(0.15); pad2->SetRightMargin(0.05);
    pad2->SetGridy(); pad2->Draw(); pad2->cd();

    mu->setVal(0);
    RooHist*  hdata   = (RooHist*) frame->findObject("Data");
    RooCurve* hcurve  = (RooCurve*)frame->findObject("BkgLine");
    RooCurve* hband1  = (RooCurve*)frame->findObject("ModelBand1");
    RooCurve* hband2  = (RooCurve*)frame->findObject("ModelBand2");

    RooHist* hdiff     = MakeDiffHist(hdata, hcurve);
    TGraph*  diffBand1 = MakeDiffBand(hband1, hcurve);
    TGraph*  diffBand2 = MakeDiffBand(hband2, hcurve);

    RooPlot* frame2 = x->frame(Title(" "));
    if (diffBand2 && diffBand2->GetN() > 0) frame2->addObject(diffBand2, "F");
    if (diffBand1 && diffBand1->GetN() > 0) frame2->addObject(diffBand1, "F");
    if (hdiff) frame2->addPlotable(hdiff, "P");

    frame2->GetYaxis()->SetTitle("Data - Bkg");
    frame2->GetYaxis()->SetTitleSize(0.10);
    frame2->GetYaxis()->SetTitleOffset(0.6);
    frame2->GetYaxis()->SetLabelSize(0.08);
    frame2->GetYaxis()->SetRangeUser(-20, 20);
    frame2->GetYaxis()->SetNdivisions(505);
    frame2->GetXaxis()->SetTitle("Classifier Score");
    frame2->GetXaxis()->SetTitleSize(0.12);
    frame2->GetXaxis()->SetLabelSize(0.10);
    frame2->GetXaxis()->SetTitleOffset(1.1);
    frame2->Draw();

    TLine* lineZero = new TLine(x->getMin(), 0.0, x->getMax(), 0.0);
    lineZero->SetLineStyle(2); lineZero->SetLineColor(kBlack);
    lineZero->Draw("same");

    c2->SaveAs("./figures/1e11_bkg_fig_hist_fre.png");
    c2->SaveAs("/data/home/zzh/Filt_Event/Note_Writing/figures/1e11_bkg_fig_hist_fre.pdf");

    // ----------------------------------------------------------
    // 最终汇总
    // ----------------------------------------------------------
    cout << "\n===== Final Summary =====" << endl;
    if (mu_bkg)
        cout << "mu_bkg = " << mu_bkg->getVal()
             << " +/- " << mu_bkg->getError() << endl;
    cout << Form("Observed 90%% CLs Limit  : %.5f", upperLimit) << endl;
    cout << Form("Expected (-2s/-1s/med/+1s/+2s): %.4f / %.4f / %.4f / %.4f / %.4f",
                 expLimitM2, expLimitM1, expLimit, expLimitP1, expLimitP2) << endl;
    cout << ">>> All plots saved to ./figures/" << endl;
}