

#include "TFile.h"
#include "RooWorkspace.h"
#include "RooAbsPdf.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooFitResult.h"
#include "RooPlot.h"
#include "RooHist.h"
#include "RooCurve.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/AsymptoticCalculator.h"
#include "RooStats/HypoTestInverter.h"
#include "RooStats/HypoTestInverterResult.h"
#include "RooStats/HypoTestInverterPlot.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TAxis.h"
#include "TLine.h"
#include "TGraph.h" 

using namespace RooFit;
using namespace RooStats;
using namespace std;

// ==========================================================
// 辅助函数：手动计算 Data - Bkg (Difference)
// ==========================================================
RooHist* MakeDiffHist(RooHist* data, RooCurve* curve) {
    if (!data || !curve) return nullptr;
    RooHist* diff = new RooHist(*data); 
    diff->SetName("h_diff");
    for(int i=0; i < diff->GetN(); i++) {
        double x, y;
        data->GetPoint(i, x, y);
        double modelVal = curve->interpolate(x);
        
        // 计算残差 Data - Model
        diff->SetPoint(i, x, y - modelVal);
        
        // 误差棒保持绝对大小不变
        double el = data->GetErrorYlow(i);
        double eh = data->GetErrorYhigh(i);
        diff->SetPointError(i, 0, 0, el, eh);
    }
    return diff;
}

// ==========================================================
// 辅助函数：计算误差带在 Diff 图中的相对位置 (围绕 0)
// ==========================================================
TGraph* MakeDiffBand(RooCurve* band, RooCurve* central) {
    if (!band || !central) return nullptr;
    if (band->GetN() <= 0) return nullptr; 

    TGraph* diffBand = new TGraph(band->GetN());
    diffBand->SetName(Form("%s_diff", band->GetName()));
    for(int i=0; i < band->GetN(); i++) {
        double x, y;
        band->GetPoint(i, x, y);
        double y_central = central->interpolate(x);
        
        // 将绝对误差带转为残差误差 (y_band - y_central)
        diffBand->SetPoint(i, x, y - y_central); 
    }
    diffBand->SetFillColor(band->GetFillColor());
    diffBand->SetFillStyle(band->GetFillStyle());
    diffBand->SetLineColor(band->GetLineColor());
    diffBand->SetLineWidth(band->GetLineWidth());
    return diffBand;
}
// ==========================================================
// 辅助函数：计算 Data / Bkg (Ratio)
// ==========================================================
RooHist* MakeRatioHist(RooHist* data, RooCurve* curve) {
    if (!data || !curve) return nullptr;
    RooHist* ratio = new RooHist(*data);
    ratio->SetName("h_ratio");
    
    const double EPS = 1e-9; // 防止除以 0

    for(int i=0; i < ratio->GetN(); i++) {
        double x, y_data;
        data->GetPoint(i, x, y_data);
        double y_bkg = curve->interpolate(x);
        
        if (y_bkg > EPS) {
            // 计算比值：Data / Bkg
            ratio->SetPoint(i, x, y_data / y_bkg);
            
            // 误差传递：sigma_ratio = sigma_data / Bkg
            double el = data->GetErrorYlow(i) / y_bkg;
            double eh = data->GetErrorYhigh(i) / y_bkg;
            ratio->SetPointError(i, 0, 0, el, eh);
        } else {
            // 如果背景为 0，将比值设为 0（或者你也可以设为 1，取决于绘图偏好）
            // 这里设为 0 并去掉误差棒，避免视觉干扰
            ratio->SetPoint(i, x, 0);
            ratio->SetPointError(i, 0, 0, 0, 0);
        }
    }
    return ratio;
}

// ==========================================================
// 辅助函数：计算误差带在 Ratio 图中的相对位置 (围绕 1.0)
// ==========================================================
TGraph* MakeRatioBand(RooCurve* band, RooCurve* central) {
    if (!band || !central || band->GetN() <= 0) return nullptr;

    TGraph* ratioBand = new TGraph(band->GetN());
    ratioBand->SetName(Form("%s_ratio", band->GetName()));
    
    const double EPS = 1e-9;

    for(int i=0; i < band->GetN(); i++) {
        double x, y_band;
        band->GetPoint(i, x, y_band);
        double y_central = central->interpolate(x);
        
        if (y_central > EPS) {
            // 比值带：Band_Value / Central_Value
            ratioBand->SetPoint(i, x, y_band / y_central);
        } else {
            // 背景为 0 时，将误差带收缩至 1.0
            ratioBand->SetPoint(i, x, 1.0);
        }
    }
    ratioBand->SetFillColor(band->GetFillColor());
    ratioBand->SetFillStyle(band->GetFillStyle());
    ratioBand->SetLineColor(band->GetLineColor());
    ratioBand->SetLineWidth(band->GetLineWidth());
    return ratioBand;
}
// ==========================================================
// 风格设置
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

void RunLimit_Plot() {
    SetAcademicStyle();

    const char* filename = "results/Monopole_Search_combined_Monopole_Limit_Setting_model.root";
    TFile *file = TFile::Open(filename);
    if (!file || file->IsZombie()) { cout << "Error opening file!" << endl; return; }

    RooWorkspace *w = (RooWorkspace*)file->Get("combined");
    if (!w) { cout << "Error getting workspace!" << endl; return; }

    ModelConfig *mc = (ModelConfig*)w->obj("ModelConfig");
    RooAbsData *data = w->data("obsData");
    RooRealVar *mu = (RooRealVar*)w->var("mu");
    RooRealVar* mu_bkg = (RooRealVar*)w->var("mu_bkg");
    RooRealVar *x = w->var("obs_x_LHAASO_Monopole_Channel"); 
    RooAbsPdf *model = w->pdf("model_LHAASO_Monopole_Channel");

    // ==========================================================
    // Part 1: 计算上限
    // ==========================================================
    cout << "\n>>> Starting Limit Calculation..." << endl;
    
    mu->setVal(1.0);
    mu->setConstant(false);
    if(mu_bkg) {
        mu_bkg->setVal(1.0);
        mu_bkg->setConstant(false);
    }

    AsymptoticCalculator ac(*data, *mc, *mc);
    ac.SetOneSided(true);
    HypoTestInverter calc(ac);
    calc.SetConfidenceLevel(0.90);
    calc.UseCLs(true);
    calc.SetVerbose(false);

    calc.SetFixedScan(50, 0.0, 0.01); 

    HypoTestInverterResult *r = calc.GetInterval();
    double upperLimit = r->UpperLimit();
    cout << "Observed 90% Limit: " << upperLimit << endl;

    // ==========================================================
    // Part 2: 绘制 Limit Scan (Brazil Plot)
    // ==========================================================
    TCanvas *c1 = new TCanvas("c1", "Limit Scan", 1000, 800);
    c1->SetLeftMargin(0.14);
    c1->SetBottomMargin(0.15); 
    c1->SetRightMargin(0.08);

    if (r && r->ArraySize() > 0) {
        HypoTestInverterPlot *plot = new HypoTestInverterPlot("HTI_Result_Plot", "", r);
        plot->Draw(); 
    } else {
        cout << "\n[WARNING] HypoTestInverterResult is empty! Skipping Limit Scan plot." << endl;
    }

    c1->Update(); 
    
    double xMin = gPad->GetUxmin();
    double xMax = gPad->GetUxmax();
    
    TLine *line = new TLine(xMin, 0.1, xMax, 0.1); 
    line->SetLineColor(kRed);
    line->SetLineWidth(2);
    line->Draw("same");

    TLatex lat;
    lat.SetNDC();
    lat.SetTextFont(132);
    lat.SetTextSize(0.04);
    
    double txtX = 0.23; 
    double txtY = 0.88; 

    double expLimit = r->GetExpectedUpperLimit(0);
    double expLimitPlus1 = r->GetExpectedUpperLimit(1);
    double expLimitMinus1 = r->GetExpectedUpperLimit(-1);
    double expLimitPlus2 = r->GetExpectedUpperLimit(2);
    double expLimitMinus2 = r->GetExpectedUpperLimit(-2);

    cout << "Expected Limit (-1 sigma): " << expLimitMinus1 << endl;
    cout << "Expected Limit (Median)  : " << expLimit << endl;
    cout << "Expected Limit (+1 sigma): " << expLimitPlus1 << endl;
    cout << "Expected Limit (+2 sigma): " << expLimitPlus2 << endl;
    cout << "Expected Limit (-2 sigma): " << expLimitMinus2 << endl;

    lat.DrawLatex(txtX, txtY, "LHAASO Preliminary");
    lat.DrawLatex(txtX, txtY - 0.06, "Monopole Analysis, #gamma = 10^{5}");
    lat.DrawLatex(txtX, txtY - 0.11, "90% C.L. Upper Limit");
    lat.DrawLatex(txtX, txtY - 0.17, Form("Observed #mu < %.3f", upperLimit));
    lat.DrawLatex(txtX, txtY - 0.22, Form("Expected #mu < %.3f", expLimit));
    lat.DrawLatex(txtX, txtY - 0.27, Form("Expected #pm 1#sigma: [%.3f, %.3f]", expLimitMinus1, expLimitPlus1));
    lat.DrawLatex(txtX, txtY - 0.32, Form("Expected #pm 2#sigma: [%.3f, %.3f]", expLimitMinus2, expLimitPlus2));


    c1->SaveAs("./figures/1e11_CLs_scan.png");
    c1->SaveAs("/data/home/zzh/Filt_Event/Note_Writing/figures/1e11_CLs_scan.pdf");

    // ==========================================================
    // Part 3: 绘制 Post-Fit 分布图
    // ==========================================================
    cout << "\n>>> Plotting Distributions..." << endl;
    w->Print();
    mu->setVal(0); 
    mu->setConstant(true); 
    RooFitResult *fitRes = model->fitTo(*data, Save(), PrintLevel(-1));
    if (fitRes->status() != 0) {
    cout << "WARNING: Fit did not converge! Status = " << fitRes->status() << endl;
    } else {
        cout << ">>> Fit converged successfully (Status 0)." << endl;
    }
    RooPlot *frame = x->frame(Title(" ")); 

    data->plotOn(frame, Name("Data"), DataError(RooAbsData::Poisson), MarkerStyle(20), MarkerSize(1));

    model->plotOn(frame, Name("ModelBand2"), VisualizeError(*fitRes, 2, kTRUE), 
                 FillColor(kYellow), LineColor(kNone)); 
    
    model->plotOn(frame, Name("ModelBand1"), VisualizeError(*fitRes, 1, kTRUE), 
                 FillColor(kGreen), LineColor(kNone)); 
                 
    model->plotOn(frame, Name("BkgLine"), LineColor(kBlue), LineWidth(2));

    mu->setConstant(false);
    double signalScale = 10.0; 
    mu->setMax(upperLimit * signalScale * 2.0);
    mu->setVal(upperLimit * signalScale); 
    
    model->plotOn(frame, Name("SignalLine"), Components("Signal*"), 
                  LineColor(kRed), LineStyle(kDashed), LineWidth(3));

    data->plotOn(frame, Name("Data"), DataError(RooAbsData::Poisson), MarkerStyle(20), MarkerSize(1), LineWidth(1));
    // log scale

    
    
    // // --- Canvas ---
    TCanvas *c2 = new TCanvas("c2", "Fit Result", 1600, 1600);
    
    TPad *pad1 = new TPad("pad1", "pad1", 0, 0.3, 1, 1.0);
    pad1->SetBottomMargin(0.02); 
    pad1->SetLeftMargin(0.15); 
    pad1->SetRightMargin(0.05);
    pad1->SetTopMargin(0.05);
    pad1->Draw();
    pad1->cd();
    // frame->SetMinimum(50);
    // frame->SetMaximum(1e4);
    pad1->SetLogy(1);


    frame->GetYaxis()->SetTitle("Events / Avg. Bin Width");
    frame->GetYaxis()->SetTitleOffset(1.3); 
    frame->GetXaxis()->SetLabelSize(0); 
    frame->Draw();

    TLegend *leg = new TLegend(0.6, 0.6, 0.90, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.04);
    leg->AddEntry(frame->findObject("Data"), "Data", "pe");
    leg->AddEntry(frame->findObject("BkgLine"), "Background Fit", "l");
    leg->AddEntry(frame->findObject("ModelBand1"), "Bkg. Unc. #pm 1#sigma", "f");
    
    if (frame->findObject("ModelBand2")) {
        leg->AddEntry(frame->findObject("ModelBand2"), "Bkg. Unc. #pm 2#sigma", "f");
    }
    
    leg->AddEntry(frame->findObject("SignalLine"), Form("Signal (#mu_{up} #times %.0f)", signalScale), "l");
    leg->Draw();

    lat.DrawLatex(0.25, 0.85, "LHAASO Preliminary");
    lat.DrawLatex(0.25, 0.78, "Monopole Analysis, #gamma = 10^{5}");
    lat.DrawLatex(0.25, 0.71, "Background Only Fit");

    // // Lower Pad (现在用于显示 Data - Bkg)
    // c2->cd();
    // TPad *pad2 = new TPad("pad2", "pad2", 0, 0.0, 1, 0.3);
    // pad2->SetTopMargin(0.02);
    // pad2->SetBottomMargin(0.35); 
    // pad2->SetLeftMargin(0.15);   
    // pad2->SetRightMargin(0.05);
    // pad2->SetGridy(); 
    // pad2->Draw();
    // pad2->cd();

    // mu->setVal(0); 
    // RooHist* hdata = (RooHist*)frame->findObject("Data");
    // RooCurve* hcurve = (RooCurve*)frame->findObject("BkgLine");
    // RooCurve* hband1 = (RooCurve*)frame->findObject("ModelBand1");
    // RooCurve* hband2 = (RooCurve*)frame->findObject("ModelBand2");

    // // 使用新的 Diff 函数
    // RooHist *hdiff = MakeDiffHist(hdata, hcurve);
    // TGraph *diffBand1 = MakeDiffBand(hband1, hcurve);
    // TGraph *diffBand2 = MakeDiffBand(hband2, hcurve);

    // RooPlot *frame2 = x->frame(Title(" "));
    
    // // 按顺序添加 Diff 带子和数据点
    // if (diffBand2 && diffBand2->GetN() > 0) frame2->addObject(diffBand2, "F"); 
    // if (diffBand1 && diffBand1->GetN() > 0) frame2->addObject(diffBand1, "F"); 
    // if (hdiff) frame2->addPlotable(hdiff, "P");
    
    // // 更新 Y 轴设置
    // frame2->GetYaxis()->SetTitle("Data - Bkg");
    // frame2->GetYaxis()->SetTitleSize(0.10);
    // frame2->GetYaxis()->SetTitleOffset(0.6);
    // frame2->GetYaxis()->SetLabelSize(0.08);
    // // 取消硬编码的 RangeUser，让 ROOT 根据残差大小自动缩放，或者你可以根据需要手动解除下面的注释并设置范围
    // frame2->GetYaxis()->SetRangeUser(-60, 60); 
    // frame2->GetYaxis()->SetNdivisions(505);

    // frame2->GetXaxis()->SetTitle("Classifier Score");
    // frame2->GetXaxis()->SetTitleSize(0.12);
    // frame2->GetXaxis()->SetLabelSize(0.10);
    // frame2->GetXaxis()->SetTitleOffset(1.1); 
    
    // frame2->Draw();

    // // 更新参考线至 y = 0
    // TLine *lineZero = new TLine(x->getMin(), 0.0, x->getMax(), 0.0);
    // lineZero->SetLineStyle(2);
    // lineZero->SetLineColor(kBlack);
    // lineZero->Draw("same");

    // Lower Pad (Ratio)
    c2->cd();
    TPad *pad2 = new TPad("pad2", "pad2", 0, 0.0, 1, 0.3);
    pad2->SetTopMargin(0.02);
    pad2->SetBottomMargin(0.35); 
    pad2->SetLeftMargin(0.15);   
    pad2->SetRightMargin(0.05);
    pad2->SetGridy(); 
    pad2->Draw();
    pad2->cd();

    mu->setVal(0); 
    RooHist* hdata   = (RooHist*)frame->findObject("Data");
    RooCurve* hcurve = (RooCurve*)frame->findObject("BkgLine");
    RooCurve* hband1 = (RooCurve*)frame->findObject("ModelBand1");
    RooCurve* hband2 = (RooCurve*)frame->findObject("ModelBand2");

    // 调用新的 Ratio 函数
    RooHist *hratio      = MakeRatioHist(hdata, hcurve);
    TGraph *ratioBand1   = MakeRatioBand(hband1, hcurve);
    TGraph *ratioBand2   = MakeRatioBand(hband2, hcurve);

    RooPlot *frame2 = x->frame(Title(" "));
    
    // 绘制顺序：先 2-sigma 颜色，再 1-sigma 颜色，最后数据点
    if (ratioBand2) frame2->addObject(ratioBand2, "F"); 
    if (ratioBand1) frame2->addObject(ratioBand1, "F"); 
    if (hratio)     frame2->addPlotable(hratio, "P");
    
    // Y 轴设置
    frame2->GetYaxis()->SetTitle("Data / Bkg");
    frame2->GetYaxis()->SetTitleSize(0.10);
    frame2->GetYaxis()->SetTitleOffset(0.6);
    frame2->GetYaxis()->SetLabelSize(0.08);
    
    // Ratio 图通常设置在 0 到 2 之间，或者根据你的偏好收窄
    frame2->GetYaxis()->SetRangeUser(0.0, 2.0); 
    frame2->GetYaxis()->SetNdivisions(505);

    // X 轴设置
    frame2->GetXaxis()->SetTitle("Classifier Score");
    frame2->GetXaxis()->SetTitleSize(0.12);
    frame2->GetXaxis()->SetLabelSize(0.10);
    frame2->GetXaxis()->SetTitleOffset(1.1); 
    
    frame2->Draw();

    // 参考线：y = 1.0 (代表数据与背景完全吻合)
    TLine *lineOne = new TLine(x->getMin(), 1.0, x->getMax(), 1.0);
    lineOne->SetLineStyle(2);
    lineOne->SetLineColor(kBlack);
    lineOne->Draw("same");

    c2->SaveAs("./figures/1e11_bkg_fig_hist.png");
    c2->SaveAs("/data/home/zzh/Filt_Event/Note_Writing/figures/1e11_bkg_fig_hist.pdf");

    cout << "\n>>> All plots saved." << endl;
    if(mu_bkg) cout << "Value: " << mu_bkg->getVal() << " +/- " << mu_bkg->getError() << endl;
    cout << "Observed 90% Limit: " << upperLimit << endl;
    cout << "Expected Limit (-2 sigma): " << r->GetExpectedUpperLimit(-2) << endl;
    cout << "Expected Limit (-1 sigma): " << r->GetExpectedUpperLimit(-1) << endl;
    cout << "Expected Limit (Median)  : " << r->GetExpectedUpperLimit(0)  << endl;
    cout << "Expected Limit (+1 sigma): " << r->GetExpectedUpperLimit(1)  << endl;
    cout << "Expected Limit (+2 sigma): " << r->GetExpectedUpperLimit(2)  << endl;
}




