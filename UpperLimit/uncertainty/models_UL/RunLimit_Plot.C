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

using namespace RooFit;
using namespace RooStats;
using namespace std;

// ==========================================================
// 辅助函数：手动计算 Ratio
// ==========================================================
RooHist* MakeRatioHist(RooHist* data, RooCurve* curve) {
    if (!data || !curve) return nullptr;
    RooHist* ratio = new RooHist(*data); 
    ratio->SetName("h_ratio");
    for(int i=0; i < ratio->GetN(); i++) {
        double x, y;
        data->GetPoint(i, x, y);
        double modelVal = curve->interpolate(x);
        if (modelVal > 1e-10) {
            double r = y / modelVal;
            ratio->SetPoint(i, x, r);
            double el = data->GetErrorYlow(i);
            double eh = data->GetErrorYhigh(i);
            ratio->SetPointError(i, 0, 0, el/modelVal, eh/modelVal);
        } else {
            ratio->SetPoint(i, x, 0); 
            ratio->SetPointError(i, 0, 0, 0, 0);
        }
    }
    return ratio;
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
    
    // 全局边距
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

    const char* filename = "results/MyMonopoleAnalysis_combined_MyMeasurement_model.root";
    TFile *file = TFile::Open(filename);
    if (!file || file->IsZombie()) { cout << "Error opening file!" << endl; return; }

    RooWorkspace *w = (RooWorkspace*)file->Get("combined");
    if (!w) { cout << "Error getting workspace!" << endl; return; }

    ModelConfig *mc = (ModelConfig*)w->obj("ModelConfig");
    RooAbsData *data = w->data("obsData");
    RooRealVar *mu = (RooRealVar*)w->var("mu");
    RooRealVar* mu_bkg = (RooRealVar*)w->var("mu_bkg");
    RooRealVar *x = w->var("obs_x_single_channel"); 
    RooAbsPdf *model = w->pdf("model_single_channel");

    // ==========================================================
    // Part 1: 计算上限
    // ==========================================================
    cout << "\n>>> Starting Limit Calculation..." << endl;
    
    // [关键修正1] 在计算前重置 mu，防止之前的状态干扰
    mu->setVal(1.0);
    mu->setConstant(false);

    AsymptoticCalculator ac(*data, *mc, *mc);
    ac.SetOneSided(true);
    HypoTestInverter calc(ac);
    calc.SetConfidenceLevel(0.90);
    calc.UseCLs(true);
    calc.SetVerbose(false);

    // [关键修正2] 放宽扫描范围，防止 Limit 落在范围外导致计算失败
    // 之前 0.6 可能太窄了，如果 Expected 波动大一点就会出界
    calc.SetFixedScan(50, 0.0, 0.8); 

    HypoTestInverterResult *r = calc.GetInterval();
    double upperLimit = r->UpperLimit();
    cout << "Observed 90% Limit: " << upperLimit << endl;

    // ==========================================================
    // Part 2: 绘制 Limit Scan (Brazil Plot)
    // ==========================================================
    if (upperLimit > 1.9) {
        cout << "[WARNING] Limit hit upper bound (2.0). Scan range might be too narrow!" << endl;
    }

    TCanvas *c1 = new TCanvas("c1", "Limit Scan", 800, 600);
    c1->SetLeftMargin(0.14);
    c1->SetBottomMargin(0.12);

    HypoTestInverterPlot *plot = new HypoTestInverterPlot("HTI_Result_Plot", "", r);
    plot->Draw(); 
    // plot->Draw("CLb 2CL"); 
    // 调整一下 Brazil 图的 Y 轴范围，防止被切
    // if (plot->GetYaxis()) plot->GetYaxis()->SetRangeUser(0, 1.05);

    c1->Update(); 
    
    // 参考线
    TLine *line = new TLine(0.0, 0.1, 2.0, 0.1); // 注意这里的 x2 要和 scan max 一致
    line->SetLineColor(kRed);
    line->SetLineWidth(2);
    line->Draw("same");

    // 文字标注 (移到右上方，避开曲线)
    TLatex lat;
    lat.SetNDC();
    lat.SetTextFont(132);
    lat.SetTextSize(0.045);
    
    double txtX = 0.25; 
    double txtY = 0.88; 

    lat.DrawLatex(txtX, txtY, "LHAASO Preliminary");
    lat.DrawLatex(txtX, txtY - 0.06, "90% C.L. Upper Limit");
    lat.DrawLatex(txtX, txtY - 0.12, Form("Observed #mu < %.3f", upperLimit));
    lat.DrawLatex(txtX, txtY - 0.18, Form("Expected #mu < %.3f", r->GetExpectedUpperLimit(0)));

    c1->SaveAs("./figures/Figure_1_CLs_Scan_v3.png");
    c1->SaveAs("./figures/Figure_1_CLs_Scan_v3.pdf");

    // ==========================================================
    // Part 3: 绘制 Post-Fit 分布图
    // ==========================================================
    cout << "\n>>> Plotting Distributions..." << endl;

    // 1. Background Only Fit
    mu->setVal(0); 
    mu->setConstant(true); 
    RooFitResult *fitRes = model->fitTo(*data, Save(), PrintLevel(-1));

    RooPlot *frame = x->frame(Title(" ")); // Title 设为空格，去掉顶部的默认标题

    // 2. Data (底层参考)
    data->plotOn(frame, Name("Data"), DataError(RooAbsData::Poisson), MarkerStyle(20), MarkerSize(2));

    // 3. Background Fit + Error Band
    model->plotOn(frame, Name("ModelBand"), VisualizeError(*fitRes, 1, kFALSE), 
                 FillColor(kYellow), LineColor(kNone)); 
    model->plotOn(frame, Name("BkgLine"), LineColor(kBlue), LineWidth(2));

    // 4. [关键修正3] 信号绘制
    mu->setConstant(false);
    
    // 加大倍数，比如 500 倍，确保能看见
    double signalScale = 100.0; 
    mu->setVal(upperLimit * signalScale); 
    
    // [关键修正4] 使用通配符 "signal*" 匹配组件名
    // HistFactory 组件名通常是 signal_channelName_...
    model->plotOn(frame, Name("SignalLine"), Components("signal*"), 
                  LineColor(kRed), LineStyle(kDashed), LineWidth(3));

    // 5. Data (顶层)
    data->plotOn(frame, Name("Data"), DataError(RooAbsData::Poisson), MarkerStyle(20), MarkerSize(2));

    // --- Canvas ---
    TCanvas *c2 = new TCanvas("c2", "Fit Result", 1600, 1600);
    
    // Upper Pad
    TPad *pad1 = new TPad("pad1", "pad1", 0, 0.3, 1, 1.0);
    pad1->SetBottomMargin(0.02); 
    pad1->SetLeftMargin(0.15); // 增加左边距
    pad1->SetRightMargin(0.05);
    pad1->SetTopMargin(0.05);
    // pad1->SetLogy(1);
    pad1->Draw();
    pad1->cd();
    
    frame->GetYaxis()->SetTitle("Events / Bin");
    frame->GetYaxis()->SetTitleOffset(1.3); // 防止切字
    frame->GetXaxis()->SetLabelSize(0); 
    // frame->GetYaxis()->SetRangeUser(1e-1, data->sumEntries()*50); // 留足空间
    frame->Draw();

    // 图例 (移到右上角空白处)
    TLegend *leg = new TLegend(0.6, 0.6, 0.90, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.04);
    leg->AddEntry(frame->findObject("Data"), "Data", "pe");
    leg->AddEntry(frame->findObject("BkgLine"), "Background Fit", "l");
    leg->AddEntry(frame->findObject("ModelBand"), "Bkg. Unc. #pm 1#sigma", "f");
    leg->AddEntry(frame->findObject("SignalLine"), Form("Signal (#mu_{up} #times %.0f)", signalScale), "l");
    leg->Draw();

    lat.DrawLatex(0.25, 0.85, "LHAASO Preliminary");
    lat.DrawLatex(0.25, 0.78, "Background Only Fit");

    // Lower Pad
    c2->cd();
    TPad *pad2 = new TPad("pad2", "pad2", 0, 0.0, 1, 0.3);
    pad2->SetTopMargin(0.02);
    pad2->SetBottomMargin(0.35); // 底部留给X轴标题
    pad2->SetLeftMargin(0.15);   // 对齐上图
    pad2->SetRightMargin(0.05);
    pad2->SetGridy(); 
    pad2->Draw();
    pad2->cd();

    // Ratio
    RooHist* hdata = (RooHist*)frame->findObject("Data");
    RooCurve* hcurve = (RooCurve*)frame->findObject("BkgLine");
    RooHist *hratio = MakeRatioHist(hdata, hcurve);

    RooPlot *frame2 = x->frame(Title(" "));
    frame2->addPlotable(hratio, "P");
    
    // Ratio 样式
    frame2->GetYaxis()->SetTitle("Data / Pred.");
    frame2->GetYaxis()->SetTitleSize(0.10);
    frame2->GetYaxis()->SetTitleOffset(0.6);
    frame2->GetYaxis()->SetLabelSize(0.08);
    frame2->GetYaxis()->SetRangeUser(0.7, 1.3); 
    frame2->GetYaxis()->SetNdivisions(505);

    frame2->GetXaxis()->SetTitle("Classifier Score");
    frame2->GetXaxis()->SetTitleSize(0.12);
    frame2->GetXaxis()->SetLabelSize(0.10);
    frame2->GetXaxis()->SetTitleOffset(1.1); 
    
    frame2->Draw();

    TLine *lineOne = new TLine(x->getMin(), 1.0, x->getMax(), 1.0);
    lineOne->SetLineStyle(2);
    lineOne->SetLineColor(kBlack);
    lineOne->Draw("same");

    c2->SaveAs("./figures/Figure_2_Dist_Fit_v3.png");
    c2->SaveAs("./figures/Figure_2_Dist_Fit_v3.pdf");

    cout << "\n>>> All plots saved." << endl;
    cout << "Value: " << mu_bkg->getVal() << " +/- " << mu_bkg->getError() << endl;
    cout << "Value: " << r->GetExpectedUpperLimit(0) << endl;
}