// #include "TFile.h"
// #include "RooWorkspace.h"
// #include "RooAbsPdf.h"
// #include "RooRealVar.h"
// #include "RooDataSet.h"
// #include "RooFitResult.h"
// #include "RooPlot.h"
// #include "RooHist.h"
// #include "RooCurve.h"
// #include "RooStats/ModelConfig.h"
// #include "RooStats/AsymptoticCalculator.h"
// #include "RooStats/HypoTestInverter.h"
// #include "RooStats/HypoTestInverterResult.h"
// #include "RooStats/HypoTestInverterPlot.h"
// #include "TCanvas.h"
// #include "TStyle.h"
// #include "TLegend.h"
// #include "TLatex.h"
// #include "TAxis.h"
// #include "TLine.h"
// #include "TGraph.h" // 新增头文件用于绘制误差带区域

// using namespace RooFit;
// using namespace RooStats;
// using namespace std;

// // ==========================================================
// // 辅助函数 1：手动计算 Data / Model Ratio
// // ==========================================================
// RooHist* MakeRatioHist(RooHist* data, RooCurve* curve) {
//     if (!data || !curve) return nullptr;
//     RooHist* ratio = new RooHist(*data); 
//     ratio->SetName("h_ratio");
//     for(int i=0; i < ratio->GetN(); i++) {
//         double x, y;
//         data->GetPoint(i, x, y);
//         double modelVal = curve->interpolate(x);
//         if (modelVal > 1e-10) {
//             double r = y / modelVal;
//             ratio->SetPoint(i, x, r);
//             double el = data->GetErrorYlow(i);
//             double eh = data->GetErrorYhigh(i);
//             ratio->SetPointError(i, 0, 0, el/modelVal, eh/modelVal);
//         } else {
//             ratio->SetPoint(i, x, 0); 
//             ratio->SetPointError(i, 0, 0, 0, 0);
//         }
//     }
//     return ratio;
// }

// // ==========================================================
// // 辅助函数 2：计算误差带在 Ratio 图中的相对位置 [新增]
// // ==========================================================
// TGraph* MakeRatioBand(RooCurve* band, RooCurve* central) {
//     if (!band || !central) return nullptr;
//     TGraph* ratioBand = new TGraph(band->GetN());
//     ratioBand->SetName(Form("%s_ratio", band->GetName()));
//     for(int i=0; i < band->GetN(); i++) {
//         double x, y;
//         band->GetPoint(i, x, y);
//         double y_central = central->interpolate(x);
//         if (y_central > 1e-10) {
//             // 将绝对误差带转为相对误差 (Ratio)
//             ratioBand->SetPoint(i, x, y / y_central);
//         } else {
//             ratioBand->SetPoint(i, x, 1.0); 
//         }
//     }
//     // 继承原误差带的颜色和样式
//     ratioBand->SetFillColor(band->GetFillColor());
//     ratioBand->SetFillStyle(band->GetFillStyle());
//     ratioBand->SetLineColor(band->GetLineColor());
//     ratioBand->SetLineWidth(band->GetLineWidth());
//     return ratioBand;
// }

// // ==========================================================
// // 风格设置
// // ==========================================================
// void SetAcademicStyle() {
//     gStyle->SetOptStat(0);
//     gStyle->SetOptTitle(0);
//     gStyle->SetPadTickX(1);
//     gStyle->SetPadTickY(1);
//     gStyle->SetLineWidth(2);
    
//     // 全局边距
//     gStyle->SetPadLeftMargin(0.15); 
//     gStyle->SetPadBottomMargin(0.14);
//     gStyle->SetPadRightMargin(0.05);
//     gStyle->SetPadTopMargin(0.06);

//     int font = 132; 
//     gStyle->SetTextFont(font);
//     gStyle->SetLabelFont(font, "xyz");
//     gStyle->SetTitleFont(font, "xyz");
//     gStyle->SetLabelSize(0.05, "xyz");
//     gStyle->SetTitleSize(0.06, "xyz");
//     gStyle->SetTitleOffset(1.1, "y");
//     gStyle->SetTitleOffset(1.1, "x");
//     gStyle->SetLegendFont(font);
// }

// void RunLimit_Plot() {
//     SetAcademicStyle();

//     const char* filename = "results/MyMonopoleAnalysis_combined_MyMeasurement_model.root";
//     TFile *file = TFile::Open(filename);
//     if (!file || file->IsZombie()) { cout << "Error opening file!" << endl; return; }

//     RooWorkspace *w = (RooWorkspace*)file->Get("combined");
//     if (!w) { cout << "Error getting workspace!" << endl; return; }

//     ModelConfig *mc = (ModelConfig*)w->obj("ModelConfig");
//     RooAbsData *data = w->data("obsData");
//     RooRealVar *mu = (RooRealVar*)w->var("mu");
//     RooRealVar* mu_bkg = (RooRealVar*)w->var("mu_bkg");
//     RooRealVar *x = w->var("obs_x_single_channel"); 
//     RooAbsPdf *model = w->pdf("model_single_channel");

//     // ==========================================================
//     // Part 1: 计算上限
//     // ==========================================================
//     cout << "\n>>> Starting Limit Calculation..." << endl;
    
//     mu->setVal(1.0);
//     mu->setConstant(false);

//     AsymptoticCalculator ac(*data, *mc, *mc);
//     ac.SetOneSided(true);
//     HypoTestInverter calc(ac);
//     calc.SetConfidenceLevel(0.90);
//     calc.UseCLs(true);
//     calc.SetVerbose(false);

//     calc.SetFixedScan(50, 0.0, 0.8); 

//     HypoTestInverterResult *r = calc.GetInterval();
//     double upperLimit = r->UpperLimit();
//     cout << "Observed 90% Limit: " << upperLimit << endl;

//     // ==========================================================
//     // Part 2: 绘制 Limit Scan (Brazil Plot)
//     // ==========================================================
//     if (upperLimit > 1.9) {
//         cout << "[WARNING] Limit hit upper bound (2.0). Scan range might be too narrow!" << endl;
//     }

//     TCanvas *c1 = new TCanvas("c1", "Limit Scan", 800, 600);
//     c1->SetLeftMargin(0.14);
//     c1->SetBottomMargin(0.12);

//     HypoTestInverterPlot *plot = new HypoTestInverterPlot("HTI_Result_Plot", "", r);
//     plot->Draw(); 

//     c1->Update(); 
    
//     TLine *line = new TLine(0.0, 0.1, 2.0, 0.1); 
//     line->SetLineColor(kRed);
//     line->SetLineWidth(2);
//     line->Draw("same");

//     TLatex lat;
//     lat.SetNDC();
//     lat.SetTextFont(132);
//     lat.SetTextSize(0.045);
    
//     double txtX = 0.25; 
//     double txtY = 0.88; 

//     lat.DrawLatex(txtX, txtY, "LHAASO Preliminary");
//     lat.DrawLatex(txtX, txtY - 0.06, "90% C.L. Upper Limit");
//     lat.DrawLatex(txtX, txtY - 0.12, Form("Observed #mu < %.3f", upperLimit));
//     lat.DrawLatex(txtX, txtY - 0.18, Form("Expected #mu < %.3f", r->GetExpectedUpperLimit(0)));

//     c1->SaveAs("./figures/Figure_1_CLs_Scan_v3.png");
//     c1->SaveAs("./figures/Figure_1_CLs_Scan_v3.pdf");

//     // ==========================================================
//     // Part 3: 绘制 Post-Fit 分布图
//     // ==========================================================
//     cout << "\n>>> Plotting Distributions..." << endl;

//     mu->setVal(0); 
//     mu->setConstant(true); 
//     RooFitResult *fitRes = model->fitTo(*data, Save(), PrintLevel(-1));

//     RooPlot *frame = x->frame(Title(" ")); 

//     // 2. Data (底层参考)
//     data->plotOn(frame, Name("Data"), DataError(RooAbsData::Poisson), MarkerStyle(20), MarkerSize(2));

//     // 3. Background Fit + Error Band [修改：增加 2-sigma 绿色带]
//     // 必须先画 2-sigma (大范围)，再画 1-sigma (小范围)，从而覆盖显示
//     model->plotOn(frame, Name("ModelBand2"), VisualizeError(*fitRes, 2, kFALSE), 
//                  FillColor(kGreen), LineColor(kNone)); 
    
//     model->plotOn(frame, Name("ModelBand1"), VisualizeError(*fitRes, 1, kFALSE), 
//                  FillColor(kYellow), LineColor(kNone)); 
                 
//     model->plotOn(frame, Name("BkgLine"), LineColor(kBlue), LineWidth(2));

//     // 4. 信号绘制
//     mu->setConstant(false);
//     double signalScale = 100.0; 
//     mu->setVal(upperLimit * signalScale); 
    
//     model->plotOn(frame, Name("SignalLine"), Components("signal*"), 
//                   LineColor(kRed), LineStyle(kDashed), LineWidth(3));

//     // 5. Data (顶层)
//     data->plotOn(frame, Name("Data"), DataError(RooAbsData::Poisson), MarkerStyle(20), MarkerSize(2));

//     // --- Canvas ---
//     TCanvas *c2 = new TCanvas("c2", "Fit Result", 1600, 1600);
    
//     // Upper Pad
//     TPad *pad1 = new TPad("pad1", "pad1", 0, 0.3, 1, 1.0);
//     pad1->SetBottomMargin(0.02); 
//     pad1->SetLeftMargin(0.15); 
//     pad1->SetRightMargin(0.05);
//     pad1->SetTopMargin(0.05);
//     // pad1->SetLogy(1);
//     pad1->Draw();
//     pad1->cd();
    
//     frame->GetYaxis()->SetTitle("Events / Bin");
//     frame->GetYaxis()->SetTitleOffset(1.3); 
//     frame->GetXaxis()->SetLabelSize(0); 
//     frame->Draw();

//     // 图例 [修改：加入 2-sigma 的说明]
//     TLegend *leg = new TLegend(0.6, 0.6, 0.90, 0.88);
//     leg->SetBorderSize(0);
//     leg->SetFillStyle(0);
//     leg->SetTextSize(0.04);
//     leg->AddEntry(frame->findObject("Data"), "Data", "pe");
//     leg->AddEntry(frame->findObject("BkgLine"), "Background Fit", "l");
//     leg->AddEntry(frame->findObject("ModelBand1"), "Bkg. Unc. #pm 1#sigma", "f");
//     leg->AddEntry(frame->findObject("ModelBand2"), "Bkg. Unc. #pm 2#sigma", "f");
//     leg->AddEntry(frame->findObject("SignalLine"), Form("Signal (#mu_{up} #times %.0f)", signalScale), "l");
//     leg->Draw();

//     lat.DrawLatex(0.25, 0.85, "LHAASO Preliminary");
//     lat.DrawLatex(0.25, 0.78, "Background Only Fit");

//     // Lower Pad
//     c2->cd();
//     TPad *pad2 = new TPad("pad2", "pad2", 0, 0.0, 1, 0.3);
//     pad2->SetTopMargin(0.02);
//     pad2->SetBottomMargin(0.35); 
//     pad2->SetLeftMargin(0.15);   
//     pad2->SetRightMargin(0.05);
//     pad2->SetGridy(); 
//     pad2->Draw();
//     pad2->cd();

//     // [关键修改：提取带子，构建 Ratio Band，并绘制]
//     mu->setVal(0); // 重置 mu 为 0 以计算纯背景比值
//     RooHist* hdata = (RooHist*)frame->findObject("Data");
//     RooCurve* hcurve = (RooCurve*)frame->findObject("BkgLine");
//     RooCurve* hband1 = (RooCurve*)frame->findObject("ModelBand1");
//     RooCurve* hband2 = (RooCurve*)frame->findObject("ModelBand2");

//     RooHist *hratio = MakeRatioHist(hdata, hcurve);
//     TGraph *ratioBand1 = MakeRatioBand(hband1, hcurve);
//     TGraph *ratioBand2 = MakeRatioBand(hband2, hcurve);

//     RooPlot *frame2 = x->frame(Title(" "));
    
//     // 按顺序添加到画框中：先2sigma，再1sigma，最后散点数据
//     if (ratioBand2) frame2->addObject(ratioBand2, "F"); // F 表示按区域填充(Filled)
//     if (ratioBand1) frame2->addObject(ratioBand1, "F"); 
//     frame2->addPlotable(hratio, "P");
    
//     // Ratio 样式
//     frame2->GetYaxis()->SetTitle("Data / Pred.");
//     frame2->GetYaxis()->SetTitleSize(0.10);
//     frame2->GetYaxis()->SetTitleOffset(0.6);
//     frame2->GetYaxis()->SetLabelSize(0.08);
//     frame2->GetYaxis()->SetRangeUser(0.5, 1.5); 
//     frame2->GetYaxis()->SetNdivisions(505);

//     frame2->GetXaxis()->SetTitle("Classifier Score");
//     frame2->GetXaxis()->SetTitleSize(0.12);
//     frame2->GetXaxis()->SetLabelSize(0.10);
//     frame2->GetXaxis()->SetTitleOffset(1.1); 
    
//     frame2->Draw();

//     TLine *lineOne = new TLine(x->getMin(), 1.0, x->getMax(), 1.0);
//     lineOne->SetLineStyle(2);
//     lineOne->SetLineColor(kBlack);
//     lineOne->Draw("same");

//     c2->SaveAs("./figures/Figure_2_Dist_Fit_v3.png");
//     c2->SaveAs("./figures/Figure_2_Dist_Fit_v3.pdf");

//     cout << "\n>>> All plots saved." << endl;
//     cout << "Value: " << mu_bkg->getVal() << " +/- " << mu_bkg->getError() << endl;
//     cout << "Value: " << r->GetExpectedUpperLimit(0) << endl;
// }



// #include "TFile.h"
// #include "RooWorkspace.h"
// #include "RooAbsPdf.h"
// #include "RooRealVar.h"
// #include "RooDataSet.h"
// #include "RooFitResult.h"
// #include "RooPlot.h"
// #include "RooHist.h"
// #include "RooCurve.h"
// #include "RooStats/ModelConfig.h"
// #include "RooStats/AsymptoticCalculator.h"
// #include "RooStats/HypoTestInverter.h"
// #include "RooStats/HypoTestInverterResult.h"
// #include "RooStats/HypoTestInverterPlot.h"
// #include "TCanvas.h"
// #include "TStyle.h"
// #include "TLegend.h"
// #include "TLatex.h"
// #include "TAxis.h"
// #include "TLine.h"

// using namespace RooFit;
// using namespace RooStats;
// using namespace std;

// // ==========================================================
// // 辅助函数：手动计算 Ratio
// // ==========================================================
// RooHist* MakeRatioHist(RooHist* data, RooCurve* curve) {
//     if (!data || !curve) return nullptr;
//     RooHist* ratio = new RooHist(*data); 
//     ratio->SetName("h_ratio");
//     for(int i=0; i < ratio->GetN(); i++) {
//         double x, y;
//         data->GetPoint(i, x, y);
//         double modelVal = curve->interpolate(x);
//         if (modelVal > 1e-10) {
//             double r = y / modelVal;
//             ratio->SetPoint(i, x, r);
//             double el = data->GetErrorYlow(i);
//             double eh = data->GetErrorYhigh(i);
//             ratio->SetPointError(i, 0, 0, el/modelVal, eh/modelVal);
//         } else {
//             ratio->SetPoint(i, x, 0); 
//             ratio->SetPointError(i, 0, 0, 0, 0);
//         }
//     }
//     return ratio;
// }

// // ==========================================================
// // 风格设置
// // ==========================================================
// void SetAcademicStyle() {
//     gStyle->SetOptStat(0);
//     gStyle->SetOptTitle(0);
//     gStyle->SetPadTickX(1);
//     gStyle->SetPadTickY(1);
//     gStyle->SetLineWidth(2);
    
//     // 全局边距
//     gStyle->SetPadLeftMargin(0.15); 
//     gStyle->SetPadBottomMargin(0.14);
//     gStyle->SetPadRightMargin(0.05);
//     gStyle->SetPadTopMargin(0.06);

//     int font = 132; 
//     gStyle->SetTextFont(font);
//     gStyle->SetLabelFont(font, "xyz");
//     gStyle->SetTitleFont(font, "xyz");
//     gStyle->SetLabelSize(0.05, "xyz");
//     gStyle->SetTitleSize(0.06, "xyz");
//     gStyle->SetTitleOffset(1.1, "y");
//     gStyle->SetTitleOffset(1.1, "x");
//     gStyle->SetLegendFont(font);
// }

// void RunLimit_Plot() {
//     SetAcademicStyle();

//     const char* filename = "results/MyMonopoleAnalysis_combined_MyMeasurement_model.root";
//     TFile *file = TFile::Open(filename);
//     if (!file || file->IsZombie()) { cout << "Error opening file!" << endl; return; }

//     RooWorkspace *w = (RooWorkspace*)file->Get("combined");
//     if (!w) { cout << "Error getting workspace!" << endl; return; }

//     ModelConfig *mc = (ModelConfig*)w->obj("ModelConfig");
//     RooAbsData *data = w->data("obsData");
//     RooRealVar *mu = (RooRealVar*)w->var("mu");
//     RooRealVar* mu_bkg = (RooRealVar*)w->var("mu_bkg");
//     RooRealVar *x = w->var("obs_x_single_channel"); 
//     RooAbsPdf *model = w->pdf("model_single_channel");

//     // ==========================================================
//     // Part 1: 计算上限
//     // ==========================================================
//     cout << "\n>>> Starting Limit Calculation..." << endl;
    
//     // [关键修正1] 在计算前重置 mu，防止之前的状态干扰
//     mu->setVal(1.0);
//     mu->setConstant(false);

//     AsymptoticCalculator ac(*data, *mc, *mc);
//     ac.SetOneSided(true);
//     HypoTestInverter calc(ac);
//     calc.SetConfidenceLevel(0.90);
//     calc.UseCLs(true);
//     calc.SetVerbose(false);

//     // [关键修正2] 放宽扫描范围，防止 Limit 落在范围外导致计算失败
//     // 之前 0.6 可能太窄了，如果 Expected 波动大一点就会出界
//     calc.SetFixedScan(50, 0.0, 0.8); 

//     HypoTestInverterResult *r = calc.GetInterval();
//     double upperLimit = r->UpperLimit();
//     cout << "Observed 90% Limit: " << upperLimit << endl;

//     // ==========================================================
//     // Part 2: 绘制 Limit Scan (Brazil Plot)
//     // ==========================================================
//     if (upperLimit > 1.9) {
//         cout << "[WARNING] Limit hit upper bound (2.0). Scan range might be too narrow!" << endl;
//     }

//     TCanvas *c1 = new TCanvas("c1", "Limit Scan", 800, 600);
//     c1->SetLeftMargin(0.14);
//     c1->SetBottomMargin(0.12);

//     HypoTestInverterPlot *plot = new HypoTestInverterPlot("HTI_Result_Plot", "", r);
//     plot->Draw(); 
//     // plot->Draw("CLb 2CL"); 
//     // 调整一下 Brazil 图的 Y 轴范围，防止被切
//     // if (plot->GetYaxis()) plot->GetYaxis()->SetRangeUser(0, 1.05);

//     c1->Update(); 
    
//     // 参考线
//     TLine *line = new TLine(0.0, 0.1, 2.0, 0.1); // 注意这里的 x2 要和 scan max 一致
//     line->SetLineColor(kRed);
//     line->SetLineWidth(2);
//     line->Draw("same");

//     // 文字标注 (移到右上方，避开曲线)
//     TLatex lat;
//     lat.SetNDC();
//     lat.SetTextFont(132);
//     lat.SetTextSize(0.045);
    
//     double txtX = 0.25; 
//     double txtY = 0.88; 

//     lat.DrawLatex(txtX, txtY, "LHAASO Preliminary");
//     lat.DrawLatex(txtX, txtY - 0.06, "90% C.L. Upper Limit");
//     lat.DrawLatex(txtX, txtY - 0.12, Form("Observed #mu < %.3f", upperLimit));
//     lat.DrawLatex(txtX, txtY - 0.18, Form("Expected #mu < %.3f", r->GetExpectedUpperLimit(0)));

//     c1->SaveAs("./figures/Figure_1_CLs_Scan_v3.png");
//     c1->SaveAs("./figures/Figure_1_CLs_Scan_v3.pdf");

//     // ==========================================================
//     // Part 3: 绘制 Post-Fit 分布图
//     // ==========================================================
//     cout << "\n>>> Plotting Distributions..." << endl;

//     // 1. Background Only Fit
//     mu->setVal(0); 
//     mu->setConstant(true); 
//     RooFitResult *fitRes = model->fitTo(*data, Save(), PrintLevel(-1));

//     RooPlot *frame = x->frame(Title(" ")); // Title 设为空格，去掉顶部的默认标题

//     // 2. Data (底层参考)
//     data->plotOn(frame, Name("Data"), DataError(RooAbsData::Poisson), MarkerStyle(20), MarkerSize(2));

//     // 3. Background Fit + Error Band
//     model->plotOn(frame, Name("ModelBand"), VisualizeError(*fitRes, 1, kFALSE), 
//                  FillColor(kYellow), LineColor(kNone)); 
//     model->plotOn(frame, Name("BkgLine"), LineColor(kBlue), LineWidth(2));

//     // 4. [关键修正3] 信号绘制
//     mu->setConstant(false);
    
//     // 加大倍数，比如 500 倍，确保能看见
//     double signalScale = 100.0; 
//     mu->setVal(upperLimit * signalScale); 
    
//     // [关键修正4] 使用通配符 "signal*" 匹配组件名
//     // HistFactory 组件名通常是 signal_channelName_...
//     model->plotOn(frame, Name("SignalLine"), Components("signal*"), 
//                   LineColor(kRed), LineStyle(kDashed), LineWidth(3));

//     // 5. Data (顶层)
//     data->plotOn(frame, Name("Data"), DataError(RooAbsData::Poisson), MarkerStyle(20), MarkerSize(2));

//     // --- Canvas ---
//     TCanvas *c2 = new TCanvas("c2", "Fit Result", 1600, 1600);
    
//     // Upper Pad
//     TPad *pad1 = new TPad("pad1", "pad1", 0, 0.3, 1, 1.0);
//     pad1->SetBottomMargin(0.02); 
//     pad1->SetLeftMargin(0.15); // 增加左边距
//     pad1->SetRightMargin(0.05);
//     pad1->SetTopMargin(0.05);
//     // pad1->SetLogy(1);
//     pad1->Draw();
//     pad1->cd();
    
//     frame->GetYaxis()->SetTitle("Events / Bin");
//     frame->GetYaxis()->SetTitleOffset(1.3); // 防止切字
//     frame->GetXaxis()->SetLabelSize(0); 
//     // frame->GetYaxis()->SetRangeUser(1e-1, data->sumEntries()*50); // 留足空间
//     frame->Draw();

//     // 图例 (移到右上角空白处)
//     TLegend *leg = new TLegend(0.6, 0.6, 0.90, 0.88);
//     leg->SetBorderSize(0);
//     leg->SetFillStyle(0);
//     leg->SetTextSize(0.04);
//     leg->AddEntry(frame->findObject("Data"), "Data", "pe");
//     leg->AddEntry(frame->findObject("BkgLine"), "Background Fit", "l");
//     leg->AddEntry(frame->findObject("ModelBand"), "Bkg. Unc. #pm 1#sigma", "f");
//     leg->AddEntry(frame->findObject("SignalLine"), Form("Signal (#mu_{up} #times %.0f)", signalScale), "l");
//     leg->Draw();

//     lat.DrawLatex(0.25, 0.85, "LHAASO Preliminary");
//     lat.DrawLatex(0.25, 0.78, "Background Only Fit");

//     // Lower Pad
//     c2->cd();
//     TPad *pad2 = new TPad("pad2", "pad2", 0, 0.0, 1, 0.3);
//     pad2->SetTopMargin(0.02);
//     pad2->SetBottomMargin(0.35); // 底部留给X轴标题
//     pad2->SetLeftMargin(0.15);   // 对齐上图
//     pad2->SetRightMargin(0.05);
//     pad2->SetGridy(); 
//     pad2->Draw();
//     pad2->cd();

//     // Ratio
//     RooHist* hdata = (RooHist*)frame->findObject("Data");
//     RooCurve* hcurve = (RooCurve*)frame->findObject("BkgLine");
//     RooHist *hratio = MakeRatioHist(hdata, hcurve);

//     RooPlot *frame2 = x->frame(Title(" "));
//     frame2->addPlotable(hratio, "P");
    
//     // Ratio 样式
//     frame2->GetYaxis()->SetTitle("Data / Pred.");
//     frame2->GetYaxis()->SetTitleSize(0.10);
//     frame2->GetYaxis()->SetTitleOffset(0.6);
//     frame2->GetYaxis()->SetLabelSize(0.08);
//     frame2->GetYaxis()->SetRangeUser(0.7, 1.3); 
//     frame2->GetYaxis()->SetNdivisions(505);

//     frame2->GetXaxis()->SetTitle("Classifier Score");
//     frame2->GetXaxis()->SetTitleSize(0.12);
//     frame2->GetXaxis()->SetLabelSize(0.10);
//     frame2->GetXaxis()->SetTitleOffset(1.1); 
    
//     frame2->Draw();

//     TLine *lineOne = new TLine(x->getMin(), 1.0, x->getMax(), 1.0);
//     lineOne->SetLineStyle(2);
//     lineOne->SetLineColor(kBlack);
//     lineOne->Draw("same");

//     c2->SaveAs("./figures/Figure_2_Dist_Fit_v3.png");
//     c2->SaveAs("./figures/Figure_2_Dist_Fit_v3.pdf");

//     cout << "\n>>> All plots saved." << endl;
//     cout << "Value: " << mu_bkg->getVal() << " +/- " << mu_bkg->getError() << endl;
//     cout << "Value: " << r->GetExpectedUpperLimit(0) << endl;
// }













// #include "TFile.h"
// #include "RooWorkspace.h"
// #include "RooAbsPdf.h"
// #include "RooRealVar.h"
// #include "RooDataSet.h"
// #include "RooFitResult.h"
// #include "RooPlot.h"
// #include "RooHist.h"
// #include "RooCurve.h"
// #include "RooStats/ModelConfig.h"
// #include "RooStats/AsymptoticCalculator.h"
// #include "RooStats/HypoTestInverter.h"
// #include "RooStats/HypoTestInverterResult.h"
// #include "RooStats/HypoTestInverterPlot.h"
// #include "TCanvas.h"
// #include "TStyle.h"
// #include "TLegend.h"
// #include "TLatex.h"
// #include "TAxis.h"
// #include "TLine.h"
// #include "TGraph.h" 

// using namespace RooFit;
// using namespace RooStats;
// using namespace std;

// // ==========================================================
// // 辅助函数：手动计算 Ratio
// // ==========================================================
// RooHist* MakeRatioHist(RooHist* data, RooCurve* curve) {
//     if (!data || !curve) return nullptr;
//     RooHist* ratio = new RooHist(*data); 
//     ratio->SetName("h_ratio");
//     for(int i=0; i < ratio->GetN(); i++) {
//         double x, y;
//         data->GetPoint(i, x, y);
//         double modelVal = curve->interpolate(x);
//         if (modelVal > 1e-10) {
//             double r = y / modelVal;
//             ratio->SetPoint(i, x, r);
//             double el = data->GetErrorYlow(i);
//             double eh = data->GetErrorYhigh(i);
//             ratio->SetPointError(i, 0, 0, el/modelVal, eh/modelVal);
//         } else {
//             ratio->SetPoint(i, x, 0); 
//             ratio->SetPointError(i, 0, 0, 0, 0);
//         }
//     }
//     return ratio;
// }

// // ==========================================================
// // 辅助函数：计算误差带在 Ratio 图中的相对位置 [增加终极防崩溃保护]
// // ==========================================================
// TGraph* MakeRatioBand(RooCurve* band, RooCurve* central) {
//     if (!band || !central) return nullptr;
//     // 【关键修复 1】：如果输入的 band 是空的，直接返回空，防止生成 0 个点的 TGraph
//     if (band->GetN() <= 0) return nullptr; 

//     TGraph* ratioBand = new TGraph(band->GetN());
//     ratioBand->SetName(Form("%s_ratio", band->GetName()));
//     for(int i=0; i < band->GetN(); i++) {
//         double x, y;
//         band->GetPoint(i, x, y);
//         double y_central = central->interpolate(x);
//         if (y_central > 1e-10) {
//             ratioBand->SetPoint(i, x, y / y_central);
//         } else {
//             ratioBand->SetPoint(i, x, 1.0); 
//         }
//     }
//     ratioBand->SetFillColor(band->GetFillColor());
//     ratioBand->SetFillStyle(band->GetFillStyle());
//     ratioBand->SetLineColor(band->GetLineColor());
//     ratioBand->SetLineWidth(band->GetLineWidth());
//     return ratioBand;
// }

// // ==========================================================
// // 风格设置
// // ==========================================================
// void SetAcademicStyle() {
//     gStyle->SetOptStat(0);
//     gStyle->SetOptTitle(0);
//     gStyle->SetPadTickX(1);
//     gStyle->SetPadTickY(1);
//     gStyle->SetLineWidth(2);
    
//     gStyle->SetPadLeftMargin(0.15); 
//     gStyle->SetPadBottomMargin(0.14);
//     gStyle->SetPadRightMargin(0.05);
//     gStyle->SetPadTopMargin(0.06);

//     int font = 132; 
//     gStyle->SetTextFont(font);
//     gStyle->SetLabelFont(font, "xyz");
//     gStyle->SetTitleFont(font, "xyz");
//     gStyle->SetLabelSize(0.05, "xyz");
//     gStyle->SetTitleSize(0.06, "xyz");
//     gStyle->SetTitleOffset(1.1, "y");
//     gStyle->SetTitleOffset(1.1, "x");
//     gStyle->SetLegendFont(font);
// }

// void RunLimit_Plot() {
//     SetAcademicStyle();

//     const char* filename = "results/MyMonopoleAnalysis_combined_MyMeasurement_model.root";
//     TFile *file = TFile::Open(filename);
//     if (!file || file->IsZombie()) { cout << "Error opening file!" << endl; return; }

//     RooWorkspace *w = (RooWorkspace*)file->Get("combined");
//     if (!w) { cout << "Error getting workspace!" << endl; return; }

//     ModelConfig *mc = (ModelConfig*)w->obj("ModelConfig");
//     RooAbsData *data = w->data("obsData");
//     RooRealVar *mu = (RooRealVar*)w->var("mu");
//     RooRealVar* mu_bkg = (RooRealVar*)w->var("mu_bkg");
//     RooRealVar *x = w->var("obs_x_single_channel"); 
//     RooAbsPdf *model = w->pdf("model_single_channel");

//     // ==========================================================
//     // Part 1: 计算上限
//     // ==========================================================
//     cout << "\n>>> Starting Limit Calculation..." << endl;
    
//     // 【关键修复 2】：彻底重置所有相关参数，防止连续运行时的内存状态污染
//     mu->setVal(1.0);
//     mu->setConstant(false);
//     if(mu_bkg) {
//         mu_bkg->setVal(1.0);
//         mu_bkg->setConstant(false);
//     }

//     AsymptoticCalculator ac(*data, *mc, *mc);
//     ac.SetOneSided(true);
//     HypoTestInverter calc(ac);
//     calc.SetConfidenceLevel(0.90);
//     calc.UseCLs(true);
//     calc.SetVerbose(false);

//     calc.SetFixedScan(70, 0.0, 0.7); 

//     HypoTestInverterResult *r = calc.GetInterval();
//     double upperLimit = r->UpperLimit();
//     cout << "Observed 90% Limit: " << upperLimit << endl;

//     // ==========================================================
//     // Part 2: 绘制 Limit Scan (Brazil Plot)
//     // ==========================================================
//     TCanvas *c1 = new TCanvas("c1", "Limit Scan", 1000, 800);
//     c1->SetLeftMargin(0.14);
//     // 【修改点 1】：增大底部边缘留白，防止 "mu" 被切掉 (从 0.12 改为 0.15)
//     c1->SetBottomMargin(0.15); 
//     c1->SetRightMargin(0.08);

//     if (r && r->ArraySize() > 0) {
//         HypoTestInverterPlot *plot = new HypoTestInverterPlot("HTI_Result_Plot", "", r);
//         plot->Draw(); 
//     } else {
//         cout << "\n[WARNING] HypoTestInverterResult is empty! Skipping Limit Scan plot." << endl;
//     }

//     // 必须 Update，让 ROOT 生成并计算坐标轴的实际范围
//     c1->Update(); 
    
//     // 【修改点 2】：动态获取真实坐标轴的范围，防止红线越界
//     double xMin = gPad->GetUxmin();
//     double xMax = gPad->GetUxmax();
    
//     // 使用动态获取的 xMax 画线
//     TLine *line = new TLine(xMin, 0.1, xMax, 0.1); 
//     line->SetLineColor(kRed);
//     line->SetLineWidth(2);
//     line->Draw("same");

//     TLatex lat;
//     lat.SetNDC();
//     lat.SetTextFont(132);
//     lat.SetTextSize(0.04);
    
//     double txtX = 0.23; 
//     double txtY = 0.88; 

//     // 获取 Expected Limits
//     double expLimit = r->GetExpectedUpperLimit(0);
//     double expLimitPlus1 = r->GetExpectedUpperLimit(1);
//     double expLimitMinus1 = r->GetExpectedUpperLimit(-1);

//     // 打印到终端
//     cout << "Expected Limit (-1 sigma): " << expLimitMinus1 << endl;
//     cout << "Expected Limit (Median)  : " << expLimit << endl;
//     cout << "Expected Limit (+1 sigma): " << expLimitPlus1 << endl;

//     // 画到图表上
//     lat.DrawLatex(txtX, txtY, "LHAASO Preliminary");
//     lat.DrawLatex(txtX, txtY - 0.06, "Monopole Analysis, #gamma = 10^{5}");
//     lat.DrawLatex(txtX, txtY - 0.11, "90% C.L. Upper Limit");
//     lat.DrawLatex(txtX, txtY - 0.17, Form("Observed #mu < %.3f", upperLimit));
    
//     // 添加 Expected 和 1 sigma 信息
//     lat.DrawLatex(txtX, txtY - 0.22, Form("Expected #mu < %.3f", expLimit));
//     lat.DrawLatex(txtX, txtY - 0.27, Form("Expected #pm 1#sigma: [%.3f, %.3f]", expLimitMinus1, expLimitPlus1));

//     c1->SaveAs("./figures/1e10_CLs_scan.png");
//     c1->SaveAs("./figures/1e10_CLs_scan.pdf");

//     // ==========================================================
//     // Part 3: 绘制 Post-Fit 分布图
//     // ==========================================================
//     cout << "\n>>> Plotting Distributions..." << endl;

//     mu->setVal(0); 
//     mu->setConstant(true); 
//     RooFitResult *fitRes = model->fitTo(*data, Save(), PrintLevel(-1));

//     RooPlot *frame = x->frame(Title(" ")); 

//     data->plotOn(frame, Name("Data"), DataError(RooAbsData::Poisson), MarkerStyle(20), MarkerSize(2));

//     // 绘制 2-sigma 和 1-sigma 带
//     model->plotOn(frame, Name("ModelBand2"), VisualizeError(*fitRes, 2, kTRUE), 
//                  FillColor(kYellow), LineColor(kNone)); 
    
//     model->plotOn(frame, Name("ModelBand1"), VisualizeError(*fitRes, 1, kTRUE), 
//                  FillColor(kGreen), LineColor(kNone)); 
                 
//     model->plotOn(frame, Name("BkgLine"), LineColor(kBlue), LineWidth(2));

//     // 信号绘制
//     mu->setConstant(false);
//     double signalScale = 100.0; 
//     mu->setMax(upperLimit * signalScale * 2.0);
//     mu->setVal(upperLimit * signalScale); 
    
//     model->plotOn(frame, Name("SignalLine"), Components("signal*"), 
//                   LineColor(kRed), LineStyle(kDashed), LineWidth(3));

//     data->plotOn(frame, Name("Data"), DataError(RooAbsData::Poisson), MarkerStyle(20), MarkerSize(1), LineWidth(1));

//     // --- Canvas ---
//     TCanvas *c2 = new TCanvas("c2", "Fit Result", 1600, 1600);
    
//     TPad *pad1 = new TPad("pad1", "pad1", 0, 0.3, 1, 1.0);
//     pad1->SetBottomMargin(0.02); 
//     pad1->SetLeftMargin(0.15); 
//     pad1->SetRightMargin(0.05);
//     pad1->SetTopMargin(0.05);
//     pad1->Draw();
//     pad1->cd();
    
//     frame->GetYaxis()->SetTitle("Events / Bin");
//     frame->GetYaxis()->SetTitleOffset(1.3); 
//     frame->GetXaxis()->SetLabelSize(0); 
//     frame->Draw();

//     TLegend *leg = new TLegend(0.6, 0.6, 0.90, 0.88);
//     leg->SetBorderSize(0);
//     leg->SetFillStyle(0);
//     leg->SetTextSize(0.04);
//     leg->AddEntry(frame->findObject("Data"), "Data", "pe");
//     leg->AddEntry(frame->findObject("BkgLine"), "Background Fit", "l");
//     leg->AddEntry(frame->findObject("ModelBand1"), "Bkg. Unc. #pm 1#sigma", "f");
    
//     // 【关键修复 4】：仅当真的画出了 2-sigma 带时才加图例
//     if (frame->findObject("ModelBand2")) {
//         leg->AddEntry(frame->findObject("ModelBand2"), "Bkg. Unc. #pm 2#sigma", "f");
//     }
    
//     leg->AddEntry(frame->findObject("SignalLine"), Form("Signal (#mu_{up} #times %.0f)", signalScale), "l");
//     leg->Draw();

//     lat.DrawLatex(0.25, 0.85, "LHAASO Preliminary");
//     lat.DrawLatex(0.25, 0.78, "Monopole Analysis, #gamma = 10^{5}");
//     lat.DrawLatex(0.25, 0.71, "Background Only Fit");

//     // Lower Pad
//     c2->cd();
//     TPad *pad2 = new TPad("pad2", "pad2", 0, 0.0, 1, 0.3);
//     pad2->SetTopMargin(0.02);
//     pad2->SetBottomMargin(0.35); 
//     pad2->SetLeftMargin(0.15);   
//     pad2->SetRightMargin(0.05);
//     pad2->SetGridy(); 
//     pad2->Draw();
//     pad2->cd();

//     mu->setVal(0); 
//     RooHist* hdata = (RooHist*)frame->findObject("Data");
//     RooCurve* hcurve = (RooCurve*)frame->findObject("BkgLine");
//     RooCurve* hband1 = (RooCurve*)frame->findObject("ModelBand1");
//     RooCurve* hband2 = (RooCurve*)frame->findObject("ModelBand2");

//     RooHist *hratio = MakeRatioHist(hdata, hcurve);
//     TGraph *ratioBand1 = MakeRatioBand(hband1, hcurve);
//     TGraph *ratioBand2 = MakeRatioBand(hband2, hcurve);

//     RooPlot *frame2 = x->frame(Title(" "));
    
//     // 【关键修复 5】：绘制 Ratio 带子前，双重检查点数是否大于 0
//     if (ratioBand2 && ratioBand2->GetN() > 0) frame2->addObject(ratioBand2, "F"); 
//     if (ratioBand1 && ratioBand1->GetN() > 0) frame2->addObject(ratioBand1, "F"); 
    
//     if (hratio) frame2->addPlotable(hratio, "P");
    
//     frame2->GetYaxis()->SetTitle("Data / Pred.");
//     frame2->GetYaxis()->SetTitleSize(0.10);
//     frame2->GetYaxis()->SetTitleOffset(0.6);
//     frame2->GetYaxis()->SetLabelSize(0.08);
//     frame2->GetYaxis()->SetRangeUser(0.7, 1.3); 
//     frame2->GetYaxis()->SetNdivisions(505);

//     frame2->GetXaxis()->SetTitle("Classifier Score");
//     frame2->GetXaxis()->SetTitleSize(0.12);
//     frame2->GetXaxis()->SetLabelSize(0.10);
//     frame2->GetXaxis()->SetTitleOffset(1.1); 
    
//     frame2->Draw();

//     TLine *lineOne = new TLine(x->getMin(), 1.0, x->getMax(), 1.0);
//     lineOne->SetLineStyle(2);
//     lineOne->SetLineColor(kBlack);
//     lineOne->Draw("same");

//     c2->SaveAs("./figures/1e10_80bins_bkg_fig_hist.png");
//     c2->SaveAs("./figures/1e10_80bins_bkg_fig_hist.pdf");

//     cout << "\n>>> All plots saved." << endl;
//     if(mu_bkg) cout << "Value: " << mu_bkg->getVal() << " +/- " << mu_bkg->getError() << endl;
    
//     // 返回中心值和 1, 2 sigma 的 expected limit
//     cout << "Expected Limit (-2 sigma): " << r->GetExpectedUpperLimit(-2) << endl;
//     cout << "Expected Limit (-1 sigma): " << r->GetExpectedUpperLimit(-1) << endl;
//     cout << "Expected Limit (Median)  : " << r->GetExpectedUpperLimit(0)  << endl;
//     cout << "Expected Limit (+1 sigma): " << r->GetExpectedUpperLimit(1)  << endl;
//     cout << "Expected Limit (+2 sigma): " << r->GetExpectedUpperLimit(2)  << endl;
// }


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

    calc.SetFixedScan(70, 0.0, 0.7); 

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

    cout << "Expected Limit (-1 sigma): " << expLimitMinus1 << endl;
    cout << "Expected Limit (Median)  : " << expLimit << endl;
    cout << "Expected Limit (+1 sigma): " << expLimitPlus1 << endl;

    lat.DrawLatex(txtX, txtY, "LHAASO Preliminary");
    lat.DrawLatex(txtX, txtY - 0.06, "Monopole Analysis, #gamma = 10^{5}");
    lat.DrawLatex(txtX, txtY - 0.11, "90% C.L. Upper Limit");
    lat.DrawLatex(txtX, txtY - 0.17, Form("Observed #mu < %.3f", upperLimit));
    lat.DrawLatex(txtX, txtY - 0.22, Form("Expected #mu < %.3f", expLimit));
    lat.DrawLatex(txtX, txtY - 0.27, Form("Expected #pm 1#sigma: [%.3f, %.3f]", expLimitMinus1, expLimitPlus1));

    c1->SaveAs("./figures/1e10_CLs_scan.png");
    c1->SaveAs("/data/home/zzh/Filt_Event/Note_Writing/figures/1e10_CLs_scan.pdf");

    // ==========================================================
    // Part 3: 绘制 Post-Fit 分布图
    // ==========================================================
    cout << "\n>>> Plotting Distributions..." << endl;

    mu->setVal(0); 
    mu->setConstant(true); 
    RooFitResult *fitRes = model->fitTo(*data, Save(), PrintLevel(-1));

    RooPlot *frame = x->frame(Title(" ")); 

    data->plotOn(frame, Name("Data"), DataError(RooAbsData::Poisson), MarkerStyle(20), MarkerSize(1));

    model->plotOn(frame, Name("ModelBand2"), VisualizeError(*fitRes, 2, kTRUE), 
                 FillColor(kYellow), LineColor(kNone)); 
    
    model->plotOn(frame, Name("ModelBand1"), VisualizeError(*fitRes, 1, kTRUE), 
                 FillColor(kGreen), LineColor(kNone)); 
                 
    model->plotOn(frame, Name("BkgLine"), LineColor(kBlue), LineWidth(2));

    mu->setConstant(false);
    double signalScale = 1.0; 
    mu->setMax(upperLimit * signalScale * 2.0);
    mu->setVal(upperLimit * signalScale); 
    
    model->plotOn(frame, Name("SignalLine"), Components("signal*"), 
                  LineColor(kRed), LineStyle(kDashed), LineWidth(3));

    data->plotOn(frame, Name("Data"), DataError(RooAbsData::Poisson), MarkerStyle(20), MarkerSize(1), LineWidth(1));
    // log scale
    
    
    // --- Canvas ---
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


    frame->GetYaxis()->SetTitle("Events / Bin");
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

    // Lower Pad (现在用于显示 Data - Bkg)
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
    RooHist* hdata = (RooHist*)frame->findObject("Data");
    RooCurve* hcurve = (RooCurve*)frame->findObject("BkgLine");
    RooCurve* hband1 = (RooCurve*)frame->findObject("ModelBand1");
    RooCurve* hband2 = (RooCurve*)frame->findObject("ModelBand2");

    // 使用新的 Diff 函数
    RooHist *hdiff = MakeDiffHist(hdata, hcurve);
    TGraph *diffBand1 = MakeDiffBand(hband1, hcurve);
    TGraph *diffBand2 = MakeDiffBand(hband2, hcurve);

    RooPlot *frame2 = x->frame(Title(" "));
    
    // 按顺序添加 Diff 带子和数据点
    if (diffBand2 && diffBand2->GetN() > 0) frame2->addObject(diffBand2, "F"); 
    if (diffBand1 && diffBand1->GetN() > 0) frame2->addObject(diffBand1, "F"); 
    if (hdiff) frame2->addPlotable(hdiff, "P");
    
    // 更新 Y 轴设置
    frame2->GetYaxis()->SetTitle("Data - Bkg");
    frame2->GetYaxis()->SetTitleSize(0.10);
    frame2->GetYaxis()->SetTitleOffset(0.6);
    frame2->GetYaxis()->SetLabelSize(0.08);
    // 取消硬编码的 RangeUser，让 ROOT 根据残差大小自动缩放，或者你可以根据需要手动解除下面的注释并设置范围
    frame2->GetYaxis()->SetRangeUser(-60, 60); 
    frame2->GetYaxis()->SetNdivisions(505);

    frame2->GetXaxis()->SetTitle("Classifier Score");
    frame2->GetXaxis()->SetTitleSize(0.12);
    frame2->GetXaxis()->SetLabelSize(0.10);
    frame2->GetXaxis()->SetTitleOffset(1.1); 
    
    frame2->Draw();

    // 更新参考线至 y = 0
    TLine *lineZero = new TLine(x->getMin(), 0.0, x->getMax(), 0.0);
    lineZero->SetLineStyle(2);
    lineZero->SetLineColor(kBlack);
    lineZero->Draw("same");

    c2->SaveAs("./figures/1e10_bkg_fig_hist.png");
    c2->SaveAs("/data/home/zzh/Filt_Event/Note_Writing/figures/1e10_bkg_fig_hist.pdf");

    cout << "\n>>> All plots saved." << endl;
    if(mu_bkg) cout << "Value: " << mu_bkg->getVal() << " +/- " << mu_bkg->getError() << endl;
    cout << "Observed 90% Limit: " << upperLimit << endl;
    cout << "Expected Limit (-2 sigma): " << r->GetExpectedUpperLimit(-2) << endl;
    cout << "Expected Limit (-1 sigma): " << r->GetExpectedUpperLimit(-1) << endl;
    cout << "Expected Limit (Median)  : " << r->GetExpectedUpperLimit(0)  << endl;
    cout << "Expected Limit (+1 sigma): " << r->GetExpectedUpperLimit(1)  << endl;
    cout << "Expected Limit (+2 sigma): " << r->GetExpectedUpperLimit(2)  << endl;
}




