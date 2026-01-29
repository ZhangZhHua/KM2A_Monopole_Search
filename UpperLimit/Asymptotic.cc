



// Asymptotic_validate.cc
#include "RooRealVar.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"
#include "RooAddPdf.h"
#include "RooWorkspace.h"
#include "RooPlot.h"
#include "RooFitResult.h"
#include "RooMinimizer.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/HypoTestInverter.h"
#include "RooStats/AsymptoticCalculator.h"
#include "RooStats/ProfileLikelihoodTestStat.h"
#include "RooStats/HypoTestInverterPlot.h"
#include "RooStats/FrequentistCalculator.h"
#include "RooFit.h"
// 引入新的PDF
#include "RooExponential.h"
#include "RooChebychev.h"
#include "RooGamma.h"
#include "RooBernstein.h"
// #include "RooBeta.h"
#include "RooFormulaVar.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TStyle.h"
#include "TMatrixD.h"
#include "TLatex.h"
#include "TLine.h"
#include "TLegend.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace RooFit;
using namespace RooStats;
// g++ -std=c++17 Asymptotic.cc `root-config --cflags --libs` -lRooFit -lRooFitCore -lRooStats -o Asymptotic && ./Asymptotic

int main()
{
    // -------------------------- INPUT FILES --------------------------
    std::string sigPath = "/home/zhonghua/Filt_Event/transformer/1e10_Ponly_optimized/val_preds_best_sig.txt";
    std::string bkgPath = "/home/zhonghua/Filt_Event/transformer/1e10_Ponly_optimized/val_preds_best_bkg.txt";

    std::ifstream sigFile(sigPath);
    std::ifstream bkgFile(bkgPath);
    std::vector<double> sigVals, bkgVals;
    double vv;
    while (sigFile >> vv) sigVals.push_back(vv);
    while (bkgFile >> vv) bkgVals.push_back(vv);
    sigFile.close(); bkgFile.close();
    
    // -------------------------- HISTOGRAMS --------------------------
    int nbins = 20;
    TH1F* h_sig = new TH1F("h_sig", "Signal", nbins, 0., 1.);
    TH1F* h_bkg = new TH1F("h_bkg", "Background", nbins, 0., 1.);
    for (double x : sigVals) h_sig->Fill(x);
    for (double x : bkgVals) h_bkg->Fill(x);
    if (h_sig->Integral() > 0) h_sig->Scale(1.0 / h_sig->Integral());
    if (h_bkg->Integral() > 0) h_bkg->Scale(1.0 / h_bkg->Integral());



    RooRealVar x("x", "Classifier output", 0., 1.);
    RooDataHist sigData("sigData", "Signal Data", RooArgList(x), Import(*h_sig));
    RooDataHist bkgData("bkgData", "Background Data", RooArgList(x), Import(*h_bkg));

    // 2. 定义分析型PDF及其参数
    RooRealVar c_bkg("c_bkg", "Bkg Exp Coefficient", -1.0, -50.0, 0.0);
    RooExponential bkg_pdf("bkg_pdf", "Background PDF", x, c_bkg);
    bkg_pdf.fitTo(bkgData, PrintLevel(-1));
    // SIG: RooBernstein - 专门为有界区间设计
    RooRealVar b0("b0", "Bernstein coefficient 0", 0.1, 0.0, 20.0);
    RooRealVar b1("b1", "Bernstein coefficient 1", 1.0, 0.0, 20.0);
    RooRealVar b2("b2", "Bernstein coefficient 2", 5.0, 0.0, 100.0);
    RooRealVar b3("b3", "Bernstein coefficient 3", 10.0, 0.0, 100.0);
    RooBernstein sig_pdf("sig_pdf", "Signal Bernstein PDF", x, RooArgList(b0, b1, b2, b3));

    sig_pdf.fitTo(sigData, PrintLevel(-1));
    b0.setConstant(true);
    b1.setConstant(true);
    b2.setConstant(true);
    b3.setConstant(true);
    c_bkg.setConstant(true);

    // nominal yields - you can adjust these to match your experiment
    double Nsig_nominal = 535;
    double Nbkg_nominal = 26430;
    double muMax = 0.5;
    int nPoints = 60;
    RooRealVar nb("nb", "background yield", Nbkg_nominal);
    nb.setConstant(true);
    RooRealVar mu("mu", "signal strength", muMax, 0., muMax); // wide upper bound

    RooFormulaVar ns_scaled("ns_scaled", "ns_scaled", Form("%f*@0", Nsig_nominal), RooArgList(mu));
    RooAddPdf model("model", "S+B model (extended)",RooArgList(sig_pdf, bkg_pdf), RooArgList(ns_scaled, nb));
    // RooDataSet* data = bkg_pdf.generate(x, (Int_t)Nbkg_nominal);
    // -------------------------- EXP DATA --------------------------
    // The observed upper limit is stronger than expected, due to a downward fluctuation of the background
    std::string expPath = "/home/zhonghua/Filt_Event/transformer/1e10_Ponly_optimized/exp_probs_optimized.txt";
    std::ifstream expFile(expPath);
    std::vector<double> expVals; 
    double val;
    while (expFile >> val) expVals.push_back(val);
    expFile.close();
    TH1F* h_exp = new TH1F("h_exp", "Experimental Data", 50, 0., 1.);
    for (double xval : expVals) h_exp->Fill(xval);
    double scale_factor = Int_t(Nbkg_nominal);   
    if (h_exp->Integral() > 0) h_exp->Scale(1.0 / h_exp->Integral());
    h_exp->Scale(scale_factor);
    
    RooDataHist dataHist("dataHist", "Experimental Data", RooArgList(x), Import(*h_exp));
    RooDataHist* data = &dataHist;
    RooHistPdf data_pdf("data_pdf", "Expdata HistPdf", x, dataHist);

    // -------------------------- PLOT PDFs vs binned hist --------------------------
    gStyle->SetOptStat(0);
    TFile* fout = new TFile("asymptotic_validate.root", "RECREATE");
    TCanvas* c_pdf = new TCanvas("c_pdf", "PDF Comparison", 900, 600);
        RooPlot* frame = x.frame(50);
        frame->SetTitle("Signal / Background PDF comparison (Fitted)");
        sigData.plotOn(frame, MarkerColor(kRed), MarkerStyle(20), Name("sigMCData"));
        bkgData.plotOn(frame, MarkerColor(kBlue), MarkerStyle(20), Name("bkgMCData"));
        sig_pdf.plotOn(frame, LineColor(kRed), Normalization(1), Name("sigMCPdf"));
        bkg_pdf.plotOn(frame, LineColor(kBlue), Normalization(1), Name("bkgMCPdf"));
        data_pdf.plotOn(frame, LineColor(kBlack), Normalization(1), Name("expdataPdf"));
        // data->plotOn(frame, Normalization(1), MarkerColor(kBlack), MarkerStyle(20), Name("data"));
        frame->Draw();
        TLegend* leg = new TLegend(0.6, 0.7, 0.88, 0.88);  // (x1,y1,x2,y2): 右上角位置，可调整
                leg->SetTextSize(0.03);
                leg->AddEntry(frame->findObject("sigMCData"), "Signal MC", "p");
                leg->AddEntry(frame->findObject("bkgMCData"), "Background MC", "p");
                leg->AddEntry(frame->findObject("sigMCPdf"), "Signal PDF fit", "l");
                leg->AddEntry(frame->findObject("bkgMCPdf"), "Background PDF fit", "l");
                leg->AddEntry(frame->findObject("expdataPdf"), "Exp data PDF", "l");

                leg->Draw();
    c_pdf->Write();

    TCanvas* c_stack = new TCanvas("c_counts", "Event Counts", 900, 600);
        gStyle->SetOptStat(0);
        frame->Clear();
        frame->SetTitle("Event Distribution");
        frame->GetXaxis()->SetTitle("Predicted scores by Transformer");
        frame->GetYaxis()->SetTitle("Event count");
        frame->GetXaxis()->SetTitleSize(0.05);
        frame->GetYaxis()->SetTitleSize(0.05);
        frame->GetXaxis()->SetLabelSize(0.04);
        frame->GetYaxis()->SetLabelSize(0.04);

        bkg_pdf.plotOn(frame,LineColor(kBlue),Normalization(Nbkg_nominal, RooAbsReal::NumEvent),Name("bkgPdf"));
        sig_pdf.plotOn(frame,LineColor(kRed),Normalization(Nsig_nominal * 100, RooAbsReal::NumEvent),Name("sigPdfScaled"));
    

        // sigData.plotOn(frame,MarkerColor(kRed + 1),MarkerStyle(21), Normalization(Nsig_nominal * 10, RooAbsReal::NumEvent),  Name("sigMCData"));
        // bkgData.plotOn(frame,MarkerColor(kBlue + 1),MarkerStyle(22),Normalization(Nbkg_nominal, RooAbsReal::NumEvent),Name("bkgMCData"));
        data_pdf.plotOn(frame,LineColor(kBlack),MarkerColor(kBlack),MarkerStyle(20),Normalization(Nbkg_nominal, RooAbsReal::NumEvent), Name("expdataPdf"));

        frame->Draw();
        // TLegend* leg = new TLegend(0.58, 0.68, 0.88, 0.88);
        leg->Clear();
        leg->SetTextSize(0.035);
        // leg->AddEntry(frame->findObject("bkgMCData"), "Background MC", "p");
        leg->AddEntry(frame->findObject("bkgPdf"), "Background (fit)", "l");
        leg->AddEntry(frame->findObject("sigPdfScaled"), "Signal (fit) (#times 100)", "l");
        // leg->AddEntry(frame->findObject("sigMCData"), "Signal MC (#times 100)", "p");
        leg->AddEntry(frame->findObject("expdataPdf"), "Expected Experimental data", "l");
        leg->Draw();

        // c_stack->SetLogy();
    c_stack->Write();

    // -------------------------- WORKSPACE and TWO ModelConfig --------------------------
    RooWorkspace w("w");
    w.import(x);
    w.import(model);
    w.import(b0);
    w.import(b1);
    w.import(b2);
    w.import(b3);
    w.import(nb);
    w.import(mu);
    w.import(*data);
    
    // sb ModelConfig: mu is POI and free
    ModelConfig mc_sb("ModelConfig_sb", &w);
    mc_sb.SetPdf(*w.pdf("model"));
    mc_sb.SetParametersOfInterest(RooArgSet(*w.var("mu")));
    mc_sb.SetObservables(RooArgSet(*w.var("x")));
    // mc_sb.SetObservables(RooArgSet(x)); // 使用本地变量 x
    
    // w.defineSet("nuisParams", "b0,b1,b2,b3,c_bkg,nb");
    // mc_sb.SetNuisanceParameters(*w.set("nuisParams"));

    // snapshot optional:
    RooArgSet snap_sb(mu);
    snap_sb.setRealValue("mu", 1.0); // arbitrary snapshot for sb (not used for null)
    mc_sb.SetSnapshot(snap_sb);
    w.import(mc_sb);

    // b-only ModelConfig: copy mc_sb logic but snapshot mu=0 (null hypothesis)
    ModelConfig mc_b("ModelConfig_b", &w);
    mc_b.SetPdf(*w.pdf("model"));
    mc_b.SetParametersOfInterest(RooArgSet(*w.var("mu")));
    mc_b.SetObservables(RooArgSet(x));
    // mc_b.SetNuisanceParameters(*w.set("nuisParams")); // 同样设置nuisance
    RooArgSet snap_b(mu);
    snap_b.setRealValue("mu", 0.0);
    mc_b.SetSnapshot(snap_b);
    w.import(mc_b);

    std::cout << "Performing unconditional fit (mu free) to get muHat..." << std::endl;
    RooFitResult* fitRes = model.fitTo(*data, Save(true), Extended(true), PrintLevel(1));
    std::cout << "Fitted mu = " << mu.getVal() << "  (err=" << mu.getError() << ")\n";
    std::cout << "Fitted nb = " << nb.getVal() << "  (err=" << nb.getError() << ")\n";

    // mu.setVal(0.0);

    ProfileLikelihoodTestStat pll(*mc_b.GetPdf());
    pll.SetOneSided(true);

    AsymptoticCalculator ac(*data, mc_b, mc_sb);
    ac.SetOneSided(true);

    HypoTestInverter inverter(ac, &mu, 0);
    inverter.SetConfidenceLevel(0.90);
    inverter.UseCLs(true);
    inverter.SetTestStatistic(pll);
    
    inverter.SetFixedScan(nPoints, 0., muMax);

    // -------------------------- Run Asymptotic Inversion --------------------------
    std::cout << "Running Asymptotic HypoTestInverter..." << std::endl;
    HypoTestInverterResult* result = nullptr;
    try {
        result = inverter.GetInterval();
    } catch (const std::exception &e) {
        std::cerr << "Exception while running inverter.GetInterval(): " << e.what() << std::endl;
    }
    if (!result) {
        std::cerr << "Error: HypoTestInverter returned null result." << std::endl;
        fout->Close();
        return 1;
    }

    // -------------------------- Print limits --------------------------
    double upper_obs = result->UpperLimit();
    double upper_exp = result->GetExpectedUpperLimit(0);
    double upper_exp_p1 = result->GetExpectedUpperLimit(1);
    double upper_exp_m1 = result->GetExpectedUpperLimit(-1);
    double upper_exp_p2 = result->GetExpectedUpperLimit(2);
    double upper_exp_m2 = result->GetExpectedUpperLimit(-2);

    std::cout << "\n==================== Asymptotic CLs RESULTS ====================\n";
    std::cout << "Observed 90% CL upper limit on mu = " << upper_obs << std::endl;
    std::cout << "Expected -2 sigma  = " << upper_exp_m2 << std::endl;
    std::cout << "Expected -1 sigma  = " << upper_exp_m1 << std::endl;
    std::cout << "Expected (median)  = " << upper_exp << std::endl;
    std::cout << "Expected +1 sigma  = " << upper_exp_p1 << std::endl;
    std::cout << "Expected +2 sigma  = " << upper_exp_p2 << std::endl;
    
    std::cout << "===============================================================\n";

    // -------------------------- Write plots and result safely --------------------------
    fout->cd();
    double mu_up = result->UpperLimit();
    HypoTestInverterPlot* plot = new HypoTestInverterPlot("plot", "Asymptotic CLs Limit", result);
    TCanvas* c_cls = new TCanvas("c_cls", "CLs Limit", 900, 600);
    plot->Draw("CLb 2CL");
    c_cls->SetTitle("95% CLs Upper Limit");
    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.04);
    latex.DrawLatex(0.15, 0.85, Form("90%% CL upper limit: #mu_{up} = %.4f", mu_up));
    latex.DrawLatex(0.2, 0.80, Form("-2#sigma: %.4f, -1#sigma: %.4f", upper_exp_m2, upper_exp_m1));
    latex.DrawLatex(0.2, 0.75, Form("+1#sigma: %.4f, +2#sigma: %.4f", upper_exp_p1, upper_exp_p2));
    TLine* line = new TLine(mu_up, 0.0, mu_up, 1.0);
    line->SetLineColor(kRed);
    line->SetLineStyle(2);
    line->Draw("same");
    c_cls->Write();
    plot->Write("HypoTestInverterPlot");
    result->Write("HypoTestInverterResult");
    w.writeToFile("workspace_for_Asymptotic_validate.root");
    fout->Write();
    fout->Close();

    std::cout << "Saved results to asymptotic_validate.root\n";
    // std::cout << "h_exp->Integral() = " << h_exp->Integral() << std::endl;

    // cleanup (let OS reclaim ROOT-managed objects; avoid deleting result/plot prematurely)
    delete c_pdf;
    delete c_cls;
    // do not delete result or plot if you wrote them with ROOT (they are managed by file)
    return 0;
}