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
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace RooFit;
using namespace RooStats;
// g++ -std=c++17 test.cc `root-config --cflags --libs` -lRooFit -lRooFitCore -lRooStats -o test && ./test


int main()
{
    // -------------------------- INPUT FILES --------------------------
    std::string sigPath = "/home/zhonghua/Filt_Event/transformer/1e10_Ponly_optimized/val_preds_best_sig.txt";
    std::string bkgPath = "/home/zhonghua/Filt_Event/transformer/1e10_Ponly_optimized/val_preds_best_bkg.txt";

    std::ifstream sigFile(sigPath);
    std::ifstream bkgFile(bkgPath);
    if (!sigFile.is_open() || !bkgFile.is_open()) {
        std::cerr << "Error: cannot open input files:\n  " << sigPath << "\n  " << bkgPath << std::endl;
        return 1;
    }
    std::vector<double> sigVals, bkgVals;
    double vv;
    while (sigFile >> vv) sigVals.push_back(vv);
    while (bkgFile >> vv) bkgVals.push_back(vv);
    sigFile.close(); bkgFile.close();
    if (sigVals.empty() || bkgVals.empty()) {
        std::cerr << "Error: empty input arrays." << std::endl;
        return 1;
    }

    // -------------------------- HISTOGRAMS --------------------------
    int nbins = 25;
    TH1F* h_sig = new TH1F("h_sig", "Signal", nbins, 0., 1.);
    TH1F* h_bkg = new TH1F("h_bkg", "Background", nbins, 0., 1.);
    for (double x : sigVals) h_sig->Fill(x);
    for (double x : bkgVals) h_bkg->Fill(x);
    if (h_sig->Integral() > 0) h_sig->Scale(1.0 / h_sig->Integral());
    if (h_bkg->Integral() > 0) h_bkg->Scale(1.0 / h_bkg->Integral());

    // -------------------------- PDF / MODEL --------------------------
    RooRealVar x("x", "Classifier output", 0., 1.);

    RooDataHist sigData("sigData", "Signal Data", RooArgList(x), Import(*h_sig));
    RooDataHist bkgData("bkgData", "Background Data", RooArgList(x), Import(*h_bkg));

    RooHistPdf sig_pdf("sig_pdf", "Signal HistPdf", x, sigData, 1);
    RooHistPdf bkg_pdf("bkg_pdf", "Background HistPdf", x, bkgData, 1);

    double Nsig_nominal = 535;
    double Nbkg_nominal = 26430;
    double muMax = 1;
    int nPoints = 60;
    RooRealVar nb("nb", "background yield", Nbkg_nominal);
    nb.setConstant(true);
    RooRealVar mu("mu", "signal strength", muMax, 0., muMax); // wide upper bound

    RooFormulaVar ns_scaled("ns_scaled", "ns_scaled", Form("%f*@0", Nsig_nominal), RooArgList(mu));
    RooAddPdf model("model", "S+B model (extended)",RooArgList(sig_pdf, bkg_pdf), RooArgList(ns_scaled, nb));
    // RooDataSet* data = bkg_pdf.generate(x, (Int_t)Nbkg_nominal);
    // -------------------------- EXP DATA --------------------------
    std::string expPath = "/home/zhonghua/Filt_Event/transformer/1e10_Ponly_optimized/exp_probs_optimized.txt";
    std::ifstream expFile(expPath);
    std::vector<double> expVals; 
    double val;
    while (expFile >> val) expVals.push_back(val);
    expFile.close();
    TH1F* h_exp = new TH1F("h_exp", "Experimental Data", nbins, 0., 1.);
    for (double xval : expVals) h_exp->Fill(xval);
    double scale_factor = Int_t(Nbkg_nominal);   
    if (h_exp->Integral() > 0) h_exp->Scale(1.0 / h_exp->Integral());
    h_exp->Scale(scale_factor);
    RooDataHist dataHist("dataHist", "Experimental Data", RooArgList(x), Import(*h_exp));
    RooDataHist* data = &dataHist;

    
    // -------------------------- PLOT PDFs vs binned hist --------------------------
    gStyle->SetOptStat(0);
    TFile* fout = new TFile("test_hist.root", "RECREATE");

    TCanvas* c_pdf = new TCanvas("c_pdf", "PDF Comparison", 900, 600);
    RooPlot* frame = x.frame(50);
    sigData.plotOn(frame, MarkerColor(kRed), MarkerStyle(20));
    sig_pdf.plotOn(frame, LineColor(kRed));
    bkgData.plotOn(frame, MarkerColor(kBlue), MarkerStyle(20));
    bkg_pdf.plotOn(frame, LineColor(kBlue));
    frame->SetTitle("Signal / Background PDF comparison");
    frame->Draw();
    c_pdf->Write();

    // -------------------------- WORKSPACE and TWO ModelConfig --------------------------
    RooWorkspace w("w");
    // import objects into workspace (so ModelConfig can refer to them by name)
    w.import(model);
    w.import(nb);
    w.import(mu);
    w.import(*data);

    // sb ModelConfig: mu is POI and free
    ModelConfig mc_sb("ModelConfig_sb", &w);
    mc_sb.SetPdf(*w.pdf("model"));
    mc_sb.SetParametersOfInterest(RooArgSet(*w.var("mu")));
    mc_sb.SetObservables(RooArgSet(*w.var("x"))); // Note: w.var("x") doesn't exist because x wasn't imported as a RooRealVar object; instead set observables manually below
    RooArgSet snap_sb(mu);
    snap_sb.setRealValue("mu", 1.0); // arbitrary snapshot for sb (not used for null)
    mc_sb.SetSnapshot(snap_sb);
    w.import(mc_sb);

    // b-only ModelConfig: copy mc_sb logic but snapshot mu=0 (null hypothesis)
    ModelConfig mc_b("ModelConfig_b", &w);
    mc_b.SetPdf(*w.pdf("model"));
    mc_b.SetParametersOfInterest(RooArgSet(*w.var("mu")));
    mc_b.SetObservables(RooArgSet(x));
    RooArgSet snap_b(mu);
    snap_b.setRealValue("mu", 0.0);
    mc_b.SetSnapshot(snap_b);
    w.import(mc_b);

    // -------------------------- Do an unconditional fit to data (mu free) ----------
    // This fit gives muHat and fitted ns/nb which are useful diagnostics
    std::cout << "Performing unconditional fit (mu free) to get muHat..." << std::endl;
    model.fitTo(*data, Save(true), Extended(true));
    // mu.setVal(0.0);
    ProfileLikelihoodTestStat pll(*mc_b.GetPdf());
    pll.SetOneSided(true);

    // IMPORTANT: pass sb and b ModelConfig separately
    AsymptoticCalculator ac(*data, mc_b, mc_sb);
    ac.SetOneSided(true);

    // HypoTestInverter takes AsymptoticCalculator&, RooRealVar* poi, double (unused)
    HypoTestInverter inverter(ac, &mu, 0.0);
    inverter.SetConfidenceLevel(0.90);
    inverter.UseCLs(true);
    inverter.SetTestStatistic(pll);

    inverter.SetFixedScan(nPoints, 0.0, muMax);

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
    // w.writeToFile("workspace_for_Asymptotic_validate.root");
    fout->Write();
    fout->Close();

    std::cout << "Saved results to asymptotic_validate.root\n";

    // -------------------------- OPTIONAL: quick toy-MC cross-check (commented by default) ----------
    //
    // If you want to cross-check with toys, you can enable this block.
    // It will take longer but helps validate Asymptotic result.
    //
    
    // std::cout << "Running quick toy-MC Frequentist check (300 toys) ...\n";
    // FrequentistCalculator freq(*data, mc_b, mc_sb);
    // freq.SetToys(300, 300);
    // HypoTestInverter inverter_freq(freq, &mu, 0.0);
    // inverter_freq.SetConfidenceLevel(0.90);
    // inverter_freq.UseCLs(true);
    // inverter_freq.SetTestStatistic(pll);
    // inverter_freq.SetFixedScan(20, 0.0, 0.5);
    // HypoTestInverterResult* r_freq = inverter_freq.GetInterval();
    // if (r_freq) {
    //     std::cout << "Frequentist observed upper = " << r_freq->UpperLimit() << "\n";
    //     delete r_freq;
    // } else {
    //     std::cout << "Frequentist run failed or returned null.\n";
    // }
    

    // cleanup (let OS reclaim ROOT-managed objects; avoid deleting result/plot prematurely)
    delete c_pdf;
    delete c_cls;
    // do not delete result or plot if you wrote them with ROOT (they are managed by file)
    return 0;
}