// Frequentist_validate.cc
#include "RooRealVar.h"
#include "RooDataHist.h"
#include "RooAddPdf.h"
#include "RooWorkspace.h"
#include "RooPlot.h"
#include "RooFitResult.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/HypoTestInverter.h"
#include "RooStats/FrequentistCalculator.h"
#include "RooStats/ProfileLikelihoodTestStat.h"
#include "RooStats/SamplingDistribution.h"
#include "RooStats/HypoTestInverterPlot.h"
#include "RooFormulaVar.h"
#include "RooExponential.h"
#include "RooBernstein.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TStyle.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace RooFit;
using namespace RooStats;
// g++ -std=c++17 Frequentist.cc `root-config --cflags --libs` -lRooFit -lRooFitCore -lRooStats -o Frequentist && ./Frequentist

int main()
{
    // -------------------------- INPUT FILES --------------------------
    std::string sigPath = "/home/zhonghua/Filt_Event/transformer/1e10_Ponly_optimized/val_preds_best_sig.txt";
    std::string bkgPath = "/home/zhonghua/Filt_Event/transformer/1e10_Ponly_optimized/val_preds_best_bkg.txt";

    std::ifstream sigFile(sigPath);
    std::ifstream bkgFile(bkgPath);
    if (!sigFile.is_open() || !bkgFile.is_open()) {
        std::cerr << "Error: cannot open input files." << std::endl;
        return 1;
    }

    std::vector<double> sigVals, bkgVals;
    double vv;
    while (sigFile >> vv) sigVals.push_back(vv);
    while (bkgFile >> vv) bkgVals.push_back(vv);
    sigFile.close(); bkgFile.close();

    int nbins = 20;
    TH1F* h_sig = new TH1F("h_sig", "Signal", nbins, 0., 1.);
    TH1F* h_bkg = new TH1F("h_bkg", "Background", nbins, 0., 1.);
    for (auto x : sigVals) h_sig->Fill(x);
    for (auto x : bkgVals) h_bkg->Fill(x);
    h_sig->Scale(1.0 / h_sig->Integral());
    h_bkg->Scale(1.0 / h_bkg->Integral());

    RooRealVar x("x", "Classifier output", 0., 1.);
    RooDataHist sigData("sigData", "Signal Data", RooArgList(x), Import(*h_sig));
    RooDataHist bkgData("bkgData", "Background Data", RooArgList(x), Import(*h_bkg));

    RooRealVar c_bkg("c_bkg", "Bkg Exp Coeff", -1.0, -50., 0.);
    RooExponential bkg_pdf("bkg_pdf", "Background PDF", x, c_bkg);
    bkg_pdf.fitTo(bkgData, PrintLevel(-1));

    RooRealVar b0("b0", "Bernstein c0", 0.1, 0., 20.);
    RooRealVar b1("b1", "Bernstein c1", 1.0, 0., 20.);
    RooRealVar b2("b2", "Bernstein c2", 5.0, 0., 100.);
    RooRealVar b3("b3", "Bernstein c3", 10.0, 0., 100.);
    RooBernstein sig_pdf("sig_pdf", "Signal Bernstein PDF", x, RooArgList(b0,b1,b2,b3));
    sig_pdf.fitTo(sigData, PrintLevel(-1));

    // Freeze shapes
    b0.setConstant(true); b1.setConstant(true); b2.setConstant(true); b3.setConstant(true); c_bkg.setConstant(true);

    double Nsig_nominal = 535;
    double Nbkg_nominal = 26430;
    double muMax = 1;

    RooRealVar nb("nb", "background yield", Nbkg_nominal);
    nb.setConstant(true);
    RooRealVar mu("mu", "signal strength", muMax, 0., muMax);
    RooFormulaVar ns_scaled("ns_scaled", "ns_scaled", Form("%f*@0", Nsig_nominal), RooArgList(mu));

    RooAddPdf model("model", "S+B model (extended)", RooArgList(sig_pdf,bkg_pdf), RooArgList(ns_scaled, nb));
    // RooDataSet* data = bkg_pdf.generate(x, (Int_t)Nbkg_nominal);
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


    // -------------------------- WORKSPACE --------------------------
    RooWorkspace w("w");
    w.import(x); w.import(model);
    w.import(mu); w.import(nb);
    w.import(*data);

    // ModelConfig for S+B (alt)
    ModelConfig mc_sb("ModelConfig_sb", &w);
    mc_sb.SetPdf(model);
    mc_sb.SetParametersOfInterest(RooArgSet(mu));
    mc_sb.SetObservables(RooArgSet(x));
    RooArgSet snap_sb(mu);
    snap_sb.setRealValue("mu", 1.0);
    mc_sb.SetSnapshot(snap_sb);
    w.import(mc_sb);

    // ModelConfig for B-only (null)
    ModelConfig mc_b("ModelConfig_b", &w);
    mc_b.SetPdf(model);
    mc_b.SetParametersOfInterest(RooArgSet(mu));
    mc_b.SetObservables(RooArgSet(x));
    RooArgSet snap_b(mu);
    snap_b.setRealValue("mu", 0.0);
    mc_b.SetSnapshot(snap_b);
    w.import(mc_b);

    // -------------------------- FrequentistCalculator --------------------------
    std::cout << "Performing Frequentist (ToyMC) CLs calculation..." << std::endl;
    ProfileLikelihoodTestStat pll(*mc_b.GetPdf());
    pll.SetOneSided(true);

    FrequentistCalculator fc(*data, mc_b, mc_sb);
    fc.SetToys(300, 300);  // (toys for sb, toys for b)

    HypoTestInverter inverter(fc, &mu);
    inverter.SetConfidenceLevel(0.90);
    inverter.UseCLs(true);
    inverter.SetFixedScan(10, 0., muMax);
    inverter.SetTestStatistic(pll);
    inverter.SetVerbose(true);

    HypoTestInverterResult* result = inverter.GetInterval();
    if (!result) { std::cerr << "Error: null result!" << std::endl; return 1; }

    // -------------------------- Output --------------------------
    double upper_obs = result->UpperLimit();
    double upper_exp = result->GetExpectedUpperLimit(0);
    double upper_exp_p1 = result->GetExpectedUpperLimit(1);
    double upper_exp_m1 = result->GetExpectedUpperLimit(-1);

    std::cout << "\n==================== Frequentist (ToyMC) CLs RESULTS ====================\n";
    std::cout << "Observed 90% CL upper limit on mu = " << upper_obs << std::endl;
    std::cout << "Expected (median)  = " << upper_exp << std::endl;
    std::cout << "Expected +1 sigma  = " << upper_exp_p1 << std::endl;
    std::cout << "Expected -1 sigma  = " << upper_exp_m1 << std::endl;
    std::cout << "=======================================================================\n";

    // -------------------------- Save plot --------------------------
    TFile* fout = new TFile("frequentist_validate.root", "RECREATE");
    HypoTestInverterPlot* plot = new HypoTestInverterPlot("plot", "Frequentist CLs Limit", result);
    TCanvas* c_cls = new TCanvas("c_cls", "CLs Limit", 900, 600);
    plot->Draw("CLs");
    c_cls->Write();
    plot->Write("HypoTestInverterPlot");
    result->Write("HypoTestInverterResult");
    w.writeToFile("workspace_for_Frequentist_validate.root");
    fout->Close();

    std::cout << "Saved to frequentist_validate.root\n";
    return 0;
}