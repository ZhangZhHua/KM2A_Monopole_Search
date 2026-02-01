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
// g++ -std=c++17 Evaluate_exp.cc `root-config --cflags --libs` -lRooFit -lRooFitCore -lRooStats -o Evaluate_exp && ./Evaluate_exp
int main(){
    TFile* f = TFile::Open("workspace_for_Asymptotic_validate.root");
    RooWorkspace* w = (RooWorkspace*)f->Get("w");
    std::ifstream dataFile("/home/zhonghua/Filt_Event/transformer/1e10_Ponly/val_preds_best_bkg_5.txt");
    std::vector<double> vals;
    double xval;
    while (dataFile >> xval) vals.push_back(xval);
    dataFile.close();

    int nbins = 10;
    TH1F* h_data = new TH1F("h_data", "Observed Data", nbins, 0., 1.);
    for (double v : vals) h_data->Fill(v);
    if (h_data->Integral() > 0) h_data->Scale(1.0 / h_data->Integral() * 117300 );

    RooRealVar* x = w->var("x");
    RooDataHist dataHist("dataHist", "Observed Data", RooArgList(*x), Import(*h_data));

    ModelConfig* mc_sb = (ModelConfig*)w->obj("ModelConfig_sb");
    ModelConfig* mc_b  = (ModelConfig*)w->obj("ModelConfig_b");
    
    AsymptoticCalculator ac(dataHist, *mc_b, *mc_sb);
    ac.SetOneSided(true);

    HypoTestInverter inverter(ac, w->var("mu"));
    inverter.SetConfidenceLevel(0.90);
    inverter.UseCLs(true);
    inverter.SetFixedScan(10, 0., 0.5);
    HypoTestInverterResult* result = inverter.GetInterval();

    double upper_obs = result->UpperLimit();
    double upper_exp = result->GetExpectedUpperLimit(0);
    double upper_exp_p1 = result->GetExpectedUpperLimit(1);
    double upper_exp_m1 = result->GetExpectedUpperLimit(-1);

    std::cout << "\n==================== Asymptotic CLs RESULTS ====================\n";
    std::cout << "Observed 90% CL upper limit on mu = " << upper_obs << std::endl;
    std::cout << "Expected (median)  = " << upper_exp << std::endl;
    std::cout << "Expected +1 sigma  = " << upper_exp_p1 << std::endl;
    std::cout << "Expected -1 sigma  = " << upper_exp_m1 << std::endl;
    std::cout << "===============================================================\n";

    TFile* fout = new TFile("asymptotic_exp.root", "RECREATE");
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
    TLine* line = new TLine(mu_up, 0.0, mu_up, 1.0);
    line->SetLineColor(kRed);
    line->SetLineStyle(2);
    line->Draw("same");
    c_cls->Write();
    plot->Write("HypoTestInverterPlot");
    result->Write("HypoTestInverterResult");
    // w->writeToFile("workspace_for_Asymptotic_validate.root");
    fout->Write();
    fout->Close();
}