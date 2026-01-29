// ###############################################################################
// #  产生用于分类的数据  km2a.root and rec_km2a.root ==> zzh.root for next steps
// #  作者:  Zhang Zhonghua 
// #  时间:  2025.7.30
// ###############################################################################

// (rec_)km2a.root dir: 
// Simulated Gamma:    /eos/user/z/zhangzhonghua/DATfile/gamma/1e4_1e5/DAT000001.root  or  rec_DAT000001.root
// Simulated Proton:   /eos/user/z/zhangzhonghua/DATfile/proton/1e4_1e5/DAT000001.root  or  rec_DAT000001.root
// Simulated Monopole: /eos/user/z/zhangzhonghua/paper_v0/monopole/QGSII_m5/1e9_1e10/DAT000001.root  or  rec_DAT000001.root

// zzh.root dir:
// Simulated Gamma: /home/lhaaso/zhangzhonghua/KM2AMCrec_V3/data/Dataset_Filted/Simulation/gamma/test

// Usage: ./Data4Classify $DataSource $Mode $Km2aRootFile $RecRootFile $FiltedRootFile(output) $log10E_low $log10E_high
//  ($DataSource: simulation or experiment, $mode: 0: gamma, 2: proton, 1: monopole) 


#include <TFile.h>
#include <TTree.h>
#include <TGraph.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TClonesArray.h>
#include <TStyle.h>
#include <fstream>   
#include <iostream>
#include <vector>
#include <string> 
#include <cmath>     
#include "LHEvent.h"     
// #include "KM2AEvent.h"


const double PI = 3.14159265358979323846;

double recE_func_NpE1(double NpE1, double theta, double a, double b, double c, double d){
    double costh = cos(theta);
    return a * log10(NpE1 + 1) + b / costh + c / (costh * costh) + d;   // log10(TeV)
}

double cal_recE_NpE1(double NpE1, double theta){
    
    if (NpE1 <= -1.0) {
        std::cerr << "Error: NpE1 must be > -1. Got NpE1 = " << NpE1 << std::endl;
        return -9999;  // 标记非法值或抛出异常
    }

    double logNpe = log10(NpE1 + 1);
    double a, b, c, d;
   
    if (logNpe < 1.5) {
        a = 0.19214949;
        b = -2.11486897;
        c = 0.66640887;
        d = 1.9398595195402297;
    } else if (logNpe < 2.0) {
        a = 0.706771;
        b = -0.9212677;
        c = 0.17995167;
        d = 0.5547663532352864;
    } else if (logNpe < 2.5) {
        a = 0.89591788;
        b = -1.62958701;
        c = 0.50492983;
        d = 0.5658062950848612;
    } else if (logNpe < 3.0) {
        a = 1.10378692;
        b = -1.72242266;
        c = 0.51862491;
        d = 0.07801886572555161;
    } else if (logNpe < 4.0) {
        a = 1.14228936;
        b = -5.83435264;
        c = 2.20019744;
        d = 2.2660410774664;
    } else {
        std::cerr << "Warning: log10(NpE1 + 1) = " << logNpe << " is out of defined range." << std::endl;
        return -9999;
    }

    return recE_func_NpE1(NpE1, theta, a, b, c, d);
}


int filter0(double rec_r, double rec_theta,double NpE1,double NpE2){
    if (rec_r > 600){
        return -9999;
    }
    if ((rec_theta<0) && (rec_theta>50*PI/18*5)){
        return -9999;
    }
    if (NpE1<0 || NpE2<0){
        return -9999;
    }
    if (NpE1/NpE2<2){
        return -9999;
    }
    return 1;
}

int ProcessEvents(  const std::string& datasource,
                    const std::string& km2arootfile, 
                    const std::string& recrootfile,
                    const std::string& filtedrootfile,
                    double recE_limit_low,
                    double recE_limit_high
                    ) 
{
            int ev_n;
            double mjd,dt;
            double NpE1,NpE2,NpE3,NuW1,NuW2,NuW3;
            double trueE,theta,phi,x,y;
            double rec_theta,rec_phi,rec_x,rec_y,rec_r,recE;
            float  rec_Echi,rec_Mchi,rec_Ex,rec_Ey,rec_Ez,rec_Eage,rec_Mage;
            int rec_Endf,rec_Mndf,NhitE,NhitM,NtrigE,NtrigE2,NfiltE,NfiltM;

        if (datasource == "simulation"){
            std::string simu_krootfile=km2arootfile;
            std::string simu_recrootfile=recrootfile;
            std::string simu_filted_rootfile=filtedrootfile;

           
            // 打开(rec)km2a.root文件
            TFile simufile(simu_krootfile.c_str(), "READ");
            TFile recfile(simu_recrootfile.c_str(), "READ");
            if (simufile.IsZombie()) {
            std::cerr << "Error: Cannot open file " << simu_krootfile << "\n";
            return 1;
            }
            if (recfile.IsZombie()) {
                std::cerr << "Error: Cannot open file " << "\n";
                return 1;
            }
            // 链接TTree and event
            TTree* rec_tree = nullptr;
            recfile.GetObject("Rec", rec_tree);
            LHRecEvent* rec_event = nullptr;
            rec_tree->SetBranchAddress("Rec", &rec_event);
            
            TTree* simu_tree = nullptr;
            simufile.GetObject("event", simu_tree);
            LHEvent* simu_event = nullptr;
            simu_tree->SetBranchAddress("Event", &simu_event);
            // 创建zzh.root文件
            TFile filtedfile(simu_filted_rootfile.c_str(), "RECREATE"); // "RECREATE"模式会覆盖已存在的文件
            TTree* filted_tree = new TTree("filted_tree", "Filted Simulated Event");
            LHFiltedEvent* filted_event = new LHFiltedEvent();
            filted_tree->Branch("FiltedEvent", &filted_event);

            // 遍历TTree,筛选,读取数据
            Long64_t nentries = rec_tree->GetEntries();
            for (Long64_t i = 0; i < nentries; ++i) {
                rec_tree->GetEntry(i);
                simu_tree->GetEntry(i);
                trueE = rec_event->GetE()/1000.0;  //TeV  simu only
               
                ev_n = rec_event->GetEvN();
                mjd = rec_event->GetMjd();
                dt = rec_event->GetDt();
                
                NpE1 = rec_event->GetNpE1();
                NpE2 = rec_event->GetNpE2();
                NpE3 = rec_event->GetNpE3();
                NuW1 = rec_event->GetNuW1();
                NuW2 = rec_event->GetNuW2();
                NuW3 = -10;

                NhitE = rec_event->GetNhitE();
                NhitM = rec_event->GetNhitM();
                NtrigE = rec_event->GetNtrigE();
                NtrigE2 = -10;
                NfiltE = rec_event->GetNfiltE();
                NfiltM = rec_event->GetNfiltM();

                rec_theta = rec_event->GetRec_theta();
                rec_phi = rec_event->GetRec_phi();
                rec_x = rec_event->GetRec_x();
                rec_y = rec_event->GetRec_y();
                rec_r = std::sqrt(std::pow(rec_x,2)+std::pow(rec_y,2));
                rec_Ex = rec_event->GetRecEx();
                rec_Ey = rec_event->GetRecEy();
                rec_Ez = rec_event->GetRecEz();


                rec_Eage = rec_event->GetRecEage();
                rec_Mage = rec_event->GetRecMage();
                rec_Echi = rec_event->GetRecEchi();
                rec_Endf = rec_event->GetRecEndf();
                rec_Mchi = rec_event->GetRecMchi();
                rec_Mndf = rec_event->GetRecMndf();

                // 进行筛选
                if (filter0(rec_r,rec_theta,NpE1,NpE2)==-9999){
                    continue;
                }
                // 重建能量
                recE = cal_recE_NpE1(NpE1,rec_theta);
                // recE_limit_low = 0.63809;   // log10(E GeV)=9,log10(recE_low TeV)=0.63809,recE_high=1.28334 
                // recE_limit_high = 1.28334;
                if (recE<recE_limit_low || recE>recE_limit_high){
                    continue;
                }


                // 保存数据
                filted_event->SetEvN(simu_event->GetEvN());
                filted_event->SetTrueE(simu_event->GetE()/1000.0);
                filted_event->SetId(simu_event->GetId());
                filted_event->SetTheta(simu_event->GetTheta());
                filted_event->SetPhi(simu_event->GetPhi());
                filted_event->SetX(simu_event->GetCorex());
                filted_event->SetY(simu_event->GetCorey());
                
                filted_event->SetMjd(mjd);
                filted_event->SetDt(dt);
                
                
                filted_event->SetNhitE(NhitE);
                filted_event->SetNhitM(NhitM);
                filted_event->SetNtrigE(NtrigE);
                filted_event->SetNtrigE2(NtrigE2);
                filted_event->SetNfiltE(NfiltE);
                filted_event->SetNfiltM(NfiltM);

                filted_event->SetNpE1(NpE1);
                filted_event->SetNpE2(NpE2);
                filted_event->SetNpE3(NpE3);
                filted_event->SetNuW1(NuW1);
                filted_event->SetNuW2(NuW2);
                filted_event->SetNuW3(NuW3);
                filted_event->SetRecTheta(rec_theta);
                filted_event->SetRecPhi(rec_phi);
                filted_event->SetRecX(rec_x);
                filted_event->SetRecY(rec_y);
                filted_event->SetRecR(rec_r);

                filted_event->SetRecEx(rec_Ex);
                filted_event->SetRecEy(rec_Ey);
                filted_event->SetRecEz(rec_Ez);
                filted_event->SetRecEage(rec_Eage);
                filted_event->SetRecMage(rec_Mage);
                filted_event->SetRecEchi(rec_Echi);
                filted_event->SetRecEndf(rec_Endf);
                filted_event->SetRecMchi(rec_Mchi);
                filted_event->SetRecMndf(rec_Mndf);

                filted_event->SetRecE(recE);
                
                filted_event->SetLog10E_limit_low(recE_limit_low);
                filted_event->SetLog10E_limit_high(recE_limit_high);
        
                // === 拷贝 Hits ===
                TClonesArray* hitsE = simu_event->GetHitsE();
                for (int j = 0; j < hitsE->GetEntries(); ++j) {
                    LHHit* hit = (LHHit*)hitsE->At(j);
                    filted_event->AddHitE(hit->GetId(), hit->GetTime(), hit->GetPe(), hit->GetNp());
                }
                TClonesArray* hitsM = simu_event->GetHitsM();
                for (int j = 0; j < hitsM->GetEntries(); ++j) {
                    LHHit* hit = (LHHit*)hitsM->At(j);
                    filted_event->AddHitM(hit->GetId(), hit->GetTime(), hit->GetPe(), hit->GetNp());
                }
                
                filted_tree->Fill(); 
                filted_event->Clear(); 
            
            }

            filtedfile.cd();
            filted_tree->Write();

            recfile.Close();
            simufile.Close();
            filtedfile.Close();
            
            return 0;
        }
        else if (datasource == "experiment"){
            std::string exp_recfile = km2arootfile;  // 实验数据只有一个输入文件
            
            // 打开实验数据文件 (包含原始事件和重建事件)
            TFile expfile(exp_recfile.c_str(), "READ");
            if (expfile.IsZombie()) {
                std::cerr << "Error: Cannot open experimental data file " << exp_recfile << "\n";
                return 1;
            }
            
            // 获取原始事件树 (LHEvent)
            TTree* exp_lhevent_tree = nullptr;
                expfile.GetObject("lhtree", exp_lhevent_tree);
                if (!exp_lhevent_tree) {
                    std::cerr << "Error: Cannot find 'event' tree in experimental data file\n";
                    return 1;
                }
                LHEvent* exp_lhevent = nullptr;
                exp_lhevent_tree->SetBranchAddress("LHEvent", &exp_lhevent);
                
            // 获取重建事件树 (LHRecEvent)
            TTree* exp_rec_tree = nullptr;
                expfile.GetObject("rectree", exp_rec_tree);
                if (!exp_rec_tree) {
                    std::cerr << "Error: Cannot find 'Rec' tree in experimental data file\n";
                    return 1;
                }
                LHRecEvent* exp_rec_event = nullptr;
                exp_rec_tree->SetBranchAddress("RecEvent", &exp_rec_event);
                
            // 确保两个树条目数相同
            Long64_t nentries = exp_lhevent_tree->GetEntries();
            if (nentries != exp_rec_tree->GetEntries()) {
                std::cerr << "Error: Mismatched number of entries between event tree (" << nentries 
                        << ") and rec tree (" << exp_rec_tree->GetEntries() << ")\n";
                return 1;
            }
            
            // 创建过滤后的事件文件
            TFile filtedfile(filtedrootfile.c_str(), "RECREATE");
            TTree* filted_tree = new TTree("filted_tree", "Filted Experimental Events");
            LHFiltedEvent* filted_event = new LHFiltedEvent();
            filted_tree->Branch("FiltedEvent", &filted_event);
            
            // 遍历所有事件
            for (Long64_t i = 0; i < nentries; ++i) {
                exp_lhevent_tree->GetEntry(i);
                exp_rec_tree->GetEntry(i);
                
                // 获取重建参数
                ev_n = exp_rec_event->GetEvN();
                mjd = exp_rec_event->GetMjd();
                dt = exp_rec_event->GetDt();
                
                NpE1 = exp_rec_event->GetNpE1();
                NpE2 = exp_rec_event->GetNpE2();
                NpE3 = exp_rec_event->GetNpE3();
                NuW1 = exp_rec_event->GetNuW1();
                NuW2 = exp_rec_event->GetNuW2();
                NuW3 = -10;

                NhitE = exp_rec_event->GetNhitE();
                NhitM = exp_rec_event->GetNhitM();
                NtrigE = exp_rec_event->GetNtrigE();
                NtrigE2 = -10;
                NfiltE = exp_rec_event->GetNfiltE();
                NfiltM = exp_rec_event->GetNfiltM();

                rec_theta = exp_rec_event->GetRec_theta();
                rec_phi = exp_rec_event->GetRec_phi();
                rec_x = exp_rec_event->GetRec_x();
                rec_y = exp_rec_event->GetRec_y();
                rec_r = std::sqrt(std::pow(rec_x,2)+std::pow(rec_y,2));
                rec_Ex = exp_rec_event->GetRecEx();
                rec_Ey = exp_rec_event->GetRecEy();
                rec_Ez = exp_rec_event->GetRecEz();


                rec_Eage = exp_rec_event->GetRecEage();
                rec_Mage = exp_rec_event->GetRecMage();
                rec_Echi = exp_rec_event->GetRecEchi();
                rec_Endf = exp_rec_event->GetRecEndf();
                rec_Mchi = exp_rec_event->GetRecMchi();
                rec_Mndf = exp_rec_event->GetRecMndf();
                
                // 进行初步筛选
                if (filter0(rec_r, rec_theta, NpE1, NpE2) == -9999) {
                    continue;
                }
                
                // 重建能量
                recE = cal_recE_NpE1(NpE1, rec_theta);
                
                // 能量范围筛选
                if (recE < recE_limit_low || recE > recE_limit_high) {
                    continue;
                }
                
                // 保存事件信息
                filted_event->SetEvN(exp_lhevent->GetEvN());
                filted_event->SetTrueE(-1.0);  // 实验数据没有真实能量
                filted_event->SetId(exp_lhevent->GetId());
                filted_event->SetTheta(exp_lhevent->GetTheta());
                filted_event->SetPhi(exp_lhevent->GetPhi());
                filted_event->SetX(exp_lhevent->GetCorex());
                filted_event->SetY(exp_lhevent->GetCorey());

                filted_event->SetMjd(mjd);
                filted_event->SetDt(dt);
                
                filted_event->SetNhitE(NhitE);
                filted_event->SetNhitM(NhitM);
                filted_event->SetNtrigE(NtrigE);
                filted_event->SetNtrigE2(NtrigE2);
                filted_event->SetNfiltE(NfiltE);
                filted_event->SetNfiltM(NfiltM);

                filted_event->SetNpE1(NpE1);
                filted_event->SetNpE2(NpE2);
                filted_event->SetNpE3(NpE3);
                filted_event->SetNuW1(NuW1);
                filted_event->SetNuW2(NuW2);
                filted_event->SetNuW3(NuW3);

                filted_event->SetRecTheta(rec_theta);
                filted_event->SetRecPhi(rec_phi);
                filted_event->SetRecX(rec_x);
                filted_event->SetRecY(rec_y);
                filted_event->SetRecR(rec_r);

                filted_event->SetRecEx(rec_Ex);
                filted_event->SetRecEy(rec_Ey);
                filted_event->SetRecEz(rec_Ez);
                filted_event->SetRecEage(rec_Eage);
                filted_event->SetRecMage(rec_Mage);
                filted_event->SetRecEchi(rec_Echi);
                filted_event->SetRecEndf(rec_Endf);
                filted_event->SetRecMchi(rec_Mchi);
                filted_event->SetRecMndf(rec_Mndf);

                filted_event->SetRecE(recE);
                filted_event->SetLog10E_limit_low(recE_limit_low);
                filted_event->SetLog10E_limit_high(recE_limit_high);
                
                // 拷贝电子探测器击中
                TClonesArray* hitsE = exp_lhevent->GetHitsE();
                for (int j = 0; j < hitsE->GetEntries(); ++j) {
                    LHHit* hit = dynamic_cast<LHHit*>(hitsE->At(j));
                    if (hit) {
                        filted_event->AddHitE(hit->GetId(), hit->GetTime(), hit->GetPe(), hit->GetNp());
                    }
                }
                
                // 拷贝缪子探测器击中
                TClonesArray* hitsM = exp_lhevent->GetHitsM();
                for (int j = 0; j < hitsM->GetEntries(); ++j) {
                    LHHit* hit = dynamic_cast<LHHit*>(hitsM->At(j));
                    if (hit) {
                        filted_event->AddHitM(hit->GetId(), hit->GetTime(), hit->GetPe(), hit->GetNp());
                    }
                }
                
                filted_tree->Fill();
                filted_event->Clear();

            }
            std::cout << "Experimental data filtering complete. " 
                    << filted_tree->GetEntries() << " events saved to " 
                    << filtedrootfile << std::endl;
            
            // 写入并关闭文件
            filtedfile.cd();
            filted_tree->Write();
            filtedfile.Close();
            expfile.Close();
            
            return 0;
        } 
        else {
            std::cerr << "Error: Invalid data source '" << datasource 
                    << "'. Must be 'simulation' or 'experiment'.\n";
            return -1;
        }
    return 0;
}

int main(int argc, char** argv){
    if (argc < 8) {
        std::cerr << "Usage: ./Data4Classify $DataSource $Mode $Km2aRootFile $RecRootFile $FiltedRootFile(output) $log10E_low $log10E_high\n" ;
        return 1;
    }

    std::string datasource=argv[1];
    int mode = std::stoi(argv[2]);
    std::string km2arootfile  =argv[3];
    std::string recrootfile   =argv[4];
    std::string filtedrootfile=argv[5];
    double recE_limit_low  = std::stod(argv[6]);
    double recE_limit_high = std::stod(argv[7]);


    ProcessEvents(datasource,km2arootfile,recrootfile,filtedrootfile,recE_limit_low,recE_limit_high);

    return 0;
    
}


/*

    // 统一处理函数
    int ProcessEvents(const std::string& input_file1, 
                    const std::string& input_file2,
                    const std::string& output_file,
                    double recE_limit_low,
                    double recE_limit_high,
                    bool is_simulation) 
    {
        // 打开输入文件
        TFile file1(input_file1.c_str(), "READ");
        TFile file2;
        if (is_simulation) {
            file2.Open(input_file2.c_str(), "READ");
        }

        // 获取输入树
        TTree* event_tree = nullptr;
        TTree* rec_tree = nullptr;
        
        if (is_simulation) {
            file1.GetObject("event", event_tree);
            file2.GetObject("Rec", rec_tree);
        } else {
            file1.GetObject("lhtree", event_tree);
            file1.GetObject("rectree", rec_tree);
        }

        // 创建输出文件
        TFile outfile(output_file.c_str(), "RECREATE");
        TTree* out_tree = new TTree("filted_tree", "Filtered Events");
        
        // 使用栈对象避免内存问题
        LHFiltedEvent filted_event;
        out_tree->Branch("FiltedEvent", &filted_event);

        // 事件循环处理
        Long64_t nentries = event_tree->GetEntries();
        for (Long64_t i = 0; i < nentries; ++i) {
            event_tree->GetEntry(i);
            rec_tree->GetEntry(i);

            // 获取事件对象
            LHEvent* event = nullptr;
            LHRecEvent* rec_event = nullptr;
            if (is_simulation) {
                event_tree->SetBranchAddress("Event", &event);
                rec_tree->SetBranchAddress("Rec", &rec_event);
            } else {
                event_tree->SetBranchAddress("LHEvent", &event);
                rec_tree->SetBranchAddress("RecEvent", &rec_event);
            }

            // 筛选逻辑
            if (!PassFilters(event, rec_event, recE_limit_low, recE_limit_high, is_simulation)) {
                continue;
            }

            // 填充输出事件
            FillFiltedEvent(&filted_event, event, rec_event, is_simulation);
            out_tree->Fill();
            filted_event.Clear();
        }

        // 写入输出
        outfile.cd();
        out_tree->Write();
        outfile.Close();

        return 0;
    }

    // 筛选函数
    bool PassFilters(LHEvent* event, LHRecEvent* rec_event, 
                    double recE_limit_low, double recE_limit_high,
                    bool is_simulation)
    {
        // 公共筛选逻辑
        double rec_r = std::sqrt(std::pow(rec_event->GetRec_x(), 2) + 
                            std::pow(rec_event->GetRec_y(), 2));
        
        if (filter0(rec_r, rec_event->GetRec_theta(), 
                rec_event->GetNpE1(), rec_event->GetNpE2()) == -9999) {
            return false;
        }

        // 能量筛选
        double recE = cal_recE_NpE1(rec_event->GetNpE1(), rec_event->GetRec_theta());
        if (recE < recE_limit_low || recE > recE_limit_high) {
            return false;
        }

        return true;
    }

    // 填充输出事件
    void FillFiltedEvent(LHFiltedEvent* out, LHEvent* event, LHRecEvent* rec_event, bool is_simulation)
    {
        // 公共字段
        out->SetEvN(event->GetEvN());
        out->SetId(event->GetId());
        // ...其他公共字段...

        // 模拟特有字段
        if (is_simulation) {
            out->SetTrueE(event->GetE()/1000.0); // TeV
        } else {
            out->SetTrueE(-1.0); // 实验数据无真实能量
        }

        // 重建参数
        out->SetNpE1(rec_event->GetNpE1());
        out->SetRecTheta(rec_event->GetRec_theta());
        // ...其他重建参数...

        // 拷贝击中
        CopyHits(out, event);
    }

    // 拷贝击中数据
    void CopyHits(LHFiltedEvent* out, LHEvent* event)
    {
        // 电子探测器击中
        TClonesArray* hitsE = event->GetHitsE();
        for (int j = 0; j < hitsE->GetEntries(); ++j) {
            LHHit* hit = dynamic_cast<LHHit*>(hitsE->At(j));
            if (hit) out->AddHitE(hit->GetId(), hit->GetTime(), hit->GetPe(), hit->GetNp());
        }

        // 缪子探测器击中
        TClonesArray* hitsM = event->GetHitsM();
        for (int j = 0; j < hitsM->GetEntries(); ++j) {
            LHHit* hit = dynamic_cast<LHHit*>(hitsM->At(j));
            if (hit) out->AddHitM(hit->GetId(), hit->GetTime(), hit->GetPe(), hit->GetNp());
        }
    }

    // 主函数
    int main(int argc, char** argv) {
        if (argc < 8) {
            std::cerr << "Usage: ./Data4Classify $DataSource $Mode $Input1 $Input2 $Output $log10E_low $log10E_high\n";
            return 1;
        }

        std::string datasource = argv[1];
        bool is_simulation = (datasource == "simulation");

        if (!is_simulation && datasource != "experiment") {
            std::cerr << "Error: Invalid data source\n";
            return -1;
        }

        return ProcessEvents(
            argv[3],                        // input_file1
            is_simulation ? argv[4] : "",   // input_file2 (实验数据不使用)
            argv[5],                        // output_file
            std::stod(argv[6]),             // recE_limit_low
            std::stod(argv[7]),             // recE_limit_high
            is_simulation                   // 是否模拟数据
        );
    }
*/