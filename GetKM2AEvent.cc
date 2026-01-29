/*

g++ -std=c++17 -O2 -Wall \
    -Iinclude -Isrc \
    src/LHEvent.cc src/LHEventDict.cc \
    src/KM2AEvent.cc src/KM2AEventDict.cc \
    GetKM2AEvent.cc \
    -o ./bin/GetKM2AEvent \
    `root-config --cflags --libs`

*/



// ./bin/GetKM2AEvent  /Users/macbook/PycharmProjects/pythonProject/searchMonopole/paper/LHAASO_Monopole/Filt_Event/Dataset_Filted/Experiment/ES.185136.KM2A_EVENT.PHYSICS_EDMD_OVERLAP.es-11.20250719115809.3001.dat.root
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
#include "KM2AEvent.h"




int main(int argc, char** argv){

    if(argc != 2){
        std::cout << "Usage: " << argv[0] << " <output file name>" << std::endl;
        return 1;
    }

    std::string inputfile = argv[1];
    TFile *file = new TFile(inputfile.c_str(),"READ");
    TTree *tree = (TTree*)file->Get("event"); 
    KM2AEvent *km2aevent = new KM2AEvent();
    tree->SetBranchAddress("Event",&km2aevent);
    if (!tree) {
    std::cerr << "Error: TTree 'Event' not found in file!" << std::endl;
    return 1;
    }

    tree->GetEntry(0);
    TClonesArray *km2ahits = km2aevent->GetHits();
    
    /*
    // Get Methods
    UInt_t ID()     {return id;};
    UInt_t Time()   {return ns;};
    double Charge();
    float  X()   const  { return  x       ; }  ; 
    float  Y()   const  { return  y       ; }  ; 
    float  Z()   const  { return  z       ; }  ; 
    float  T()   const  { return  t       ; }  ; 
    int    OE()   const  { return  oe      ; }  ;
    UChar_t Mode() const  { return mode     ; }  ;
    UInt_t  Id()  const  { return  id      ; }  ; 
    UInt_t Peda()  const  { return  peda    ; }  ;
    UInt_t Pedd()  const  { return  pedd    ; }  ;
    UChar_t PeakTimeAnode()  const  { return  peakTimeAnode   ; }  ;  
    UChar_t PeakTimeDynode()  const  { return  peakTimeDynode   ; }  ; 
    UInt_t Qa()  const  { return  qa    ; }  ;
    UInt_t Qd()  const  { return  qd    ; }  ;
    UChar_t Tag() const  { return  tag    ; }  ;
    */
   
    for (int j = 0; j < km2ahits->GetEntries(); ++j) {
        KM2AHit* hit = (KM2AHit*)km2ahits->At(j);
        printf("id:%d, x:%f, y:%f, z:%f, t:%f,  mode:%d, qa:%d, qd:%d, tag:%d\n", hit->Id(), hit->X(), hit->Y(), hit->Z(), hit->T(), hit->Mode(), hit->Qa(), hit->Qd(), hit->Tag());
    }

    file->Close();
}