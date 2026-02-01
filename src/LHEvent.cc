#include "LHEvent.h"

ClassImp(LHEvent)
ClassImp(LHHit)
ClassImp(LHFiltedEvent)
ClassImp(LHWave)
ClassImp(LHRecEvent)

LHEvent::LHEvent(){
  NhitE  =  0  ;
  NhitM  =  0  ;
  NhitW  =  0  ;
  NwaveE =  0  ;
  NwaveM =  0  ;
  HitsE  =  new TClonesArray("LHHit", 100) ;
  HitsM  =  new TClonesArray("LHHit", 100) ;
  HitsW  =  new TClonesArray("LHHit", 100) ;
  WaveE  =  new TClonesArray("LHWave", 100) ;
  WaveM  =  new TClonesArray("LHWave", 100) ;
}

LHEvent::~LHEvent(){
  HitsE->Clear() ; 
  delete  HitsE  ;
  HitsM->Clear() ;
  delete  HitsM  ;
  HitsW->Clear() ;
  delete  HitsW  ;
  WaveE->Clear() ;
  delete  WaveE  ;
  WaveM->Clear() ;
  delete  WaveM  ;
}

void LHEvent::AddHitE(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np){
  new((*HitsE)[NhitE++]) LHHit(p_id,p_time,p_pe,p_np) ;
}
void LHEvent::AddHitM(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np){
  new((*HitsM)[NhitM++]) LHHit(p_id,p_time,p_pe,p_np) ;
}
void LHEvent::AddHitW(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np){
  new((*HitsW)[NhitW++]) LHHit(p_id,p_time,p_pe,p_np) ;
}
void LHEvent::AddWaveE(Double_t dt, Double_t da){
  new((*WaveE)[NwaveE++]) LHWave(dt,da) ;
}
void LHEvent::AddWaveM(Double_t dt, Double_t da){
  new((*WaveM)[NwaveM++]) LHWave(dt,da) ;
}
void LHEvent::AddHitE(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np, Int_t p_e){
  new((*HitsE)[NhitE++]) LHHit(p_id,p_time,p_pe,p_np,p_e) ;
}
void LHEvent::AddHitM(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np,Int_t p_e){
  new((*HitsM)[NhitM++]) LHHit(p_id,p_time,p_pe,p_np,p_e) ;
}
void LHEvent::AddHitW(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np,Int_t p_e){
  new((*HitsW)[NhitW++]) LHHit(p_id,p_time,p_pe,p_np,p_e) ;
}



LHFiltedEvent::LHFiltedEvent(){
  HitsE = new TClonesArray("LHHit", 100);
  HitsM = new TClonesArray("LHHit", 100);
}

LHFiltedEvent::~LHFiltedEvent(){
  HitsE->Clear(); delete HitsE;
  HitsM->Clear(); delete HitsM;
}

void LHFiltedEvent::AddHitE(Int_t p_id, Double_t p_time, Double_t p_pe, Int_t p_np)
{
  new((*HitsE)[HitsE->GetEntriesFast()]) LHHit(p_id, p_time, p_pe, p_np);
}

void LHFiltedEvent::AddHitM(Int_t p_id, Double_t p_time, Double_t p_pe, Int_t p_np)
{
  new((*HitsM)[HitsM->GetEntriesFast()]) LHHit(p_id, p_time, p_pe, p_np);
}

void LHFiltedEvent::AddHitE(Int_t p_id, Double_t p_time, Double_t p_pe, Int_t p_np, Int_t p_e)
{
  new((*HitsE)[HitsE->GetEntriesFast()]) LHHit(p_id, p_time, p_pe, p_np, p_e);
}

void LHFiltedEvent::AddHitM(Int_t p_id, Double_t p_time, Double_t p_pe, Int_t p_np, Int_t p_e)
{
  new((*HitsM)[HitsM->GetEntriesFast()]) LHHit(p_id, p_time, p_pe, p_np, p_e);
}




LHRecEvent::LHRecEvent()
{
  NhitE  =  0  ;
  NhitM  =  0  ;
  NhitW  =  0  ;
}

LHRecEvent::~LHRecEvent()  {
  NhitE  =  0  ;
  NhitM  =  0  ;
  NhitW  =  0  ;
}
