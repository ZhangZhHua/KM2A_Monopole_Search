//////////////////////////////////////////////////////////////////////////
// Event                                                                //
// Description of the event and hit parameters                        //
//////////////////////////////////////////////////////////////////////////
#ifndef __LHEVENT_HH__
#define __LHEVENT_HH__

#include "TObject.h"
#include "TClonesArray.h"
#define _IS_MC 1

class LHEvent : public TObject {

 private:
  Int_t ev_n; 
  Double_t mjd;
  Double_t dt;  //the difference of referee time for KM2A and WCDA (T_km2a-T_wcda)
  Int_t NhitE;  //Nhit of ED
  Int_t NhitM;  //Nhit of MD
  Int_t NhitW;  //Nhit of WCDA
  Int_t NwaveE;  //Nhit of EDWave
  Int_t NwaveM;  //Nhit of MDWave
  Int_t NtrigE;  //Ntrig of ED;
  Int_t NtrigW;  //Ntrig of WCDA;
  Int_t TrigEt;
  #ifdef _IS_MC
  Double_t E;
  Int_t    Id;   //the primary type of particle
  Int_t    NpE;  //number of e/g/u in MD
  Int_t    NpW;  //number of e/g/u in WCDA
  Int_t    NuM;  //number of muon in MD
  Int_t    NuW;  //number of muon in WCDA
  Int_t    NuW2; //number of muon in WCDA E>1GeV
  Double_t Theta;
  Double_t Phi;
  Double_t Corex; 
  Double_t Corey;
  #endif

  TClonesArray * HitsE;
  TClonesArray * HitsM;
  TClonesArray * HitsW;
  TClonesArray * WaveE;
  TClonesArray * WaveM;

 public:
  LHEvent();
  virtual ~LHEvent();

  // Set Methods
  void     SetEvN(Int_t s_ev_n)        { ev_n    = s_ev_n  ; } ; 
  void	   SetMjd(Double_t s_mjd)      { mjd     = s_mjd   ; } ;
  void     SetDt(Double_t s_dt)      { dt     = s_dt   ; } ;
  void	   SetNhitE(Int_t s_nhite)     { NhitE   = s_nhite ; } ;
  void     SetNtrigE(Int_t s_nhite)    { NtrigE  = s_nhite ; } ;
  void     SetTrigEt(Int_t s_nhite)    { TrigEt  = s_nhite ; } ;
  void     SetNhitM(Int_t s_nhitm)     { NhitM   = s_nhitm ; } ;
  void     SetNhitW(Int_t s_nhitw)     { NhitW   = s_nhitw ; } ;
  void     SetNtrigW(Int_t s_nhite)    { NtrigW  = s_nhite ; } ;
  #ifdef _IS_MC
  void     SetE(Double_t s_E)         { E     = s_E     ; } ;
  void     SetId(Int_t s_id)          { Id    = s_id    ; } ;
  void     SetNpE(Int_t s_num)        { NpE   = s_num   ; } ;
  void     SetNpW(Int_t s_nuw)        { NpW   = s_nuw   ; } ;
  void     SetNuM(Int_t s_num)        { NuM   = s_num   ; } ;
  void     SetNuW(Int_t s_nuw)        { NuW   = s_nuw   ; } ;
  void     SetNuW2(Int_t s_nuw2)      { NuW2  = s_nuw2  ; } ;
  void     SetTheta(Double_t s_Theta) { Theta = s_Theta ; } ;
  void     SetPhi(Double_t s_Phi)     { Phi   = s_Phi   ; } ;
  void     SetCorex(Double_t s_Corex) { Corex = s_Corex ; } ;
  void     SetCorey(Double_t s_Corey) { Corey = s_Corey ; } ;
  #endif 
 
  // Get Methods
  Int_t    GetEvN()     const { return ev_n  ; } ;
  Double_t GetMjd()     const { return mjd   ; } ; 
  Double_t GetDt()      const { return dt   ; } ;
  Int_t    GetNhitE()   const { return NhitE ; } ;
  Int_t    GetNhitM()   const { return NhitM ; } ;
  Int_t    GetNhitW()   const { return NhitW ; } ;
  Int_t    GetNwaveE()  const { return NwaveE ; } ;
  Int_t    GetNwaveM()  const { return NwaveM ; } ;
  Int_t    GetNtrigE()  const { return NtrigE ; } ;
  Int_t    GetNtrigW()  const { return NtrigW ; } ;
  Int_t    GetTrigEt()  const { return TrigEt ; } ;
  #ifdef _IS_MC
  Double_t GetE()       const { return E     ; } ;
  Int_t    GetId()      const { return Id    ; } ;  
  Int_t    GetNpE()     const { return NpE   ; } ;
  Int_t    GetNpW()     const { return NpW   ; } ;
  Int_t    GetNuM()     const { return NuM   ; } ; 
  Int_t    GetNuW()     const { return NuW   ; } ;
  Int_t    GetNuW2()    const { return NuW2  ; } ; 
  Double_t GetTheta()   const { return Theta ; } ;
  Double_t GetPhi()     const { return Phi   ; } ;
  Double_t GetCorex()   const { return Corex ; } ;
  Double_t GetCorey()   const { return Corey ; } ;
  #endif

  TClonesArray * GetHitsE() const  { return HitsE    ; }  ;
  TClonesArray * GetHitsM() const  { return HitsM    ; }  ;
  TClonesArray * GetHitsW() const  { return HitsW    ; }  ;
  TClonesArray * GetWaveE() const  { return WaveE    ; }  ;
  TClonesArray * GetWaveM() const  { return WaveM    ; }  ;

  // To add a hit ...
  void AddHitE(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np);
  void AddHitM(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np);
  void AddHitW(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np);
  void AddWaveE(Double_t dt, Double_t da);
  void AddWaveM(Double_t dt, Double_t da);

  void AddHitE(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np, Int_t p_e);
  void AddHitM(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np, Int_t p_e);
  void AddHitW(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np, Int_t p_e);

  void Initcsz()
  //void Clear()
  {
    HitsE->Clear() ;  NhitE = 0;
    HitsM->Clear() ;  NhitM = 0;
    HitsW->Clear() ;  NhitW = 0;
    WaveE->Clear() ;  NwaveE = 0;
    WaveM->Clear() ;  NwaveM = 0; 
  };

  ClassDef(LHEvent,1) 

};

class LHHit : public TObject {
  
 private:
   Int_t    id; 
   Int_t    status; //5 is ok,2 is noise outof circle, 1 is noise after planar fit, 0 is noise hit after time filter, -1 is bad detector 
   Double_t time;
   Double_t pe; // number of pe or TDC count
   Int_t    np; //number of primary secondary particles
  
 public:
  
   // constructor and distructor
   LHHit() {};
  
   LHHit(Int_t p_id, Double_t p_time, Double_t p_pe, Int_t p_np){
       id = p_id; time = p_time; pe = p_pe;  np=p_np; status=5;
   };

   LHHit(Int_t p_id, Double_t p_time, Double_t p_pe, Int_t p_np, Int_t p_e){
       id = p_id; time = p_time; pe = p_pe;  np=p_np; status=p_e;
   };

   virtual ~LHHit() {} ;

   // Set Methods
   void     SetId(Int_t p_id)         { id     = p_id   ; } ;
   void     SetStatus(Int_t p_s)      { status = p_s    ; } ;
   void     SetTime(Double_t p_time)  { time   = p_time ; } ;
   void     SetPe(Double_t p_pe)      { pe     = p_pe   ; } ;
   void     SetNp(Int_t p_np)         { np     = p_np   ; } ;

   // Get Methods
   Int_t    GetId()   const  { return  id    ; }  ;
   Int_t    GetStatus()const { return  status; }  ;
   Double_t GetTime() const  { return  time  ; }  ;
   Double_t GetPe()   const  { return  pe    ; }  ;
   Int_t    GetNp()   const  { return  np    ; }  ;

   ClassDef(LHHit,1)
 
};


class LHWave : public TObject {

 private:
   Double_t t; //time 
   Double_t a; //ampl 

 public:
   LHWave() {};

   LHWave(Double_t dt, Double_t da){
      t =dt; a = da;
   };

   virtual ~LHWave() {} ;
   // Set Methods
   void     SetT(Double_t dt)  { t   = dt   ; } ;
   void     SetA(Double_t da)  { a   = da ; } ;
   // Get Methods
   Int_t    GetT() const  { return  t    ; }  ;
   Double_t GetA() const  { return  a  ; }  ;
   ClassDef(LHWave,1)
};


class LHFiltedEvent : public TObject {

 private:
  Int_t ev_n; 
  Double_t mjd;
  Double_t dt;  //dr?

  #ifdef _IS_MC
  Double_t trueE;
  Int_t    Id;   //the primary type of particle

  Double_t    NpE1; 
  Double_t    NpE2; 
  Double_t    NpE3; 
  Double_t    NuW1; 
  Double_t    NuW2;
  Double_t    NuW3;
  Double_t    theta;
  Double_t    phi;
  Double_t    x; 
  Double_t    y;

  Float_t rec_Echi;
  Int_t rec_Endf;
  Float_t rec_Mchi;
  Int_t rec_Mndf;

  Float_t rec_Ex;
  Float_t rec_Ey;
  Float_t rec_Ez;
  

  Int_t NhitE;  //Nhit of ED
  Int_t NhitM;  //Nhit of MD
  Int_t NtrigE;  //Ntrig of ED
  Int_t NtrigE2;  //Ntrig of ED using guarding ring
  Int_t NfiltE;  //Nhit of ED after filter out noise
  Int_t NfiltM;  //Nhit of MD after filter out noise

  Double_t    rec_theta;
  Double_t    rec_phi;
  Double_t    rec_x; 
  Double_t    rec_y;
  Double_t    rec_r;
  Double_t    recE;
  Double_t    rec_Eage;
  Double_t    rec_Mage;
  Double_t    log10E_limit_low;
  Double_t    log10E_limit_high;


  #endif

  TClonesArray * HitsE;
  TClonesArray * HitsM;


 public:
  LHFiltedEvent();
  virtual ~LHFiltedEvent();

  // Set Methods
  void     SetEvN(Int_t s_ev_n)        { ev_n    = s_ev_n  ; } ; 
  void     SetMjd(Double_t s_mjd)       {mjd = s_mjd;};
  void     SetDt(Double_t s_dt)        {dt = s_dt;};
  #ifdef _IS_MC
  void     SetTrueE(Double_t s_trueE)  { trueE   = s_trueE ; } ;
  void     SetId(Int_t s_id)          { Id      = s_id    ; } ;
  void     SetNpE1(Double_t s_num)    { NpE1    = s_num   ; } ;
  void     SetNpE2(Double_t s_num)    { NpE2    = s_num   ; } ;
  void     SetNpE3(Double_t s_num)    { NpE3    = s_num   ; } ;
  void     SetNuW1(Double_t s_nuw)    { NuW1    = s_nuw   ; } ;
  void     SetNuW2(Double_t s_nuw)    { NuW2    = s_nuw   ; } ;
  void     SetNuW3(Double_t s_nuw)    { NuW3    = s_nuw   ; } ;
  void     SetTheta(Double_t s_Theta) { theta   = s_Theta ; } ;
  void     SetPhi(Double_t s_Phi)     { phi     = s_Phi   ; } ;
  void     SetX(Double_t s_x)         { x       = s_x     ; } ;
  void     SetY(Double_t s_y)         { y       = s_y     ; } ;

  void SetRecEchi(Float_t val) { rec_Echi = val; }
  void SetRecEndf(Int_t val) { rec_Endf = val; }
  void SetRecMchi(Float_t val) { rec_Mchi = val; }
  void SetRecMndf(Int_t val) { rec_Mndf = val; }

  void SetRecEx(Float_t val) { rec_Ex = val; }
  void SetRecEy(Float_t val) { rec_Ey = val; }
  void SetRecEz(Float_t val) { rec_Ez = val; }

  void SetNhitE(Int_t val) { NhitE = val; }
  void SetNhitM(Int_t val) { NhitM = val; }
  void SetNtrigE(Int_t val) { NtrigE = val; }
  void SetNtrigE2(Int_t val) { NtrigE2 = val; }
  void SetNfiltE(Int_t val) { NfiltE = val; }
  void SetNfiltM(Int_t val) { NfiltM = val; }

  void SetRecEage(Double_t val) { rec_Eage = val; }
  void SetRecMage(Double_t val) { rec_Mage = val; }

  void     SetRecTheta(Double_t s_Theta) { rec_theta   = s_Theta ; } ;
  void     SetRecPhi(Double_t s_Phi)     { rec_phi     = s_Phi   ; } ;
  void     SetRecX(Double_t s_recx)         { rec_x       = s_recx     ; } ;
  void     SetRecY(Double_t s_recy)         { rec_y       = s_recy     ; } ;
  void     SetRecE(Double_t s_recE)         { recE       = s_recE     ; } ;
  void     SetRecR(Double_t s_recr)         { rec_r       = s_recr     ; } ;
  void     SetLog10E_limit_low(Double_t s_log10E_limit_low)         { log10E_limit_low       = s_log10E_limit_low     ; } ;
  void     SetLog10E_limit_high(Double_t s_log10E_limit_high)         { log10E_limit_high       = s_log10E_limit_high     ; } ;
  

  #endif 
 
  // Get Methods
  Int_t    GetEvN()     const { return ev_n  ; } ;
 
  #ifdef _IS_MC
  Double_t GetTrueE()   const { return trueE ; } ;
  Int_t    GetId()      const { return Id    ; } ;  
  Double_t GetNpE1()    const { return NpE1   ; } ;
  Double_t GetNpE2()    const { return NpE2   ; } ;
  Double_t GetNpE3()    const { return NpE3   ; } ;
  Double_t GetNuW1()    const { return NuW1   ; } ;
  Double_t GetNuW2()    const { return NuW2   ; } ;
  Double_t GetNuW3()    const { return NuW3   ; } ;
  Double_t GetTheta()   const { return theta ; } ;
  Double_t GetPhi()     const { return phi   ; } ;
  Double_t GetX()       const { return x     ; } ;
  Double_t GetY()       const { return y     ; } ;

  Float_t GetRecEchi() const { return rec_Echi; }
  Int_t GetRecEndf() const { return rec_Endf; }
  Float_t GetRecMchi() const { return rec_Mchi; }
  Int_t GetRecMndf() const { return rec_Mndf; }

  Float_t GetRecEx() const { return rec_Ex; }
  Float_t GetRecEy() const { return rec_Ey; }
  Float_t GetRecEz() const { return rec_Ez; }

  Int_t GetNhitE() const { return NhitE; }
  Int_t GetNhitM() const { return NhitM; }
  Int_t GetNtrigE() const { return NtrigE; }
  Int_t GetNtrigE2() const { return NtrigE2; }
  Int_t GetNfiltE() const { return NfiltE; }
  Int_t GetNfiltM() const { return NfiltM; }

  Double_t GetRecEage() const { return rec_Eage; }
  Double_t GetRecMage() const { return rec_Mage; }

  Double_t GetRecTheta()   const { return rec_theta ; } ;
  Double_t GetRecPhi()     const { return rec_phi   ; } ;
  Double_t GetRecX()       const { return rec_x     ; } ;
  Double_t GetRecY()       const { return rec_y     ; } ;
  Double_t GetRecE()       const { return recE     ; } ;
  Double_t GetRecR()       const { return rec_r     ; } ;

  Double_t GetLog10E_limit_low()       const { return log10E_limit_low     ; } ;
  Double_t GetLog10E_limit_high()       const { return log10E_limit_high     ; } ;

    #endif


  TClonesArray * GetHitsE() const  { return HitsE    ; }  ;
  TClonesArray * GetHitsM() const  { return HitsM    ; }  ;


  // To add a hit ...
  void AddHitE(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np);
  void AddHitM(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np);


  void AddHitE(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np, Int_t p_e);
  void AddHitM(Int_t p_id, Double_t p_time, Double_t p_pe,Int_t p_np, Int_t p_e);

  void Clear()
  {
    HitsE->Clear() ; 
    HitsM->Clear() ; 
  };

  ClassDef(LHFiltedEvent,1) 

};


class LHRecEvent : public TObject {
 public:
  Int_t ev_n;
  Double_t mjd;
  Double_t dt;
  #ifdef _IS_MC
  Int_t   id;
  Float_t E;
  Float_t theta;
  Float_t phi;
  Float_t corex;
  Float_t corey;
  Float_t corez;
  Int_t pNpE;  //primary number of particles of ED
  Int_t pNpW;  //primary number of particles of WCDA
  Int_t pNuM;  //primary number of muons of MD
  Int_t pNuW;  //primary number of muons of WCDA
  Int_t pNuW2; //primary number of muons of WCDA above 1GeV
  #endif
  Float_t rec_x;
  Float_t rec_y;
  Float_t rec_z;
  Float_t rec_theta;  
  Float_t rec_phi;
  Float_t rec_a;
  Float_t rec_c0;
  Float_t rec_sigma; 

  Float_t rec_Etheta_p; //planar fit 
  Float_t rec_Ephi_p;
  Float_t rec_Ec0_p;
  Float_t rec_Esigma_p;
  Float_t rec_Wtheta_p;
  Float_t rec_Wphi_p;
  Float_t rec_Wc0_p;
  Float_t rec_Wsigma_p;
  Float_t rec_Etheta_c; //conical fit 
  Float_t rec_Ephi_c;
  Float_t rec_Ec0_c;
  Float_t rec_Esigma_c;
  Float_t rec_Ea;
  Float_t rec_Wtheta_c;
  Float_t rec_Wphi_c;
  Float_t rec_Wc0_c;
  Float_t rec_Wsigma_c;
  Float_t rec_Wa;

  Float_t rec_Esize;
  Float_t rec_Wsize;
  Float_t rec_Msize;
  Float_t rec_Msize2;
  Float_t rec_Eage;
  Float_t rec_Wage;
  Float_t rec_Mage;
  Float_t rec_Mage2;
  Float_t rec_Mrm;
  Float_t rec_Echi;
  Int_t rec_Endf;
  Float_t rec_Mchi;
  Float_t rec_Mchi2;
  Int_t rec_Mndf;

  Float_t rec_Ex;
  Float_t rec_Ey;
  Float_t rec_Ez;
  Float_t rec_Wx;
  Float_t rec_Wy;
  Float_t rec_Wz;

  Int_t NhitE;  //Nhit of ED
  Int_t NhitM;  //Nhit of MD
  Int_t NhitW;  //Nhit of WCDA

  Int_t NtrigE;  //Ntrig of ED
  Int_t NtrigE2;  //Ntrig of ED using guarding ring
  Int_t NtrigW;  //Ntrig of WCDA
  Int_t NfiltE;  //Nhit of ED after filter out noise
  Int_t NfiltM;  //Nhit of MD after filter out noise
  Int_t NfiltW;  //Nhit of WCDA after filter out noise
  Int_t NliveM1;
  Int_t NliveM2;
  Int_t NliveM3;

  Float_t Redge;
  Float_t NpE1;  //number of particles of ED  with r<rwind+50, dt:-50,100
  Float_t NpE2;  //number of particles of ED  with r=40-100, dt:-50,100
  Float_t NpE3;  
  Float_t NpW;  //number of particles of WCDA
  Float_t NuM1;  //number of muons of MD with r=15,rwind+100, dt:-50,100
  Float_t NuM2;  //number of muons of MD with r=15,rwind+300, dt:-50,200
  Float_t NuM3;  //number of muons of MD with r=40,200, dt:-50,100 
  Float_t NuM4;
  Float_t NuM5;
  Float_t NuW1;  //number of muons of WCDA
  Float_t NuW2;  //number of muons of WCDA
  Float_t NuW3;  //number of muons of WCDA 
  LHRecEvent();
  virtual ~LHRecEvent();


  Int_t    GetEvN()     const { return ev_n  ; } ;
  Double_t GetMjd()    const { return mjd   ; } ;
  Double_t GetDt()     const { return dt    ; } ;
  Double_t GetE()       const { return E     ; } ;
  Int_t    GetId()      const { return id    ; } ;  

  
  Double_t GetTheta()   const { return theta ; } ;
  Double_t GetPhi()     const { return phi   ; } ;
  Double_t GetCorex()   const { return corex ; } ;
  Double_t GetCorey()   const { return corey ; } ;
  
  Double_t GetNpE1()    const { return NpE1 ; } ;
  Double_t GetNpE2()    const { return NpE2 ; };
  Double_t GetNpE3()    const { return NpE3 ; };
  Double_t GetNuW1()    const { return NuW1  ; } ; 
  Double_t GetNuW2()    const { return NuW2  ; } ;  //log(recE)
  Double_t GetNuW3()    const { return NuW3  ; } ;  //log(recE)
  Double_t GetRec_theta() const { return rec_theta ; } ;
  Double_t GetRec_phi() const { return rec_phi ; } ;
  Double_t GetRec_x()  const { return rec_x ; } ;
  Double_t GetRec_y()  const { return rec_y ; } ;

  Float_t GetRecEchi() const { return rec_Echi; }
  Int_t GetRecEndf() const { return rec_Endf; }
  Float_t GetRecMchi() const { return rec_Mchi; }
  Int_t GetRecMndf() const { return rec_Mndf; }

  Float_t GetRecEx() const { return rec_Ex; }
  Float_t GetRecEy() const { return rec_Ey; }
  Float_t GetRecEz() const { return rec_Ez; }

  Int_t GetNhitE() const { return NhitE; }
  Int_t GetNhitM() const { return NhitM; }
  Int_t GetNtrigE() const { return NtrigE; }
  Int_t GetNtrigE2() const { return NtrigE2; }
  Int_t GetNfiltE() const { return NfiltE; }
  Int_t GetNfiltM() const { return NfiltM; }

  Double_t GetRecEage() const { return rec_Eage; }
  Double_t GetRecMage() const { return rec_Mage; }
  

  ClassDef(LHRecEvent,1)
  
};




#endif


