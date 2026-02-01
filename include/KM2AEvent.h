/*******************************************************************************
 * File: KM2AEvent.h
 * Description: header file for the KM2A event and hit.
 * Comments:
 * Author: HHh
 * Created: 2010/04/01
 * Modified: 2014/09/05 for the new array@YBJ
 *******************************************************************************/

#ifndef _KM2A_Event
#define _KM2A_Event 1

#include "TObject.h"
#include "TClonesArray.h"

// 1970-01-01 MJD = 40587
const double MJD19700101 = 40587;

const int TIME_WINDOW  = 5000; // in nanoseconds, half of the DAQ time window
const double TDC_UNIT  = 1;    // in nanoseconds
const UInt_t TDC_START = 4500;
const UInt_t TDC_END   = 5500;

class KM2AEvent : public TObject {

  private:
    Int_t ev_n; 
    Double_t mjd, dt, fs;
    Int_t nhit, nmd;
    float phi[3], theta[3]; // 0: even, 1: odd, 2: the whole array
    // the follows are reconstructed by using the whole array
    Int_t nfit;
    float core_x, core_y;
    double c0, a, b, r, x2;

    TClonesArray * hits;

  public:
    // constructor and distructor
    KM2AEvent();
    virtual ~KM2AEvent();

    // Set Methods
    void SetEvN(Int_t s_ev_n)   { ev_n = s_ev_n ; } ; 
    void SetMjd(Double_t s_mjd) { mjd  = s_mjd	; } ;
    void SetFs(Double_t s_fs)   { fs   = s_fs	; } ;
    void SetDt(Double_t s_dt)   { dt   = s_dt	; } ;
    void SetNhit(Int_t s_nhit)  { nhit = s_nhit ; } ;

    // Get Methods
    Int_t    EvN()     const { return ev_n ; } ; 
    Int_t    NHit()    const { return nhit ; } ;
    Int_t    Nmd()     const { return nmd  ; } ;
    Double_t Mjd()     const { return mjd	; } ;
    Double_t Fs()      const { return fs	; } ;
    Double_t Dt()      const { return dt	; } ;
    // TODO
    float    Phi()     const { return phi[2]   ; } ;
    float    Phiz()     const { return phi[0]   ; } ;
    float    Phio()     const { return phi[1]   ; } ;
    float    Theta()   const { return theta[2] ; } ;
    float    Thetaz()   const { return theta[0] ; } ;
    float    Thetao()   const { return theta[1] ; } ;
    float    Corex()      const { return core_x; } ;
    float    Corey()      const { return core_y; } ;
    float    C0()      const { return c0; } ;
    double   A()       const { return a; } ;
    double   B()       const { return b; } ;
    TClonesArray * GetHits() const { return hits; } ;

    // To add a hit ...
    void AddHit(
      UInt_t  p_id,
      double  p_x, double p_y, double p_z, double p_t0, int p_oe,
      UChar_t p_mode,
      UInt_t  p_time_ns,
      UInt_t  p_peakAnode,
      UInt_t  p_peakDynode,
      UChar_t p_peakTimeAnode,
      UChar_t p_peakTimeDynode,
      UInt_t  p_pedAnode,
      UInt_t  p_pedDynode,
      UChar_t p_tag);

    void Init()
    {
      hits->Clear() ;  nhit = 0; nmd = 0;
      for(int i=0; i<3; i++)
      {
        theta[i] = -1;
        phi[i] = -1;
      }
    };

    void Fit(int nMode);

    ClassDef(KM2AEvent,1) 
 };

//

class KM2AHit : public TObject {

  private:
    UInt_t  id;
    UChar_t mode;
    UInt_t  ns;
    UInt_t  qa;
    UInt_t  qd;
    UChar_t peakTimeAnode;
    UChar_t peakTimeDynode;
    UInt_t  peda;
    UInt_t  pedd;
    UChar_t tag; 

    /* odd/even or others
     *   0x*0: even
     *   0x*1: odd
     *   0x*2: always used in reconstruction
     *   0x*3: never used in reconstruction
     *   0x0*: ED without MD below
     *   0x1*: ED on top of MD#1 for punch-through calculation
     *   0x2*: ED on top of MD#2 for punch-through calculation
     */
    int     oe;

    // hit coordinates and time after calibration
    float   x; 
    float   y; 
    float   z;
    float  t0;
    float   t;

  public:
    // constructor and distructor
    KM2AHit() {};
    KM2AHit(
      UInt_t  p_id,
      double  p_x, double p_y, double p_z, double p_t0, int p_oe,
      UChar_t p_mode,
      UInt_t  p_time_ns,
      UInt_t  p_peakAnode,
      UInt_t  p_peakDynode,
      UChar_t p_peakTimeAnode,
      UChar_t p_peakTimeDynode,
      UInt_t  p_pedAnode,
      UInt_t  p_pedDynode,
      UChar_t p_tag)
      {
        x = p_x, y = p_y, z = p_z, t0 = p_t0, oe = p_oe;
        id = p_id, mode = p_mode, ns = p_time_ns, qd = p_peakDynode;
        peakTimeDynode = p_peakTimeDynode, qa = p_peakAnode, peakTimeAnode = p_peakTimeAnode;
        peda = p_pedAnode, pedd = p_pedDynode, tag = p_tag;
        t = ns - t0;
      };

    virtual ~KM2AHit() {} ;

    // Set Methods
 
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
  ClassDef(KM2AHit,1) 
};

#endif
