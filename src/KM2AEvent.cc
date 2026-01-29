/*******************************************************************************
 * File: KM2AEvent.cc
 * Description: C++ source file of the KM2A classes.
 * Comments:
 * Author: HHh
 * Created: 2010/04/01
 * Modified: 2014/09/05 for the new array@YBJ
 *******************************************************************************/

#include <math.h>
#include <TMath.h>
#include <TMatrixD.h>
#include <TVectorD.h>
#include <TDecompSVD.h>
#include "KM2AEvent.h"

ClassImp(KM2AEvent)
ClassImp(KM2AHit)

KM2AEvent::KM2AEvent()
{
  nhit  = 0;
  nmd   = 0;
  hits  =  new TClonesArray("KM2AHit", 100) ;
  for(int i=0; i<3; i++)
  {
    theta[i] = -1;
    phi[i] = -1;
  } 
}

KM2AEvent::~KM2AEvent()
{
  hits->Clear() ; 
  delete  hits  ;
}

void KM2AEvent::AddHit(
       UInt_t  p_id,
       double  p_x, double p_y, double p_z, double p_t0, int p_oe,
       UChar_t p_mode,
       UInt_t p_time_ns,
       UInt_t p_peakAnode,
       UInt_t p_peakDynode,
       UChar_t p_peakTimeAnode,
       UChar_t p_peakTimeDynode,
       UInt_t p_pedAnode,
       UInt_t p_pedDynode,
       UChar_t p_tag)
{
  new((*hits)[nhit+nmd]) KM2AHit(
    p_id,
    p_x, p_y, p_z, p_t0, p_oe,
    p_mode,
    p_time_ns,
    p_peakAnode,
    p_peakDynode,
    p_peakTimeAnode,
    p_peakTimeDynode,
    p_pedAnode,
    p_pedDynode,
    p_tag);
    if(p_mode==0)  nhit++; // ED hits
    else nmd++;          // MD hits
}

void KM2AEvent::Fit(int nMode)
{
  int i, j, nPoints;
  KM2AHit *hit;

  const int ITER = 100;
  const Int_t nrVar = 3;
  const double C = 0.2998;
  //const double EPS = 1e-6;
  const double EPS = 3; // ns, residual less than time jitter is meaningless, thus weight cannot be too large

  Int_t N = nhit;
  Double_t ax[N], ay[N], az[N], at0[N], at[N];
  double x20, nn;

  a = -1; b=-1; r=-1; x2=-1;
  theta[nMode] = -1;
  phi[nMode] = -1;
  c0 = -1;
  core_x = 0;
  core_y = 0;
  if(nhit<5) return;

  nPoints = 0;
  for(i=0; i<nhit+nmd; i++)
  {
    hit = (KM2AHit *)((*hits)[i]);

    if(hit->Mode()==1) continue; // MD hit, ignore
    if((hit->OE()&0x0F)==3) continue;   // don't use this detector in reconstruction

    if((nMode<2) && ((hit->OE()&0x0F)!=nMode)) continue;
    if((hit->Charge()>3000)||(hit->Charge()<5)) continue; // TODO
    if(hit->Time()<TDC_START||hit->Time()>TDC_END) continue;

    core_x += hit->X();
    core_y += hit->Y();
    // TODO core_z?? 
    ax[nPoints]  = hit->X()/C;
    ay[nPoints]  = hit->Y()/C;
    az[nPoints]  = hit->Z()/C;
    //at0[nPoints] = hit->Time()*TDC_UNIT;
    at0[nPoints] = hit->T();
    at[nPoints]  = at0[nPoints] - az[nPoints]; // cos(theta)=1
    nPoints++;
  }
  nfit = -nPoints;
  if(nPoints<5) return;
  core_x /= nPoints;
  core_y /= nPoints;

  // Make the vectors 'Use" the data : they are not copied, the vector data
  // pointer is just set appropriately

  TVectorD x; x.Use(nPoints, ax);
  TVectorD y; y.Use(nPoints, ay);
  TVectorD t; t.Use(nPoints, at);

  TMatrixD A(nPoints,nrVar);
  TMatrixDColumn(A,0) = 1.0;
  TMatrixDColumn(A,1) = x;
  TMatrixDColumn(A,2) = y;

  TMatrixD Aw = A;
  TVectorD yw = t;

  TDecompSVD svd(Aw);
  Bool_t ok;
  TVectorD c_svd = svd.Solve(yw,ok);

  nfit = 0;
  for(j=0; j<ITER; j++)
  {
    a  = c_svd(1);
    b  = c_svd(2);
    c0 = c_svd(0);
    nn = a*a+b*b;
    if(nn>1) return;

    nn = sqrt(1-nn);
    for(i=0; i<nPoints; i++) at[i] = at0[i] - az[i]*nn;

    Aw = A;
    yw = t;
    x20 = x2, x2 = 0, nfit = 0;
    for(i=0; i<nPoints; i++)
    {
      r = fabs(at[i]-(ax[i]*a+ay[i]*b+c0));
      if(r<10) nfit++; // TODO: why 10ns?
      x2 += r*r;

      if(r<EPS) r=EPS;
      r = 1./r;

      TMatrixDRow(Aw,i) *= r;
      yw(i) *= r;
    }
    if(fabs(x2-x20)<EPS) break;
    svd.SetMatrix(Aw);
    c_svd = svd.Solve(yw,ok);
  }

  r = sqrt(a*a+b*b);
  if(r>1.) return;

  x2 /= (nPoints-nrVar);
  theta[nMode] = asin(r);
  phi[nMode]   = asin(b/r);
  if(a<0.) phi[nMode] = TMath::Pi() - phi[nMode];
  else if(phi[nMode]<0.) phi[nMode] += TMath::TwoPi();
}

double KM2AHit::Charge()
{
  double q = 0;
  //if(id==100) // new MD
  //{
  //  q = (double)qa - 304.*peda/8.;
  //  if(q>2.75e5) q = ((double)qd-304.*pedd/8.)*145.;
  //}
  //else if(id==101) // old MD
  //{
  //  q = (double)qa - 304.*peda/8.;
  //  if(q>2.75e5) q = ((double)qd-304.*pedd/8.)*145.;
  //}
  //else if(id>150) // EDs
  {
    q = qa;
    if(q>3000) q = qd*150;
  }

  return q;
}
