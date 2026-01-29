// Do NOT change. Changes will be lost next time file is generated

#define R__DICTIONARY_FILENAME dIUsersdImacbookdIPycharmProjectsdIpythonProjectdIsearchMonopoledIpaperdILHAASO_MonopoledIFilt_EventdIsrcdIKM2AEventDict
#define R__NO_DEPRECATION

/*******************************************************************/
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define G__DICTIONARY
#include "ROOT/RConfig.hxx"
#include "TClass.h"
#include "TDictAttributeMap.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TBuffer.h"
#include "TMemberInspector.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"
#include "TError.h"

#ifndef G__ROOT
#define G__ROOT
#endif

#include "RtypesImp.h"
#include "TIsAProxy.h"
#include "TFileMergeInfo.h"
#include <algorithm>
#include "TCollectionProxyInfo.h"
/*******************************************************************/

#include "TDataMember.h"

// Header files passed as explicit arguments
#include "/Users/macbook/PycharmProjects/pythonProject/searchMonopole/paper/LHAASO_Monopole/Filt_Event/include/KM2AEvent.h"

// Header files passed via #pragma extra_include

// The generated code does not explicitly qualify STL entities
namespace std {} using namespace std;

namespace ROOT {
   static void *new_KM2AEvent(void *p = nullptr);
   static void *newArray_KM2AEvent(Long_t size, void *p);
   static void delete_KM2AEvent(void *p);
   static void deleteArray_KM2AEvent(void *p);
   static void destruct_KM2AEvent(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::KM2AEvent*)
   {
      ::KM2AEvent *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::KM2AEvent >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("KM2AEvent", ::KM2AEvent::Class_Version(), "KM2AEvent.h", 24,
                  typeid(::KM2AEvent), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::KM2AEvent::Dictionary, isa_proxy, 4,
                  sizeof(::KM2AEvent) );
      instance.SetNew(&new_KM2AEvent);
      instance.SetNewArray(&newArray_KM2AEvent);
      instance.SetDelete(&delete_KM2AEvent);
      instance.SetDeleteArray(&deleteArray_KM2AEvent);
      instance.SetDestructor(&destruct_KM2AEvent);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::KM2AEvent*)
   {
      return GenerateInitInstanceLocal(static_cast<::KM2AEvent*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ::KM2AEvent*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_KM2AHit(void *p = nullptr);
   static void *newArray_KM2AHit(Long_t size, void *p);
   static void delete_KM2AHit(void *p);
   static void deleteArray_KM2AHit(void *p);
   static void destruct_KM2AHit(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::KM2AHit*)
   {
      ::KM2AHit *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::KM2AHit >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("KM2AHit", ::KM2AHit::Class_Version(), "KM2AEvent.h", 102,
                  typeid(::KM2AHit), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::KM2AHit::Dictionary, isa_proxy, 4,
                  sizeof(::KM2AHit) );
      instance.SetNew(&new_KM2AHit);
      instance.SetNewArray(&newArray_KM2AHit);
      instance.SetDelete(&delete_KM2AHit);
      instance.SetDeleteArray(&deleteArray_KM2AHit);
      instance.SetDestructor(&destruct_KM2AHit);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::KM2AHit*)
   {
      return GenerateInitInstanceLocal(static_cast<::KM2AHit*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ::KM2AHit*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

//______________________________________________________________________________
atomic_TClass_ptr KM2AEvent::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *KM2AEvent::Class_Name()
{
   return "KM2AEvent";
}

//______________________________________________________________________________
const char *KM2AEvent::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::KM2AEvent*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int KM2AEvent::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::KM2AEvent*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *KM2AEvent::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::KM2AEvent*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *KM2AEvent::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::KM2AEvent*)nullptr)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
atomic_TClass_ptr KM2AHit::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *KM2AHit::Class_Name()
{
   return "KM2AHit";
}

//______________________________________________________________________________
const char *KM2AHit::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::KM2AHit*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int KM2AHit::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::KM2AHit*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *KM2AHit::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::KM2AHit*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *KM2AHit::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::KM2AHit*)nullptr)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
void KM2AEvent::Streamer(TBuffer &R__b)
{
   // Stream an object of class KM2AEvent.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(KM2AEvent::Class(),this);
   } else {
      R__b.WriteClassBuffer(KM2AEvent::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_KM2AEvent(void *p) {
      return  p ? new(p) ::KM2AEvent : new ::KM2AEvent;
   }
   static void *newArray_KM2AEvent(Long_t nElements, void *p) {
      return p ? new(p) ::KM2AEvent[nElements] : new ::KM2AEvent[nElements];
   }
   // Wrapper around operator delete
   static void delete_KM2AEvent(void *p) {
      delete (static_cast<::KM2AEvent*>(p));
   }
   static void deleteArray_KM2AEvent(void *p) {
      delete [] (static_cast<::KM2AEvent*>(p));
   }
   static void destruct_KM2AEvent(void *p) {
      typedef ::KM2AEvent current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
} // end of namespace ROOT for class ::KM2AEvent

//______________________________________________________________________________
void KM2AHit::Streamer(TBuffer &R__b)
{
   // Stream an object of class KM2AHit.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(KM2AHit::Class(),this);
   } else {
      R__b.WriteClassBuffer(KM2AHit::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_KM2AHit(void *p) {
      return  p ? new(p) ::KM2AHit : new ::KM2AHit;
   }
   static void *newArray_KM2AHit(Long_t nElements, void *p) {
      return p ? new(p) ::KM2AHit[nElements] : new ::KM2AHit[nElements];
   }
   // Wrapper around operator delete
   static void delete_KM2AHit(void *p) {
      delete (static_cast<::KM2AHit*>(p));
   }
   static void deleteArray_KM2AHit(void *p) {
      delete [] (static_cast<::KM2AHit*>(p));
   }
   static void destruct_KM2AHit(void *p) {
      typedef ::KM2AHit current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
} // end of namespace ROOT for class ::KM2AHit

namespace {
  void TriggerDictionaryInitialization_KM2AEventDict_Impl() {
    static const char* headers[] = {
"include/KM2AEvent.h",
nullptr
    };
    static const char* includePaths[] = {
"/Users/macbook/PycharmProjects/pythonProject/searchMonopole/paper/LHAASO_Monopole/Filt_Event/include",
"/Users/macbook/anaconda3/include/",
"/Users/macbook/PycharmProjects/pythonProject/searchMonopole/paper/LHAASO_Monopole/Filt_Event/",
nullptr
    };
    static const char* fwdDeclCode = R"DICTFWDDCLS(
#line 1 "KM2AEventDict dictionary forward declarations' payload"
#pragma clang diagnostic ignored "-Wkeyword-compat"
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern int __Cling_AutoLoading_Map;
class __attribute__((annotate("$clingAutoload$include/KM2AEvent.h")))  KM2AEvent;
class __attribute__((annotate("$clingAutoload$include/KM2AEvent.h")))  KM2AHit;
)DICTFWDDCLS";
    static const char* payloadCode = R"DICTPAYLOAD(
#line 1 "KM2AEventDict dictionary payload"


#define _BACKWARD_BACKWARD_WARNING_H
// Inline headers
#include "include/KM2AEvent.h"

#undef  _BACKWARD_BACKWARD_WARNING_H
)DICTPAYLOAD";
    static const char* classesHeaders[] = {
"KM2AEvent", payloadCode, "@",
"KM2AHit", payloadCode, "@",
nullptr
};
    static bool isInitialized = false;
    if (!isInitialized) {
      TROOT::RegisterModule("KM2AEventDict",
        headers, includePaths, payloadCode, fwdDeclCode,
        TriggerDictionaryInitialization_KM2AEventDict_Impl, {}, classesHeaders, /*hasCxxModule*/false);
      isInitialized = true;
    }
  }
  static struct DictInit {
    DictInit() {
      TriggerDictionaryInitialization_KM2AEventDict_Impl();
    }
  } __TheDictionaryInitializer;
}
void TriggerDictionaryInitialization_KM2AEventDict() {
  TriggerDictionaryInitialization_KM2AEventDict_Impl();
}
