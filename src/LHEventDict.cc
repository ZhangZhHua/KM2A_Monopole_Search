// Do NOT change. Changes will be lost next time file is generated

#define R__DICTIONARY_FILENAME dIUsersdImacbookdIPycharmProjectsdIpythonProjectdIsearchMonopoledIpaperdILHAASO_MonopoledIFilt_EventdIsrcdILHEventDict
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
#include "/Users/macbook/PycharmProjects/pythonProject/searchMonopole/paper/LHAASO_Monopole/Filt_Event/include/LHEvent.h"

// Header files passed via #pragma extra_include

// The generated code does not explicitly qualify STL entities
namespace std {} using namespace std;

namespace ROOT {
   static void *new_LHEvent(void *p = nullptr);
   static void *newArray_LHEvent(Long_t size, void *p);
   static void delete_LHEvent(void *p);
   static void deleteArray_LHEvent(void *p);
   static void destruct_LHEvent(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::LHEvent*)
   {
      ::LHEvent *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::LHEvent >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("LHEvent", ::LHEvent::Class_Version(), "LHEvent.h", 12,
                  typeid(::LHEvent), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::LHEvent::Dictionary, isa_proxy, 4,
                  sizeof(::LHEvent) );
      instance.SetNew(&new_LHEvent);
      instance.SetNewArray(&newArray_LHEvent);
      instance.SetDelete(&delete_LHEvent);
      instance.SetDeleteArray(&deleteArray_LHEvent);
      instance.SetDestructor(&destruct_LHEvent);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::LHEvent*)
   {
      return GenerateInitInstanceLocal(static_cast<::LHEvent*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ::LHEvent*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_LHHit(void *p = nullptr);
   static void *newArray_LHHit(Long_t size, void *p);
   static void delete_LHHit(void *p);
   static void deleteArray_LHHit(void *p);
   static void destruct_LHHit(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::LHHit*)
   {
      ::LHHit *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::LHHit >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("LHHit", ::LHHit::Class_Version(), "LHEvent.h", 131,
                  typeid(::LHHit), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::LHHit::Dictionary, isa_proxy, 4,
                  sizeof(::LHHit) );
      instance.SetNew(&new_LHHit);
      instance.SetNewArray(&newArray_LHHit);
      instance.SetDelete(&delete_LHHit);
      instance.SetDeleteArray(&deleteArray_LHHit);
      instance.SetDestructor(&destruct_LHHit);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::LHHit*)
   {
      return GenerateInitInstanceLocal(static_cast<::LHHit*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ::LHHit*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_LHWave(void *p = nullptr);
   static void *newArray_LHWave(Long_t size, void *p);
   static void delete_LHWave(void *p);
   static void deleteArray_LHWave(void *p);
   static void destruct_LHWave(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::LHWave*)
   {
      ::LHWave *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::LHWave >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("LHWave", ::LHWave::Class_Version(), "LHEvent.h", 174,
                  typeid(::LHWave), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::LHWave::Dictionary, isa_proxy, 4,
                  sizeof(::LHWave) );
      instance.SetNew(&new_LHWave);
      instance.SetNewArray(&newArray_LHWave);
      instance.SetDelete(&delete_LHWave);
      instance.SetDeleteArray(&deleteArray_LHWave);
      instance.SetDestructor(&destruct_LHWave);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::LHWave*)
   {
      return GenerateInitInstanceLocal(static_cast<::LHWave*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ::LHWave*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_LHFiltedEvent(void *p = nullptr);
   static void *newArray_LHFiltedEvent(Long_t size, void *p);
   static void delete_LHFiltedEvent(void *p);
   static void deleteArray_LHFiltedEvent(void *p);
   static void destruct_LHFiltedEvent(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::LHFiltedEvent*)
   {
      ::LHFiltedEvent *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::LHFiltedEvent >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("LHFiltedEvent", ::LHFiltedEvent::Class_Version(), "LHEvent.h", 198,
                  typeid(::LHFiltedEvent), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::LHFiltedEvent::Dictionary, isa_proxy, 4,
                  sizeof(::LHFiltedEvent) );
      instance.SetNew(&new_LHFiltedEvent);
      instance.SetNewArray(&newArray_LHFiltedEvent);
      instance.SetDelete(&delete_LHFiltedEvent);
      instance.SetDeleteArray(&deleteArray_LHFiltedEvent);
      instance.SetDestructor(&destruct_LHFiltedEvent);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::LHFiltedEvent*)
   {
      return GenerateInitInstanceLocal(static_cast<::LHFiltedEvent*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ::LHFiltedEvent*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

namespace ROOT {
   static void *new_LHRecEvent(void *p = nullptr);
   static void *newArray_LHRecEvent(Long_t size, void *p);
   static void delete_LHRecEvent(void *p);
   static void deleteArray_LHRecEvent(void *p);
   static void destruct_LHRecEvent(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::LHRecEvent*)
   {
      ::LHRecEvent *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::LHRecEvent >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("LHRecEvent", ::LHRecEvent::Class_Version(), "LHEvent.h", 380,
                  typeid(::LHRecEvent), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::LHRecEvent::Dictionary, isa_proxy, 4,
                  sizeof(::LHRecEvent) );
      instance.SetNew(&new_LHRecEvent);
      instance.SetNewArray(&newArray_LHRecEvent);
      instance.SetDelete(&delete_LHRecEvent);
      instance.SetDeleteArray(&deleteArray_LHRecEvent);
      instance.SetDestructor(&destruct_LHRecEvent);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::LHRecEvent*)
   {
      return GenerateInitInstanceLocal(static_cast<::LHRecEvent*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ::LHRecEvent*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

//______________________________________________________________________________
atomic_TClass_ptr LHEvent::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *LHEvent::Class_Name()
{
   return "LHEvent";
}

//______________________________________________________________________________
const char *LHEvent::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::LHEvent*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int LHEvent::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::LHEvent*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *LHEvent::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::LHEvent*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *LHEvent::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::LHEvent*)nullptr)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
atomic_TClass_ptr LHHit::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *LHHit::Class_Name()
{
   return "LHHit";
}

//______________________________________________________________________________
const char *LHHit::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::LHHit*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int LHHit::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::LHHit*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *LHHit::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::LHHit*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *LHHit::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::LHHit*)nullptr)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
atomic_TClass_ptr LHWave::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *LHWave::Class_Name()
{
   return "LHWave";
}

//______________________________________________________________________________
const char *LHWave::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::LHWave*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int LHWave::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::LHWave*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *LHWave::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::LHWave*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *LHWave::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::LHWave*)nullptr)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
atomic_TClass_ptr LHFiltedEvent::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *LHFiltedEvent::Class_Name()
{
   return "LHFiltedEvent";
}

//______________________________________________________________________________
const char *LHFiltedEvent::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::LHFiltedEvent*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int LHFiltedEvent::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::LHFiltedEvent*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *LHFiltedEvent::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::LHFiltedEvent*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *LHFiltedEvent::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::LHFiltedEvent*)nullptr)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
atomic_TClass_ptr LHRecEvent::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *LHRecEvent::Class_Name()
{
   return "LHRecEvent";
}

//______________________________________________________________________________
const char *LHRecEvent::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::LHRecEvent*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int LHRecEvent::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::LHRecEvent*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *LHRecEvent::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::LHRecEvent*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *LHRecEvent::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::LHRecEvent*)nullptr)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
void LHEvent::Streamer(TBuffer &R__b)
{
   // Stream an object of class LHEvent.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(LHEvent::Class(),this);
   } else {
      R__b.WriteClassBuffer(LHEvent::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_LHEvent(void *p) {
      return  p ? new(p) ::LHEvent : new ::LHEvent;
   }
   static void *newArray_LHEvent(Long_t nElements, void *p) {
      return p ? new(p) ::LHEvent[nElements] : new ::LHEvent[nElements];
   }
   // Wrapper around operator delete
   static void delete_LHEvent(void *p) {
      delete (static_cast<::LHEvent*>(p));
   }
   static void deleteArray_LHEvent(void *p) {
      delete [] (static_cast<::LHEvent*>(p));
   }
   static void destruct_LHEvent(void *p) {
      typedef ::LHEvent current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
} // end of namespace ROOT for class ::LHEvent

//______________________________________________________________________________
void LHHit::Streamer(TBuffer &R__b)
{
   // Stream an object of class LHHit.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(LHHit::Class(),this);
   } else {
      R__b.WriteClassBuffer(LHHit::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_LHHit(void *p) {
      return  p ? new(p) ::LHHit : new ::LHHit;
   }
   static void *newArray_LHHit(Long_t nElements, void *p) {
      return p ? new(p) ::LHHit[nElements] : new ::LHHit[nElements];
   }
   // Wrapper around operator delete
   static void delete_LHHit(void *p) {
      delete (static_cast<::LHHit*>(p));
   }
   static void deleteArray_LHHit(void *p) {
      delete [] (static_cast<::LHHit*>(p));
   }
   static void destruct_LHHit(void *p) {
      typedef ::LHHit current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
} // end of namespace ROOT for class ::LHHit

//______________________________________________________________________________
void LHWave::Streamer(TBuffer &R__b)
{
   // Stream an object of class LHWave.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(LHWave::Class(),this);
   } else {
      R__b.WriteClassBuffer(LHWave::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_LHWave(void *p) {
      return  p ? new(p) ::LHWave : new ::LHWave;
   }
   static void *newArray_LHWave(Long_t nElements, void *p) {
      return p ? new(p) ::LHWave[nElements] : new ::LHWave[nElements];
   }
   // Wrapper around operator delete
   static void delete_LHWave(void *p) {
      delete (static_cast<::LHWave*>(p));
   }
   static void deleteArray_LHWave(void *p) {
      delete [] (static_cast<::LHWave*>(p));
   }
   static void destruct_LHWave(void *p) {
      typedef ::LHWave current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
} // end of namespace ROOT for class ::LHWave

//______________________________________________________________________________
void LHFiltedEvent::Streamer(TBuffer &R__b)
{
   // Stream an object of class LHFiltedEvent.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(LHFiltedEvent::Class(),this);
   } else {
      R__b.WriteClassBuffer(LHFiltedEvent::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_LHFiltedEvent(void *p) {
      return  p ? new(p) ::LHFiltedEvent : new ::LHFiltedEvent;
   }
   static void *newArray_LHFiltedEvent(Long_t nElements, void *p) {
      return p ? new(p) ::LHFiltedEvent[nElements] : new ::LHFiltedEvent[nElements];
   }
   // Wrapper around operator delete
   static void delete_LHFiltedEvent(void *p) {
      delete (static_cast<::LHFiltedEvent*>(p));
   }
   static void deleteArray_LHFiltedEvent(void *p) {
      delete [] (static_cast<::LHFiltedEvent*>(p));
   }
   static void destruct_LHFiltedEvent(void *p) {
      typedef ::LHFiltedEvent current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
} // end of namespace ROOT for class ::LHFiltedEvent

//______________________________________________________________________________
void LHRecEvent::Streamer(TBuffer &R__b)
{
   // Stream an object of class LHRecEvent.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(LHRecEvent::Class(),this);
   } else {
      R__b.WriteClassBuffer(LHRecEvent::Class(),this);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_LHRecEvent(void *p) {
      return  p ? new(p) ::LHRecEvent : new ::LHRecEvent;
   }
   static void *newArray_LHRecEvent(Long_t nElements, void *p) {
      return p ? new(p) ::LHRecEvent[nElements] : new ::LHRecEvent[nElements];
   }
   // Wrapper around operator delete
   static void delete_LHRecEvent(void *p) {
      delete (static_cast<::LHRecEvent*>(p));
   }
   static void deleteArray_LHRecEvent(void *p) {
      delete [] (static_cast<::LHRecEvent*>(p));
   }
   static void destruct_LHRecEvent(void *p) {
      typedef ::LHRecEvent current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
} // end of namespace ROOT for class ::LHRecEvent

namespace {
  void TriggerDictionaryInitialization_LHEventDict_Impl() {
    static const char* headers[] = {
"include/LHEvent.h",
nullptr
    };
    static const char* includePaths[] = {
"/Users/macbook/PycharmProjects/pythonProject/searchMonopole/paper/LHAASO_Monopole/Filt_Event/include",
"/Users/macbook/anaconda3/include/",
"/Users/macbook/PycharmProjects/pythonProject/searchMonopole/paper/LHAASO_Monopole/Filt_Event/",
nullptr
    };
    static const char* fwdDeclCode = R"DICTFWDDCLS(
#line 1 "LHEventDict dictionary forward declarations' payload"
#pragma clang diagnostic ignored "-Wkeyword-compat"
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern int __Cling_AutoLoading_Map;
class __attribute__((annotate("$clingAutoload$include/LHEvent.h")))  LHEvent;
class __attribute__((annotate("$clingAutoload$include/LHEvent.h")))  LHHit;
class __attribute__((annotate("$clingAutoload$include/LHEvent.h")))  LHWave;
class __attribute__((annotate("$clingAutoload$include/LHEvent.h")))  LHFiltedEvent;
class __attribute__((annotate("$clingAutoload$include/LHEvent.h")))  LHRecEvent;
)DICTFWDDCLS";
    static const char* payloadCode = R"DICTPAYLOAD(
#line 1 "LHEventDict dictionary payload"


#define _BACKWARD_BACKWARD_WARNING_H
// Inline headers
#include "include/LHEvent.h"

#undef  _BACKWARD_BACKWARD_WARNING_H
)DICTPAYLOAD";
    static const char* classesHeaders[] = {
"LHEvent", payloadCode, "@",
"LHFiltedEvent", payloadCode, "@",
"LHHit", payloadCode, "@",
"LHRecEvent", payloadCode, "@",
"LHWave", payloadCode, "@",
nullptr
};
    static bool isInitialized = false;
    if (!isInitialized) {
      TROOT::RegisterModule("LHEventDict",
        headers, includePaths, payloadCode, fwdDeclCode,
        TriggerDictionaryInitialization_LHEventDict_Impl, {}, classesHeaders, /*hasCxxModule*/false);
      isInitialized = true;
    }
  }
  static struct DictInit {
    DictInit() {
      TriggerDictionaryInitialization_LHEventDict_Impl();
    }
  } __TheDictionaryInitializer;
}
void TriggerDictionaryInitialization_LHEventDict() {
  TriggerDictionaryInitialization_LHEventDict_Impl();
}
