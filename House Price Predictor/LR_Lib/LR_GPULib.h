// LR_Lib.h

#pragma once

using namespace System;

extern "C" __declspec(dllexport) int __cdecl Learn(float*, float*, unsigned int, unsigned int, unsigned int, float, float, float*, float*, float*);
extern "C" __declspec(dllexport) int __cdecl Predict(float*, unsigned int, unsigned int, float*, float *, float *, float *);