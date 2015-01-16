//**************************************************************************
// vim: sw=4 sts=4 et tw=100
//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//*                                                                        *
//* Primary Authors: Sergey Gorbunov <sergey.gorbunov@cern.ch>             *
//*                  for The ALICE HLT Project.                            *
//*                                                                        *
//* Permission to use, copy, modify and distribute this software and its   *
//* documentation strictly for non-commercial purposes is hereby granted   *
//* without fee, provided that the above copyright notice appears in all   *
//* copies and that both the copyright notice and this permission notice   *
//* appear in the supporting documentation. The authors make no claims     *
//* about the suitability of this software for any purpose. It is          *
//* provided "as is" without express or implied warranty.                  *
//**************************************************************************

#include "spline2.h"
#include "../kdtree/simdize.h"
#include <iacaMarks.h>

#include <iostream>
#include <iomanip>

using namespace std;

// vim: foldmethod=marker
