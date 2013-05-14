/*  This file is part of the Vc library. {{{

    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

}}}*/

#ifndef VC_COMMON_CONST_DATA_H
#define VC_COMMON_CONST_DATA_H

#include "macros.h"
/*OUTER_NAMESPACE_BEGIN*/
namespace Vc
{

ALIGN(64) extern unsigned int RandomState[16];

namespace Common
{

ALIGN(32) extern const unsigned int AllBitsSet[8];

} // namespace Common
} // namespace Vc
/*OUTER_NAMESPACE_END*/
#include "undomacros.h"

#endif // VC_COMMON_CONST_DATA_H
