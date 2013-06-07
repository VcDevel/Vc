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

#ifndef VC_MIC_SORTHELPER_H
#define VC_MIC_SORTHELPER_H

#include <tuple>
#include "types.h"
#include "macros.h"

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

template<typename T> struct SortHelper
{
    typedef typename VectorTypeHelper<T>::Type VectorType;
    static VectorType sort(VC_ALIGNED_PARAMETER(VectorType));
    template<typename... Vs> static std::tuple<Vs...> sort(VC_ALIGNED_PARAMETER(std::tuple<Vs...>));
};

Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_MIC_SORTHELPER_H
