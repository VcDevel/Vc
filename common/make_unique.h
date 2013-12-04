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

#ifndef VC_COMMON_MAKE_UNIQUE_H
#define VC_COMMON_MAKE_UNIQUE_H

#include <memory>

#include "macros.h"

Vc_NAMESPACE_BEGIN(Common)

template<typename T> struct Deleter
{
    Vc_ALWAYS_INLINE void operator()(T *ptr) {
        ptr->~T();
        Vc::free(ptr);
    }
};

template<class T, MallocAlignment A = Vc::AlignOnVector, class... Args>
inline std::unique_ptr<T, Deleter<T>> make_unique(Args&&... args)
{
    return std::unique_ptr<T, Deleter<T>>(new(Vc::malloc<T, A>(1)) T(std::forward<Args>(args)...));
}

Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_COMMON_MAKE_UNIQUE_H
