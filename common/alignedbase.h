/*  This file is part of the Vc library. {{{
Copyright © 2015 Matthias Kretz <kretz@kde.org>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_COMMON_ALIGNEDBASE_H_
#define VC_COMMON_ALIGNEDBASE_H_

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{
template <std::size_t> Vc_INTRINSIC void *aligned_malloc(std::size_t);
Vc_ALWAYS_INLINE void free(void *);
}  // namespace Common

/**
 * \ingroup Utilities
 *
 * Helper class to ensure a given alignment.
 *
 * This class reimplements the \c new and \c delete operators to align objects allocated
 * on the heap suitably with the specified alignment \c Alignment.
 *
 * \see Vc::VectorAlignedBase
 * \see Vc::MemoryAlignedBase
 */
template <std::size_t Alignment> struct alignas(Alignment) AlignedBase
{
    Vc_FREE_STORE_OPERATORS_ALIGNED(Alignment)
};

/**
 * \ingroup Utilities
 *
 * Helper class to ensure suitable alignment for a given vector type.
 *
 * This class reimplements the \c new and \c delete operators to align objects allocated
 * on the heap suitably for objects of type \p V. This is necessary for types containing
 * over-aligned types (i.e. vector types) since the standard \c new operator does not
 * adhere to the alignment requirements of the type.
 *
 * \see Vc::MemoryAlignedBase
 * \see Vc::AlignedBase
 */
template <typename V> using VectorAlignedBase = AlignedBase<alignof(V)>;

/**
 * \ingroup Utilities
 *
 * Helper class to ensure suitable alignment for arrays of scalar objects of a given vector type.
 *
 * This class reimplements the \c new and \c delete operators to align objects allocated
 * on the heap suitably for arrays of type \p V::EntryType. Subsequent load and store
 * operations are safe to use the aligned variant.
 *
 * \see Vc::VectorAlignedBase
 * \see Vc::AlignedBase
 */
template <typename V> using MemoryAlignedBase = AlignedBase<V::MemoryAlignment>;
}

#endif  // VC_COMMON_ALIGNEDBASE_H_

// vim: foldmethod=marker
