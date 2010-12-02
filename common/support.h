/*  This file is part of the Vc library.

    Copyright (C) 2010 Matthias Kretz <kretz@kde.org>

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

*/

#ifndef VC_COMMON_SUPPORT_H
#define VC_COMMON_SUPPORT_H

#ifndef VC_GLOBAL_H
#error "Vc/global.h must be included first!"
#endif

namespace Vc
{

bool isImplementationSupported(Implementation);

#ifndef VC_COMPILE_LIB
/**
 * \ingroup Utilities
 *
 * Tests that the CPU and Operating System support the vector unit which was compiled for. This
 * function should be called before any other Vc functionality is used to check whether the program
 * will work. If this function returns \c false then the program should exit with
 * a non-zero exit status.
 *
 * If the program continues and makes use of any vector features not supported by
 * hard- or software then the program will crash.
 *
 * Example:
 * \code
 * int main()
 * {
 *   if (!Vc::currentImplementationSupported()) {
 *     std::cerr << "CPU or OS requirements not met for the compiled in vector unit!\n";
 *     exit -1;
 *   }
 *   ...
 * }
 * \endcode
 */
bool currentImplementationSupported()
{
    return isImplementationSupported(
#if VC_IMPL_Scalar
            ScalarImpl
#elif VC_IMPL_LRBni
#ifdef VC_LRBni_PROTOTYPE_H
            LRBniPrototypeImpl
#else
            LRBniImpl
#endif
#elif VC_IMPL_AVX
            AVXImpl
#elif VC_IMPL_SSE4a
            SSE4aImpl
#elif VC_IMPL_SSE4_2
            SSE42Impl
#elif VC_IMPL_SSE4_1
            SSE41Impl
#elif VC_IMPL_SSSE3
            SSSE3Impl
#elif VC_IMPL_SSE3
            SSE3Impl
#elif VC_IMPL_SSE2
            SSE2Impl
#else
            ERROR_unknown_vector_unit
#endif
            );
}
#endif // VC_COMPILE_LIB

} // namespace Vc

#endif // VC_COMMON_SUPPORT_H
