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

namespace Vc
{

enum Implementation {
    Scalar, SSE2, SSE3, SSSE3, SSE41, SSE42, SSE4a, AVX, LRBni
};

/**
 * Tests that the CPU and Operating System support the vector unit which was compiled for. This
 * function should be called before any other Vc functionality is used to check whether the program
 * will work. If currentImplementationSupported returns \c false then the program should exit with
 * a non-zero exit status. Should it continue and make use of any vector features not supported by
 * hard- or software then the program will crash.
 *
 * Example:
 * \code
 * int main()
 * {
 *   if (!Vc::currentImplementationSupported()) {
 *     std::cerr << "CPU or OS requirements not met for the compiled in vector unit!\n";
 *     exit 
 *   }
 * }
 */
bool currentImplementationSupported()
{
    return isImplementationSupported(
#if VC_IMPL_Scalar
            Scalar
#elif VC_IMPL_SSE2
            SSE2
#elif VC_IMPL_SSE3
            SSE3
#elif VC_IMPL_SSSE3
            SSSE3
#elif VC_IMPL_SSE4_1
            SSE41
#elif VC_IMPL_SSE4_2
            SSE4_2
#elif VC_IMPL_SSE4a
            SSE4a
#elif VC_IMPL_AVX
            AVX
#elif VC_IMPL_LRBni
            LRBni
#else
            ERROR_unknown_vector_unit
#endif
            );
}


bool isImplementationSupported(Implementation);

} // namespace Vc

#endif // VC_COMMON_SUPPORT_H
