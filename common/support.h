/*  This file is part of the Vc library.

    Copyright (C) 2010-2012 Matthias Kretz <kretz@kde.org>

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

/**
 * \ingroup Utilities
 *
 * Tests whether the given implementation is supported by the system the code is executing on.
 *
 * \return \c true if the OS and hardware support execution of instructions defined by \p impl.
 * \return \c false otherwise
 *
 * \param impl The SIMD target to test for.
 */
bool isImplementationSupported(Vc::Implementation impl);

#ifndef VC_COMPILE_LIB
/**
 * \ingroup Utilities
 *
 * Tests that the CPU and Operating System support the vector unit which was compiled for. This
 * function should be called before any other Vc functionality is used. It checks whether the program
 * will work. If this function returns \c false then the program should exit with a useful error
 * message before the OS has to kill it because of an invalid instruction exception.
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
 *
 * \return \c true if the OS and hardware support execution of the currently selected SIMD
 *                 instructions.
 * \return \c false otherwise
 */
#ifdef VC_GCC
    __attribute__((target("no-sse2,no-avx")))
#endif
bool currentImplementationSupported()
{
    return isImplementationSupported(
#ifdef VC_USE_VEX_CODING
            // everything will use VEX coding, so the system has to support AVX even if VC_IMPL_AVX
            // is not set
            AVXImpl
#else
            VC_IMPL
#endif
#ifdef VC_IMPL_XOP
            ) && isImplementationSupported(XopImpl
#endif
#ifdef VC_IMPL_FMA4
            ) && isImplementationSupported(Fma4Impl
#endif
            );
}
#endif // VC_COMPILE_LIB

} // namespace Vc

#endif // VC_COMMON_SUPPORT_H
