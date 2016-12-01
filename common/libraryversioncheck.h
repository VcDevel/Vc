/*  This file is part of the Vc library. {{{
Copyright © 2016 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_COMMON_LIBRARYVERSIONCHECK_H_
#define VC_COMMON_LIBRARYVERSIONCHECK_H_

#if !defined(Vc_NO_VERSION_CHECK) && !defined(Vc_COMPILE_LIB)
#include <Vc/version.h>
#include "macros.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {
/**\internal
 * This function is implemented in the libVc library and checks whether the library is
 * compatible with the version information passed via the function parameters. If it is
 * incompatible the function prints a warning and aborts.
 */
#if 0
// This would make linking against libVc optional. However, the better solution (in use
// now) is to only include this header for function declarations that require libVc. Thus,
// no library check is done unless the library must be linked anyway.
void Vc_CAT2(checkLibraryAbi, Vc_LIBRARY_ABI_VERSION) [[gnu::weak]](
    unsigned int compileTimeAbi, unsigned int versionNumber, const char *versionString)
{
    auto &&unused = [](auto, auto, auto) {};
    unused(compileTimeAbi, versionNumber, versionString);
}
#else
void Vc_CDECL Vc_CAT2(checkLibraryAbi, Vc_LIBRARY_ABI_VERSION)(
    unsigned int compileTimeAbi, unsigned int versionNumber, const char *versionString);
#endif

/**\internal
 * This constructor function is compiled into every translation unit using weak linkage,
 * matching on the full version number. The function is therefore executed on startup
 * (before main) for as many TUs compiled with different Vc versions as are linked into
 * the executable (or its libraries). It calls Vc::detail::checkLibraryAbi to ensure the
 * TU was compiled with Vc headers that are compatible to the linked libVc.
 */
template <unsigned int = versionNumber()> struct RunLibraryVersionCheck {
    RunLibraryVersionCheck()
    {
        Vc_CAT2(checkLibraryAbi, Vc_LIBRARY_ABI_VERSION)(
            Vc_LIBRARY_ABI_VERSION, Vc_VERSION_NUMBER, Vc_VERSION_STRING);
    }
    static RunLibraryVersionCheck tmp;
};
template <unsigned int N> RunLibraryVersionCheck<N> RunLibraryVersionCheck<N>::tmp;

namespace
{
static auto library_version_check_ctor = RunLibraryVersionCheck<>::tmp;
}  // unnamed namespace

}  // namespace detail
Vc_VERSIONED_NAMESPACE_END
#endif  // !defined(Vc_NO_VERSION_CHECK) && !defined(Vc_COMPILE_LIB)

#endif  // VC_COMMON_LIBRARYVERSIONCHECK_H_
