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

#ifndef VC_COMMON_LOADSTOREFLAGS_H
#define VC_COMMON_LOADSTOREFLAGS_H

#include "types.h"
#include "macros.h"

Vc_PUBLIC_NAMESPACE_BEGIN

/**
 * Hint for \ref Prefetch to select prefetches that mark the memory as exclusive.
 *
 * This hint may optimize the prefetch if the memory will subsequently be written to.
 */
struct Exclusive {};
/**
 * Hint for \ref Prefetch to select prefetches that mark the memory as shared.
 */
struct Shared {};

namespace
{

struct StreamingFlag {};
struct UnalignedFlag {};
struct PrefetchFlagBase {};
#ifdef VC_IMPL_MIC
template<size_t L1 = 8 * 64, size_t L2 = 64 * 64,
#else
// TODO: determine a good default for typical CPU use
template<size_t L1 = 16 * 64, size_t L2 = 128 * 64,
#endif
    typename ExclusiveOrShared_ = void> struct PrefetchFlag : public PrefetchFlagBase
{
    typedef ExclusiveOrShared_ ExclusiveOrShared;
    static constexpr size_t L1Stride = L1;
    static constexpr size_t L2Stride = L2;
    static constexpr bool IsExclusive = std::is_same<ExclusiveOrShared, Exclusive>::value;
    static constexpr bool IsShared = std::is_same<ExclusiveOrShared, Shared>::value;
};

template<typename Base, typename Default, typename... LoadStoreFlags> struct ExtractType
{
    typedef Default type;
};
template<typename Base, typename Default, typename T, typename... LoadStoreFlags> struct ExtractType<Base, Default, T, LoadStoreFlags...>
{
    typedef typename std::conditional<std::is_base_of<Base, T>::value, T, typename ExtractType<Base, Default, LoadStoreFlags...>::type>::type type;
};

// ICC warns about the constexpr members in LoadStoreFlags: member "LoadStoreFlags<Flags...>::IsAligned" was declared but never referenced
// who needs that warning, especially if it was referenced...
// The warning cannot be reenabled because it gets emitted whenever the LoadStoreFlags is instantiated
// somewhere, so it could be anywhere.
#ifdef VC_ICC
#pragma warning(disable: 177)
#endif
template<typename... Flags> struct LoadStoreFlags
{
private:
    // ICC doesn't grok this line:
    //template<typename Test> using TestFlag = std::is_same<typename ExtractType<StreamingFlag, void, Flags...>::type, void>;
    typedef typename ExtractType<PrefetchFlagBase, PrefetchFlag<0, 0>, Flags...>::type Prefetch;

public:
    constexpr LoadStoreFlags() {}

    static constexpr bool IsStreaming = !std::is_same<typename ExtractType<StreamingFlag, void, Flags...>::type, void>::value;
    static constexpr bool IsUnaligned = !std::is_same<typename ExtractType<UnalignedFlag, void, Flags...>::type, void>::value;
    static constexpr bool IsAligned = !IsUnaligned;
    static constexpr bool IsPrefetch = !std::is_same<typename ExtractType<PrefetchFlagBase, void, Flags...>::type, void>::value;
    static constexpr bool IsExclusivePrefetch = Prefetch::IsExclusive;
    static constexpr bool IsSharedPrefetch = Prefetch::IsShared;
    static constexpr size_t L1Stride = Prefetch::L1Stride;
    static constexpr size_t L2Stride = Prefetch::L2Stride;

    // The following EnableIf* convenience types cannot use enable_if because then no LoadStoreFlags type
    // could ever be instantiated. Instead these types are defined either as void* or void. The
    // function that does SFINAE then assigns "= nullptr" to this type. Thus, the ones with just
    // void result in substitution failure.
    typedef typename std::conditional<IsAligned   && !IsStreaming, void *, void>::type EnableIfAligned;
    typedef typename std::conditional<IsAligned   &&  IsStreaming, void *, void>::type EnableIfStreaming;
    typedef typename std::conditional<IsUnaligned && !IsStreaming, void *, void>::type EnableIfUnalignedNotStreaming;
    typedef typename std::conditional<IsUnaligned &&  IsStreaming, void *, void>::type EnableIfUnalignedAndStreaming;
    typedef typename std::conditional<IsUnaligned                , void *, void>::type EnableIfUnaligned;
    typedef typename std::conditional<IsPrefetch                 , void *, void>::type EnableIfPrefetch;
    typedef typename std::conditional<!IsPrefetch                , void *, void>::type EnableIfNotPrefetch;
};

template<> struct LoadStoreFlags<>
{
    constexpr LoadStoreFlags() {}

    static constexpr bool IsStreaming = false;
    static constexpr bool IsUnaligned = false;
    static constexpr bool IsAligned = !IsUnaligned;
    static constexpr bool IsPrefetch = false;
    static constexpr bool IsExclusivePrefetch = false;
    static constexpr bool IsSharedPrefetch = false;
    static constexpr size_t L1Stride = 0;
    static constexpr size_t L2Stride = 0;
    typedef void* EnableIfAligned;
    typedef void* EnableIfNotPrefetch;
};

template<typename... LFlags, typename... RFlags>
constexpr LoadStoreFlags<LFlags..., RFlags...> operator|(LoadStoreFlags<LFlags...>, LoadStoreFlags<RFlags...>)
{
    return LoadStoreFlags<LFlags..., RFlags...>();
}

template<typename Flags> struct EnableIfAligned : public std::enable_if<Flags::IsAligned && !Flags::IsStreaming, void *> {};
template<typename Flags> struct EnableIfStreaming : public std::enable_if<Flags::IsAligned && Flags::IsStreaming, void *> {};
template<typename Flags> struct EnableIfUnaligned : public std::enable_if<Flags::IsUnaligned, void *> {};
template<typename Flags> struct EnableIfUnalignedNotStreaming : public std::enable_if<Flags::IsUnaligned && !Flags::IsStreaming, void *> {};
template<typename Flags> struct EnableIfUnalignedAndStreaming : public std::enable_if<Flags::IsUnaligned && Flags::IsStreaming, void *> {};

} // anonymous namespace

typedef LoadStoreFlags<> AlignedT;
constexpr AlignedT Aligned;
constexpr LoadStoreFlags<StreamingFlag> Streaming;
constexpr LoadStoreFlags<UnalignedFlag> Unaligned;
constexpr LoadStoreFlags<PrefetchFlag<>> PrefetchDefault;

/**
 * \tparam L1
 * \tparam L2
 * \tparam ExclusiveOrShared
 */
template<size_t L1 = PrefetchFlag<>::L1Stride, size_t L2 = PrefetchFlag<>::L2Stride, typename ExclusiveOrShared = PrefetchFlag<>::ExclusiveOrShared>
struct Prefetch : public LoadStoreFlags<PrefetchFlag<L1, L2, ExclusiveOrShared>> {};

Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_COMMON_LOADSTOREFLAGS_H
