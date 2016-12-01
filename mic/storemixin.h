/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_STOREMIXIN_H_
#define VC_MIC_STOREMIXIN_H_

#include <type_traits>

Vc_VERSIONED_NAMESPACE_BEGIN
namespace MIC
{

template<typename Parent, typename T> class StoreMixin
{
private:
    typedef typename VectorTypeHelper<T>::Type VectorType;
    using EntryType = T;
    typedef typename DetermineVectorEntryType<T>::Type VectorEntryType;
    typedef MIC::Mask<T> Mask;

    template<typename MemType> using UpDownC = UpDownConversion<VectorEntryType, typename std::decay<MemType>::type>;

    VectorType  data() const { return static_cast<const Parent *>(this)->data(); }
    VectorType &data()       { return static_cast<      Parent *>(this)->data(); }

public:
    template <typename U,
              typename Flags = DefaultStoreTag,
              typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
    Vc_INTRINSIC_L void store(U *mem, Flags = Flags()) const Vc_INTRINSIC_R;
    template <typename U,
              typename Flags = DefaultStoreTag,
              typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
    Vc_INTRINSIC_L void store(U *mem, Mask mask, Flags = Flags()) const Vc_INTRINSIC_R;
    // the following store overloads are here to support classes that have a cast operator to EntryType.
    // Without this overload GCC complains about not finding a matching store function.
    Vc_INTRINSIC void store(EntryType *mem) const
    {
        store<EntryType, DefaultStoreTag>(mem);
    }
    template <typename Flags> Vc_INTRINSIC void store(EntryType *mem, Flags flags) const
    {
        store<EntryType, Flags>(mem, flags);
    }
    Vc_INTRINSIC void store(EntryType *mem, Mask mask) const
    {
        store<EntryType, DefaultStoreTag>(mem, mask);
    }
    template <typename Flags> Vc_INTRINSIC void store(EntryType *mem, Mask mask, Flags flags) const
    {
        store<EntryType, Flags>(mem, mask, flags);
    }

    inline void store(VectorEntryType *mem, decltype(Streaming)) const;
};

}  // namespace MIC
Vc_VERSIONED_NAMESPACE_END
#endif // VC_MIC_STOREMIXIN_H_
