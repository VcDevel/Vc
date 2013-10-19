/*  This file is part of the Vc library. {{{

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_STOREMIXIN_H
#define VC_MIC_STOREMIXIN_H

#include <type_traits>

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

template<typename Parent, typename T> class StoreMixin
{
private:
    typedef typename VectorTypeHelper<T>::Type VectorType;
    typedef typename DetermineEntryType<T>::Type EntryType;
    typedef typename DetermineVectorEntryType<T>::Type VectorEntryType;
    typedef Vc_IMPL_NAMESPACE::Mask<T> Mask;

    // helper that specializes on VectorType
    typedef VectorHelper<VectorType> HV;

    // helper that specializes on T
    typedef VectorHelper<VectorEntryType> HT;

    template<typename MemType> using UpDownC = UpDownConversion<VectorEntryType, typename std::remove_cv<MemType>::type>;

    VectorType  data() const { return static_cast<const Parent *>(this)->data(); }
    VectorType &data()       { return static_cast<      Parent *>(this)->data(); }

public:
    template<typename T2, typename Flags = AlignedT> Vc_INTRINSIC_L void store(T2 *mem, Flags = Flags()) const Vc_INTRINSIC_R;
    template<typename T2, typename Flags = AlignedT> Vc_INTRINSIC_L void store(T2 *mem, Mask mask, Flags = Flags()) const Vc_INTRINSIC_R;
    // the following store overloads are here to support classes that have a cast operator to EntryType.
    // Without this overload GCC complains about not finding a matching store function.
    Vc_INTRINSIC void store(EntryType *mem) const { store<EntryType, AlignedT>(mem); }
    template<typename Flags = AlignedT> Vc_INTRINSIC void store(EntryType *mem, Flags flags) const { store<EntryType, Flags>(mem, flags); }
    Vc_INTRINSIC void store(EntryType *mem, Mask mask) const { store<EntryType, AlignedT>(mem, mask); }
    template<typename Flags = AlignedT> Vc_INTRINSIC void store(EntryType *mem, Mask mask, Flags flags) const { store<EntryType, Flags>(mem, mask, flags); }

    inline void store(VectorEntryType *mem, decltype(Streaming)) const;
};

Vc_NAMESPACE_END
#endif // VC_MIC_STOREMIXIN_H
