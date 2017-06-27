/*  This file is part of the Vc library. {{{
Copyright © 2016-2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DATAPAR_SMART_REFERENCE_H_
#define VC_DATAPAR_SMART_REFERENCE_H_

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {
template <class U, class Accessor = U, class Friend = U,
          class ValueType = typename U::value_type>
class smart_reference
{
    friend Accessor;
    friend Friend;
    Vc_INTRINSIC smart_reference(U &o, int i) noexcept : index(i), obj(o) {}
    static constexpr bool get_noexcept = noexcept(Accessor::get(declval<U &>(), int()));
    template <typename T> static constexpr bool set_noexcept()
    {
        return noexcept(Accessor::set(declval<U &>(), int(), declval<T>()));
    }

    int index;
    U &obj;

public:
    using value_type = ValueType;

    Vc_INTRINSIC smart_reference(const smart_reference &) = delete;

    Vc_INTRINSIC operator value_type() const noexcept(get_noexcept)
    {
        return Accessor::get(obj, index);
    }

    template <typename T>
        Vc_INTRINSIC
            enable_if<std::is_same<void, decltype(Accessor::set(declval<U &>(), int(),
                                                                declval<T>()))>::value,
                      smart_reference &>
            operator=(T &&x) && noexcept(set_noexcept<T>())
    {
        Accessor::set(obj, index, std::forward<T>(x));
        return *this;
    }

// TODO: improve with operator.()

#define Vc_OP_(op_)                                                                      \
    template <typename T,                                                                \
              typename R = decltype(declval<const value_type &>() op_ declval<T>())>     \
        Vc_INTRINSIC enable_if<                                                          \
            std::is_same<void, decltype(Accessor::set(declval<U &>(), int(),             \
                                                      declval<R &&>()))>::value,         \
            smart_reference &>                                                           \
        operator op_##=(T &&x) &&                                                        \
        noexcept(                                                                        \
            get_noexcept &&                                                              \
            set_noexcept<decltype(declval<const value_type &>() op_ declval<T>())>())    \
    {                                                                                    \
        const value_type &lhs = Accessor::get(obj, index);                               \
        Accessor::set(obj, index, lhs op_ std::forward<T>(x));                           \
        return *this;                                                                    \
    }                                                                                    \
    template <typename T>                                                                \
    Vc_INTRINSIC decltype(declval<const value_type &>() op_ declval<T>()) operator op_(  \
        T &&x) const noexcept(get_noexcept)                                              \
    {                                                                                    \
        const value_type &lhs = Accessor::get(obj, index);                               \
        return lhs op_ std::forward<T>(x);                                               \
    }
    Vc_ALL_ARITHMETICS(Vc_OP_);
    Vc_ALL_SHIFTS(Vc_OP_);
    Vc_ALL_BINARY(Vc_OP_);
#undef Vc_OP_

    template <typename = void>
        Vc_INTRINSIC smart_reference &operator++() &&
        noexcept(noexcept(declval<value_type &>() = Accessor::get(declval<U &>(),
                                                                  int())) &&
                 set_noexcept<decltype(++declval<value_type &>())>())
    {
        value_type x = Accessor::get(obj, index);
        Accessor::set(obj, index, ++x);
        return *this;
    }

    template <typename = void>
        Vc_INTRINSIC value_type operator++(int) &&
        noexcept(noexcept(declval<value_type &>() = Accessor::get(declval<U &>(),
                                                                  int())) &&
                 set_noexcept<decltype(declval<value_type &>()++)>())
    {
        const value_type r = Accessor::get(obj, index);
        value_type x = r;
        Accessor::set(obj, index, ++x);
        return r;
    }

    template <typename = void>
        Vc_INTRINSIC smart_reference &operator--() &&
        noexcept(noexcept(declval<value_type &>() = Accessor::get(declval<U &>(),
                                                                  int())) &&
                 set_noexcept<decltype(--declval<value_type &>())>())
    {
        value_type x = Accessor::get(obj, index);
        Accessor::set(obj, index, --x);
        return *this;
    }

    template <typename = void>
        Vc_INTRINSIC value_type operator--(int) &&
        noexcept(noexcept(declval<value_type &>() = Accessor::get(declval<U &>(),
                                                                  int())) &&
                 set_noexcept<decltype(declval<value_type &>()--)>())
    {
        const value_type r = Accessor::get(obj, index);
        value_type x = r;
        Accessor::set(obj, index, --x);
        return r;
    }

    friend inline void swap(smart_reference &&a, smart_reference &&b) noexcept(
        all<std::is_nothrow_constructible<value_type, smart_reference &>,
            std::is_nothrow_assignable<smart_reference &&, value_type &&>>::value)
    {
        value_type tmp(a);
        static_cast<smart_reference &&>(a) = static_cast<value_type>(b);
        static_cast<smart_reference &&>(b) = std::move(tmp);
    }

    friend inline void swap(value_type &a, smart_reference &&b) noexcept(
        all<std::is_nothrow_constructible<value_type, value_type &&>,
            std::is_nothrow_assignable<value_type &, value_type &&>,
            std::is_nothrow_assignable<smart_reference &&, value_type &&>>::value)
    {
        value_type tmp(std::move(a));
        a = static_cast<value_type>(b);
        static_cast<smart_reference &&>(b) = std::move(tmp);
    }

    friend inline void swap(smart_reference &&a, value_type &b) noexcept(
        all<std::is_nothrow_constructible<value_type, smart_reference &>,
            std::is_nothrow_assignable<value_type &, value_type &&>,
            std::is_nothrow_assignable<smart_reference &&, value_type &&>>::value)
    {
        value_type tmp(a);
        static_cast<smart_reference &&>(a) = std::move(b);
        b = std::move(tmp);
    }
};
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DATAPAR_SMART_REFERENCE_H_

// vim: foldmethod=marker
