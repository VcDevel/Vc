/*  This file is part of the Vc library. {{{
Copyright © 2009-2016 Matthias Kretz <kretz@kde.org>

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

#include <utility>
#include <cstdio>

template<typename T> struct SincosReference //{{{1
{
    T x, s, c;
};
template<typename T> struct Reference
{
    T x, ref;
};

template<typename T> struct Array
{
    std::size_t size_;
    const T *data_;
    Array() : size_(0), data_(nullptr) {}
    Array(size_t s, const T *p) : size_(s), data_(p) {}
    const T *begin() const { return data_; }
    const T *end() const { return data_ + size_; }
    std::size_t size() const { return size_; }
};

namespace function {
struct sincos{ static constexpr const char *const str = "sincos"; };
struct atan  { static constexpr const char *const str = "atan"; };
struct asin  { static constexpr const char *const str = "asin"; };
struct acos  { static constexpr const char *const str = "acos"; };
struct log   { static constexpr const char *const str = "ln"; };
struct log2  { static constexpr const char *const str = "log2"; };
struct log10 { static constexpr const char *const str = "log10"; };
}

template <class F> struct testdatatype_for_function {
    template <class T> using type = Reference<T>;
};
template <> struct testdatatype_for_function<function::sincos> {
    template <class T> using type = SincosReference<T>;
};
template <class F, class T>
using testdatatype_for_function_t =
    typename testdatatype_for_function<F>::template type<T>;

#ifdef Vc_LINK_TESTDATA
template <class F, class T> struct reference_data {
    using Ref = testdatatype_for_function_t<F, T>;
    static const Ref begin_, end_;
    static const Ref *begin() { return &begin_; }
    static const Ref *end() { return &end_; }
    static std::size_t size() { return end() - begin(); }
    const Ref &operator[](std::size_t i) const { return begin()[i]; }
};

template <class F, class T>
const testdatatype_for_function_t<F, T> reference_data<F, T>::begin_ = {};
template <class F, class T>
const testdatatype_for_function_t<F, T> reference_data<F, T>::end_ = {};

#else  // Vc_LINK_TESTDATA

template<typename T> struct StaticDeleter
{
    const T *ptr;
    StaticDeleter(const T *p) : ptr(p) {}
    ~StaticDeleter() { delete[] ptr; }
};

template <class F, class T> inline std::string filename()
{
    static_assert(std::is_floating_point<T>::value, "");
    static const auto cache = std::string("reference-") + F::str +
                              (std::is_same<T, float>::value ? "-sp" : "-dp") + ".dat";
    return cache;
}
#endif  // Vc_LINK_TESTDATA

template <class Fun, class T, class Ref = testdatatype_for_function_t<Fun, T>>
Array<Ref> referenceData()
{
#ifdef Vc_LINK_TESTDATA
    return {reference_data<Fun, T>::size(), reference_data<Fun,T>::begin()};
#else   // Vc_LINK_TESTDATA
    static Array<Ref> data;
    if (data.data_ == nullptr) {
        FILE *file = std::fopen(filename<Fun, T>().c_str(), "rb");
        if (file) {
            std::fseek(file, 0, SEEK_END);
            const size_t size = std::ftell(file) / sizeof(Ref);
            std::rewind(file);
            auto mem = new Ref[size];
            static StaticDeleter<Ref> _cleanup(data.data_);
            data.size_ = std::fread(mem, sizeof(Ref), size, file);
            data.data_ = mem;
            std::fclose(file);
        } else {
            FAIL() << "the reference data " << filename<Fun, T>()
                   << " does not exist in the current working directory.";
        }
    }
    return data;
#endif  // Vc_LINK_TESTDATA
}

//}}}1
// vim: foldmethod=marker
