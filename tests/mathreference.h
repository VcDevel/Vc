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
    size_t size;
    const T *data;
    Array() : size(0), data(0) {}
    Array(size_t s, const T *p) : size(s), data(p) {}
};
template<typename T> struct StaticDeleter
{
    const T *ptr;
    StaticDeleter(const T *p) : ptr(p) {}
    ~StaticDeleter() { delete[] ptr; }
};

enum Function {
    Sincos, Atan, Asin, Acos, Log, Log2, Log10
};
template<typename T, Function F> static inline const char *filename();
template<> inline const char *filename<float , Sincos>() { return "reference-sincos-sp.dat"; }
template<> inline const char *filename<double, Sincos>() { return "reference-sincos-dp.dat"; }
template<> inline const char *filename<float , Atan  >() { return "reference-atan-sp.dat"; }
template<> inline const char *filename<double, Atan  >() { return "reference-atan-dp.dat"; }
template<> inline const char *filename<float , Asin  >() { return "reference-asin-sp.dat"; }
template<> inline const char *filename<double, Asin  >() { return "reference-asin-dp.dat"; }
// template<> inline const char *filename<float , Acos  >() { return "reference-acos-sp.dat"; }
// template<> inline const char *filename<double, Acos  >() { return "reference-acos-dp.dat"; }
template<> inline const char *filename<float , Log   >() { return "reference-ln-sp.dat"; }
template<> inline const char *filename<double, Log   >() { return "reference-ln-dp.dat"; }
template<> inline const char *filename<float , Log2  >() { return "reference-log2-sp.dat"; }
template<> inline const char *filename<double, Log2  >() { return "reference-log2-dp.dat"; }
template<> inline const char *filename<float , Log10 >() { return "reference-log10-sp.dat"; }
template<> inline const char *filename<double, Log10 >() { return "reference-log10-dp.dat"; }

#ifdef Vc_IMPL_MIC
extern "C" {
extern const Reference<double> _binary_reference_acos_dp_dat_start;
extern const Reference<double> _binary_reference_acos_dp_dat_end;
extern const Reference<float > _binary_reference_acos_sp_dat_start;
extern const Reference<float > _binary_reference_acos_sp_dat_end;
extern const Reference<double> _binary_reference_asin_dp_dat_start;
extern const Reference<double> _binary_reference_asin_dp_dat_end;
extern const Reference<float > _binary_reference_asin_sp_dat_start;
extern const Reference<float > _binary_reference_asin_sp_dat_end;
extern const Reference<double> _binary_reference_atan_dp_dat_start;
extern const Reference<double> _binary_reference_atan_dp_dat_end;
extern const Reference<float > _binary_reference_atan_sp_dat_start;
extern const Reference<float > _binary_reference_atan_sp_dat_end;
extern const Reference<double> _binary_reference_ln_dp_dat_start;
extern const Reference<double> _binary_reference_ln_dp_dat_end;
extern const Reference<float > _binary_reference_ln_sp_dat_start;
extern const Reference<float > _binary_reference_ln_sp_dat_end;
extern const Reference<double> _binary_reference_log10_dp_dat_start;
extern const Reference<double> _binary_reference_log10_dp_dat_end;
extern const Reference<float > _binary_reference_log10_sp_dat_start;
extern const Reference<float > _binary_reference_log10_sp_dat_end;
extern const Reference<double> _binary_reference_log2_dp_dat_start;
extern const Reference<double> _binary_reference_log2_dp_dat_end;
extern const Reference<float > _binary_reference_log2_sp_dat_start;
extern const Reference<float > _binary_reference_log2_sp_dat_end;
extern const SincosReference<double> _binary_reference_sincos_dp_dat_start;
extern const SincosReference<double> _binary_reference_sincos_dp_dat_end;
extern const SincosReference<float > _binary_reference_sincos_sp_dat_start;
extern const SincosReference<float > _binary_reference_sincos_sp_dat_end;
}

template <typename T, Function F>
inline std::pair<const T *, const T *> binary();
template <>
inline std::pair<const SincosReference<float> *, const SincosReference<float> *>
binary<SincosReference<float>, Sincos>()
{
    return std::make_pair(&_binary_reference_sincos_sp_dat_start,
                          &_binary_reference_sincos_sp_dat_end);
}
template <>
inline std::pair<const SincosReference<double> *,
                 const SincosReference<double> *>
binary<SincosReference<double>, Sincos>()
{
    return std::make_pair(&_binary_reference_sincos_dp_dat_start,
                          &_binary_reference_sincos_dp_dat_end);
}
template <>
inline std::pair<const Reference<float> *, const Reference<float> *>
binary<Reference<float>, Atan>()
{
    return std::make_pair(&_binary_reference_atan_sp_dat_start,
                          &_binary_reference_atan_sp_dat_end);
}
template <>
inline std::pair<const Reference<double> *, const Reference<double> *>
binary<Reference<double>, Atan>()
{
    return std::make_pair(&_binary_reference_atan_dp_dat_start,
                          &_binary_reference_atan_dp_dat_end);
}
template <>
inline std::pair<const Reference<float> *, const Reference<float> *>
binary<Reference<float>, Asin>()
{
    return std::make_pair(&_binary_reference_asin_sp_dat_start,
                          &_binary_reference_asin_sp_dat_end);
}
template <>
inline std::pair<const Reference<double> *, const Reference<double> *>
binary<Reference<double>, Asin>()
{
    return std::make_pair(&_binary_reference_asin_dp_dat_start,
                          &_binary_reference_asin_dp_dat_end);
}
template <>
inline std::pair<const Reference<float> *, const Reference<float> *>
binary<Reference<float>, Acos>()
{
    return std::make_pair(&_binary_reference_acos_sp_dat_start,
                          &_binary_reference_acos_sp_dat_end);
}
template <>
inline std::pair<const Reference<double> *, const Reference<double> *>
binary<Reference<double>, Acos>()
{
    return std::make_pair(&_binary_reference_acos_dp_dat_start,
                          &_binary_reference_acos_dp_dat_end);
}
template <>
inline std::pair<const Reference<float> *, const Reference<float> *>
binary<Reference<float>, Log>()
{
    return std::make_pair(&_binary_reference_ln_sp_dat_start,
                          &_binary_reference_ln_sp_dat_end);
}
template <>
inline std::pair<const Reference<double> *, const Reference<double> *>
binary<Reference<double>, Log>()
{
    return std::make_pair(&_binary_reference_ln_dp_dat_start,
                          &_binary_reference_ln_dp_dat_end);
}
template <>
inline std::pair<const Reference<float> *, const Reference<float> *>
binary<Reference<float>, Log2>()
{
    return std::make_pair(&_binary_reference_log2_sp_dat_start,
                          &_binary_reference_log2_sp_dat_end);
}
template <>
inline std::pair<const Reference<double> *, const Reference<double> *>
binary<Reference<double>, Log2>()
{
    return std::make_pair(&_binary_reference_log2_dp_dat_start,
                          &_binary_reference_log2_dp_dat_end);
}
template <>
inline std::pair<const Reference<float> *, const Reference<float> *>
binary<Reference<float>, Log10>()
{
    return std::make_pair(&_binary_reference_log10_sp_dat_start,
                          &_binary_reference_log10_sp_dat_end);
}
template <>
inline std::pair<const Reference<double> *, const Reference<double> *>
binary<Reference<double>, Log10>()
{
    return std::make_pair(&_binary_reference_log10_dp_dat_start,
                          &_binary_reference_log10_dp_dat_end);
}
#endif

template<typename T>
static Array<SincosReference<T> > sincosReference()
{
#ifdef Vc_IMPL_MIC
    const auto range = binary<SincosReference<T>, Sincos>();
    return {range.second - range.first, range.first};
#else
    static Array<SincosReference<T> > data;
    if (data.data == 0) {
        FILE *file = std::fopen(filename<T, Sincos>(), "rb");
        if (file) {
            std::fseek(file, 0, SEEK_END);
            const size_t size = std::ftell(file) / sizeof(SincosReference<T>);
            std::rewind(file);
            auto mem = new SincosReference<T>[size];
            static StaticDeleter<SincosReference<T> > _cleanup(data.data);
            data.size = std::fread(mem, sizeof(SincosReference<T>), size, file);
            data.data = mem;
            std::fclose(file);
        } else {
            FAIL() << "the reference data " << filename<T, Sincos>() << " does not exist in the current working directory.";
        }
    }
    return data;
#endif
}

template<typename T, Function Fun>
static Array<Reference<T> > referenceData()
{
#ifdef Vc_IMPL_MIC
    const auto range = binary<Reference<T>, Fun>();
    return {range.second - range.first, range.first};
#else
    static Array<Reference<T> > data;
    if (data.data == 0) {
        FILE *file = std::fopen(filename<T, Fun>(), "rb");
        if (file) {
            std::fseek(file, 0, SEEK_END);
            const size_t size = std::ftell(file) / sizeof(Reference<T>);
            std::rewind(file);
            auto mem = new Reference<T>[size];
            static StaticDeleter<Reference<T> > _cleanup(data.data);
            data.size = std::fread(mem, sizeof(Reference<T>), size, file);
            data.data = mem;
            std::fclose(file);
        } else {
            FAIL() << "the reference data " << filename<T, Fun>() << " does not exist in the current working directory.";
        }
    }
    return data;
#endif
}

//}}}1
// vim: foldmethod=marker
