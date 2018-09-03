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
template<> inline const char *filename<float , Sincos>() { return TESTDATA_DIR "reference-sincos-sp.dat"; }
template<> inline const char *filename<double, Sincos>() { return TESTDATA_DIR "reference-sincos-dp.dat"; }
template<> inline const char *filename<float , Atan  >() { return TESTDATA_DIR "reference-atan-sp.dat"; }
template<> inline const char *filename<double, Atan  >() { return TESTDATA_DIR "reference-atan-dp.dat"; }
template<> inline const char *filename<float , Asin  >() { return TESTDATA_DIR "reference-asin-sp.dat"; }
template<> inline const char *filename<double, Asin  >() { return TESTDATA_DIR "reference-asin-dp.dat"; }
// template<> inline const char *filename<float , Acos  >() { return TESTDATA_DIR "reference-acos-sp.dat"; }
// template<> inline const char *filename<double, Acos  >() { return TESTDATA_DIR "reference-acos-dp.dat"; }
template<> inline const char *filename<float , Log   >() { return TESTDATA_DIR "reference-ln-sp.dat"; }
template<> inline const char *filename<double, Log   >() { return TESTDATA_DIR "reference-ln-dp.dat"; }
template<> inline const char *filename<float , Log2  >() { return TESTDATA_DIR "reference-log2-sp.dat"; }
template<> inline const char *filename<double, Log2  >() { return TESTDATA_DIR "reference-log2-dp.dat"; }
template<> inline const char *filename<float , Log10 >() { return TESTDATA_DIR "reference-log10-sp.dat"; }
template<> inline const char *filename<double, Log10 >() { return TESTDATA_DIR "reference-log10-dp.dat"; }

template<typename T>
static Array<SincosReference<T> > sincosReference()
{
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
}

template<typename T, Function Fun>
static Array<Reference<T> > referenceData()
{
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
}

//}}}1
// vim: foldmethod=marker
