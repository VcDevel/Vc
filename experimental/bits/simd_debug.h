// Debug utilities for use in the simd implementation -*- C++ -*-

// Copyright © 2015-2019 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH
//                       Matthias Kretz <m.kretz@gsi.de>
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the names of contributing organizations nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef _GLIBCXX_EXPERIMENTAL_SIMD_DEBUG_H_
#define _GLIBCXX_EXPERIMENTAL_SIMD_DEBUG_H_

#if defined _GLIBCXX_SIMD_DEBUG && !defined _GLIBCXX_SIMD_ENABLE_DEBUG
#define _GLIBCXX_SIMD_ENABLE_DEBUG 1
#endif

#ifdef _GLIBCXX_SIMD_ENABLE_DEBUG
#include <iostream>
#include <iomanip>
#include <sstream>
#endif  // _GLIBCXX_SIMD_ENABLE_DEBUG

_GLIBCXX_SIMD_BEGIN_NAMESPACE
enum class __area : unsigned {
    __disabled = 0,
    __enabled = 1,
    __ = __enabled,

#ifdef _GLIBCXX_SIMD_DEBUG

#define _Sine       0x0000000000000001ull
#define _Cosine     0x0000000000000002ull
#define _SIMD_TUPLE 0x0000000000000004ull
#define _Simd_view  0x0000000000000008ull
#define _Logarithm  0x0000000000000010ull
#define _Frexp      0x0000000000000020ull

    __Sine       = ((_GLIBCXX_SIMD_DEBUG) &       _Sine) ? __enabled : __disabled,
    __Cosine     = ((_GLIBCXX_SIMD_DEBUG) &     _Cosine) ? __enabled : __disabled,
    __SIMD_TUPLE = ((_GLIBCXX_SIMD_DEBUG) & _SIMD_TUPLE) ? __enabled : __disabled,
    __Simd_view  = ((_GLIBCXX_SIMD_DEBUG) & _Simd_view ) ? __enabled : __disabled,
    __Logarithm  = ((_GLIBCXX_SIMD_DEBUG) & _Logarithm ) ? __enabled : __disabled,
    __Frexp      = ((_GLIBCXX_SIMD_DEBUG) &     _Frexp ) ? __enabled : __disabled,
#undef _Sine
#undef _Cosine
#undef _SIMD_TUPLE
#undef _Simd_view
#undef _Logarithm
#undef _Frexp

#undef _GLIBCXX_SIMD_DEBUG

#else // _GLIBCXX_SIMD_DEBUG
    __Sine = __disabled,
    __Cosine = __disabled,
    __SIMD_TUPLE = __disabled,
    __Simd_view  = __disabled,
    __Logarithm  = __disabled,
    __Frexp = __disabled,
#endif // _GLIBCXX_SIMD_DEBUG
};

#define _GLIBCXX_SIMD_DEBUG(_Area)                                                       \
    std::experimental::__debug_stream<std::experimental::__area::_##_Area>(              \
        __PRETTY_FUNCTION__, __FILE__, __LINE__, std::experimental::__debug_instr_ptr())

#ifdef _GLIBCXX_SIMD_ENABLE_DEBUG
#define _GLIBCXX_SIMD_PRETTY_PRINT(var_) std::setw(16), #var_ " = ", (var_)

#define _GLIBCXX_SIMD_CONCAT_IMPL(a_, b_, c_) a_##b_##c_
#define _GLIBCXX_SIMD_CONCAT(a_, b_, c_) _GLIBCXX_SIMD_CONCAT_IMPL(a_, b_, c_)

#define _GLIBCXX_SIMD_DEBUG_DEFERRED(_Area, ...)                                         \
    const auto &_GLIBCXX_SIMD_CONCAT(_GLIBCXX_SIMD_deferred_, __LINE__, _) =             \
        __defer([&]() { _GLIBCXX_SIMD_DEBUG(_Area)                                       \
                        (__VA_ARGS__); });
#else   // _GLIBCXX_SIMD_ENABLE_DEBUG
#define _GLIBCXX_SIMD_PRETTY_PRINT(var_) (var_)

#define _GLIBCXX_SIMD_DEBUG_DEFERRED(_Area, ...)
#endif  // _GLIBCXX_SIMD_ENABLE_DEBUG

_GLIBCXX_SIMD_ALWAYS_INLINE void *__debug_instr_ptr()
{
  void* __ip = nullptr;
#if defined _GLIBCXX_SIMD_ENABLE_DEBUG
#ifdef __x86_64__
    asm volatile("lea 0(%%rip),%0" : "=r"(__ip));
#elif defined __i386__
    asm volatile("1: movl $1b,%0" : "=r"(__ip));
#elif defined __arm__
    asm volatile("mov %0,pc" : "=r"(__ip));
#elif defined __aarch64__
    asm volatile("adr %0,." : "=r"(__ip));
#endif
#endif  //__GNUC__
    return __ip;
}

template <__area> class __debug_stream;

#ifdef _GLIBCXX_SIMD_ENABLE_DEBUG
template <> class __debug_stream<__area::__enabled>
{
    std::stringstream __buffer;
    int __color = 31;

public:
    __debug_stream(const char *__func, const char *__file, int __line, void *__instr_ptr)
    {
        __buffer << "\033[1;40;" << __color << "mDEBUG: " << __file << ':' << __line
                 << " @ " << __instr_ptr << "\n       " << __func;
    }

    ~__debug_stream()
    {
        __buffer << "\033[0m\n";
        std::cout << __buffer.str() << std::flush;
    }

  template <class... _Ts>
  __debug_stream& operator()(const _Ts&... __args)
  {
    __color = __color > 37 ? 30 : __color + 1;
    __buffer << "\n\033[1;40;" << __color << "m      ";
    [](const std::initializer_list<int>&) {}({(__print(__args, int()), 0)...});
    return *this;
  }

private:
  template <class _Tp, class = decltype(__buffer << std::declval<const _Tp&>())>
  void __print(const _Tp& __x, int)
  {
    __buffer << ' ' << __x;
  }

  template <class _Tp,
	    class = decltype(__buffer << std::declval<const _Tp&>()[0])>
  void __print(const _Tp& __x, float)
  {
    using _U = __remove_cvref_t<decltype(__x[0])>;
    __buffer << " {" << +__x[0];
    for (size_t __i = 1; __i < sizeof(_Tp) / sizeof(_U); ++__i)
      {
	__buffer << ", " << +__x[__i];
      }
    __buffer << '}';
  }

  static char hexChar(char __x) { return __x + (__x > 9 ? 87 : 48); }
  template <class _Tp>
  void __print(const _Tp& __x, ...)
  {
    __buffer.put(' ');
    using _Bytes   = char[sizeof(_Tp)];
    auto&& __bytes = reinterpret_cast<const _Bytes&>(__x);
    int    __i     = -1;
    for (const unsigned char __b : __bytes)
      {
	if (++__i && (__i & 0x3) == 0)
	  {
	    __buffer.put('\'');
	  }
	__buffer.put(hexChar(__b >> 4));
	__buffer.put(hexChar(__b & 0xf));
      }
  }
};
#endif  // _GLIBCXX_SIMD_ENABLE_DEBUGGING

template <> class __debug_stream<__area::__disabled>
{
public:
    __debug_stream(const char *, const char *, int, void *) {}
    template <class... _Ts> const __debug_stream &operator()(_Ts &&...) const { return *this; }
};

template <class _F> class __defer_raii
{
public:
    // construct the object from the given callable
    template <class _FF> __defer_raii(_FF &&__f) : __cleanup_function(std::forward<_FF>(__f))
    {
    }

    // when the object goes out of scope call the cleanup function
    ~__defer_raii() { __cleanup_function(); }

private:
    _F __cleanup_function;
};

template <typename _F> __defer_raii<_F> __defer(_F &&__f) { return {std::forward<_F>(__f)}; }

_GLIBCXX_SIMD_END_NAMESPACE

#endif  // _GLIBCXX_EXPERIMENTAL_SIMD_DEBUG_H_
