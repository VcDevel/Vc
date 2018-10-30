#ifndef _GLIBCXX_EXPERIMENTAL_SIMD_DEBUG_H_
#define _GLIBCXX_EXPERIMENTAL_SIMD_DEBUG_H_

#pragma GCC system_header

#if defined _GLIBCXX_SIMD_DEBUG && !defined _GLIBCXX_SIMD_ENABLE_DEBUG
#define _GLIBCXX_SIMD_ENABLE_DEBUG 1
#endif

#ifdef _GLIBCXX_SIMD_ENABLE_DEBUG
#include <iostream>
#include <iomanip>
#include <sstream>
#endif  // _GLIBCXX_SIMD_ENABLE_DEBUG

_GLIBCXX_SIMD_BEGIN_NAMESPACE
namespace detail
{
#define _GLIBCXX_SIMD_PRETTY_FUNCTION __PRETTY_FUNCTION__

enum class area : unsigned {
    _disabled = 0,
    _enabled = 1,
    _ = _enabled,

#ifdef _GLIBCXX_SIMD_DEBUG

#define sine       0x0000000000000001ull
#define cosine     0x0000000000000002ull
#define simd_tuple 0x0000000000000004ull
#define simd_view  0x0000000000000008ull
#define logarithm  0x0000000000000010ull
#define frexp      0x0000000000000020ull

    _sine       = ((_GLIBCXX_SIMD_DEBUG) &       sine) ? _enabled : _disabled,
    _cosine     = ((_GLIBCXX_SIMD_DEBUG) &     cosine) ? _enabled : _disabled,
    _simd_tuple = ((_GLIBCXX_SIMD_DEBUG) & simd_tuple) ? _enabled : _disabled,
    _simd_view  = ((_GLIBCXX_SIMD_DEBUG) & simd_view ) ? _enabled : _disabled,
    _logarithm  = ((_GLIBCXX_SIMD_DEBUG) & logarithm ) ? _enabled : _disabled,
    _frexp      = ((_GLIBCXX_SIMD_DEBUG) &     frexp ) ? _enabled : _disabled,
#undef sine
#undef cosine
#undef simd_tuple
#undef simd_view
#undef logarithm
#undef frexp

#undef _GLIBCXX_SIMD_DEBUG

#else // _GLIBCXX_SIMD_DEBUG
    _sine = _disabled,
    _cosine = _disabled,
    _simd_tuple = _disabled,
    _simd_view  = _disabled,
    _logarithm  = _disabled,
    _frexp = _disabled,
#endif // _GLIBCXX_SIMD_DEBUG
};

#define _GLIBCXX_SIMD_DEBUG(area_)                                                                  \
    std::experimental::detail::debug_stream<std::experimental::detail::area::_##area_>(_GLIBCXX_SIMD_PRETTY_FUNCTION, __FILE__,   \
                                                         __LINE__, std::experimental::detail::debug_instr_ptr())

#ifdef _GLIBCXX_SIMD_ENABLE_DEBUG
#define _GLIBCXX_SIMD_PRETTY_PRINT(var_) std::setw(16), #var_ " = ", (var_)

#define _GLIBCXX_SIMD_DEBUG_DEFERRED(area_, ...)                                                    \
    const auto &_GLIBCXX_SIMD_CONCAT(_GLIBCXX_SIMD_deferred_, __LINE__, _) =                                   \
        detail::defer([&]() { _GLIBCXX_SIMD_DEBUG(area_)(__VA_ARGS__); });
#else   // _GLIBCXX_SIMD_ENABLE_DEBUG
#define _GLIBCXX_SIMD_PRETTY_PRINT(var_) (var_)

#define _GLIBCXX_SIMD_DEBUG_DEFERRED(area_, ...)
#endif  // _GLIBCXX_SIMD_ENABLE_DEBUG

_GLIBCXX_SIMD_ALWAYS_INLINE void *debug_instr_ptr()
{
    void *ip = nullptr;
#if defined _GLIBCXX_SIMD_ENABLE_DEBUG
#ifdef __x86_64__
    asm volatile("lea 0(%%rip),%0" : "=r"(ip));
#elif defined __i386__
    asm volatile("1: movl $1b,%0" : "=r"(ip));
#elif defined __arm__
    asm volatile("mov %0,pc" : "=r"(ip));
#endif
#endif  //__GNUC__
    return ip;
}

template <area> class debug_stream;

#ifdef _GLIBCXX_SIMD_ENABLE_DEBUG
template <> class debug_stream<area::_enabled>
{
    std::stringstream buffer;
    int color = 31;

public:
    debug_stream(const char *func, const char *file, int line, void *instr_ptr)
    {
        buffer << "\033[1;40;" << color << "mDEBUG: " << file << ':' << line << " @ "
               << instr_ptr << "\n       " << func;
    }

    ~debug_stream()
    {
        buffer << "\033[0m\n";
        std::cout << buffer.str() << std::flush;
    }

    template <class... Ts> debug_stream &operator()(const Ts &... args)
    {
        color = color > 37 ? 30 : color + 1;
        buffer << "\n\033[1;40;" << color << "m       ";
        //buffer << "\n        ";
#if 0 // __cpp_fold_expressions
        buffer << ... << std::forward<Ts>(args);
#else
        [](const std::initializer_list<int> &) {}({(print(args, int()), 0)...});
#endif
        return *this;
    }

private:
    template <class T, class = decltype(buffer << std::declval<const T &>())>
    void print(const T &x, int)
    {
        buffer << x;
    }

    static char hexChar(char x) { return x + (x > 9 ? 87 : 48); }
    template <class T> void print(const T &x, float)
    {
        using Bytes = char[sizeof(T)];
        auto &&bytes = reinterpret_cast<const Bytes &>(x);
        int i = -1;
        for (const unsigned char b : bytes) {
            if (++i && (i & 0x3) == 0) {
                buffer.put('\'');
            }
            buffer.put(hexChar(b >> 4));
            buffer.put(hexChar(b & 0xf));
        }
    }
};
#endif  // _GLIBCXX_SIMD_ENABLE_DEBUGGING

template <> class debug_stream<area::_disabled>
{
public:
    debug_stream(const char *, const char *, int, void *) {}
    template <class... Ts> const debug_stream &operator()(Ts &&...) const { return *this; }
};

template <typename F> class defer_raii
{
public:
    // construct the object from the given callable
    template <typename FF> defer_raii(FF &&f) : cleanup_function(std::forward<FF>(f)) {}

    // when the object goes out of scope call the cleanup function
    ~defer_raii() { cleanup_function(); }

private:
    F cleanup_function;
};

template <typename F> detail::defer_raii<F> defer(F && f) { return {std::forward<F>(f)}; }

}  // namespace detail
_GLIBCXX_SIMD_END_NAMESPACE

#endif  // _GLIBCXX_EXPERIMENTAL_SIMD_DEBUG_H_
