#ifndef _GLIBCXX_EXPERIMENTAL_SIMD_MATH_H
#define _GLIBCXX_EXPERIMENTAL_SIMD_MATH_H

#pragma GCC system_header

#if __cplusplus >= 201703L

_GLIBCXX_SIMD_BEGIN_NAMESPACE
// __divides {{{1
template <class _T, class = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC _T __divides(_T a, _T b)
{
    using _U = typename Traits::value_type;
    constexpr bool is_byte = sizeof(_U) == 1;
    constexpr bool is_word = sizeof(_U) == 2;
    constexpr bool is_dword = sizeof(_U) == 4;
    constexpr bool is_ymm = sizeof(_T) == 32;
    constexpr bool is_xmm = sizeof(_T) == 16;

    if constexpr (is_dword && ((is_xmm && __have_avx) || (is_ymm && __have_avx512f))) {
        return convert<_U>(convert<double>(a) / convert<double>(b));
    } else if constexpr (is_dword) {  // really better with is_xmm?
        auto x = convert_all<__vector_type_t<double, Traits::width / 2>>(a);
        auto y = convert_all<__vector_type_t<double, Traits::width / 2>>(b);
        return convert<_T>(x[0] / y[0], x[1] / y[1]);
    } else if constexpr (is_word) {
        if constexpr ((is_xmm && __have_avx) || (is_ymm && __have_avx512f)) {
            return convert<_T>(convert<float>(a) / convert<float>(b));
        } else {
            auto x = convert_all<__vector_type_t<float, Traits::width / 2>>(a);
            auto y = convert_all<__vector_type_t<float, Traits::width / 2>>(b);
            return convert<_T>(x[0] / y[0], x[1] / y[1]);
        }
    } else if constexpr (is_byte && is_xmm && __have_avx512f) {
        return convert<_T>(convert<float>(a) / convert<float>(b));
    } else if constexpr (is_byte && ((is_xmm && __have_avx) || is_ymm && __have_avx512f)) {
        auto x = convert_all<__vector_type_t<float, Traits::width / 2>>(a);
        auto y = convert_all<__vector_type_t<float, Traits::width / 2>>(b);
        return convert<_T>(x[0] / y[0], x[1] / y[1]);
    } else if constexpr (is_byte) {
        auto x = convert_all<__vector_type_t<float, Traits::width / 4>>(a);
        auto y = convert_all<__vector_type_t<float, Traits::width / 4>>(b);
        return convert<_T>(x[0] / y[0], x[1] / y[1], x[2] / y[2], x[3] / y[3]);
    } else {
        return a / b;
    }
}
// __bit_shift_left{{{1
template <class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC __storage<_T, _N> constexpr __bit_shift_left(__storage<_T, _N> a, int b)
{
    static_assert(std::is_integral<_T>::value, "__bit_shift_left is only supported for integral types");
    if constexpr (sizeof(_T) == 1) {
        // (cf. https://gcc.gnu.org/bugzilla/show_bug.cgi?id=83894)
        if (__builtin_constant_p(b)) {
            if (b == 0) {
                return a;
            } else if (b == 1) {
                return a.d + a.d;
            } else if (b > 1 && b < 8) {
                const __uchar mask = (0xff << b) & 0xff;
                using _V = decltype(a);
                using In = typename _V::intrin_type;
                return reinterpret_cast<In>(__storage_bitcast<ushort>(a).d << b) &
                       _V::broadcast(mask).intrin();
            } else {
                __builtin_unreachable();
            }
        }
        if constexpr (_N == 16 && __have_sse2) {
            if constexpr (__have_avx512bw_vl) {
                return _mm256_cvtepi16_epi8(reinterpret_cast<__m256i>(
                    reinterpret_cast<__vector_type_t<ushort, 16>>(_mm256_cvtepi8_epi16(a))
                    << b));
            } else {
                using vshort = __vector_type_t<ushort, 8>;
                const auto mask = ((~vshort() >> 8) << b) ^ (~vshort() << 8);
                return __to_storage((reinterpret_cast<vshort>(a.d) << b) & mask);
            }
        } else if constexpr (_N == 32 && __have_avx2) {
            if constexpr(__have_avx512bw) {
                return _mm512_cvtepi16_epi8(reinterpret_cast<__m512i>(
                    reinterpret_cast<__vector_type_t<ushort, 32>>(_mm512_cvtepi8_epi16(a))
                    << b));
            } else {
                using vshort = __vector_type_t<ushort, 16>;
                const auto mask = ((~vshort() >> 8) << b) ^ (~vshort() << 8);
                return __to_storage((reinterpret_cast<vshort>(a.d) << b) & mask);
            }
        } else if constexpr (_N == 64 && __have_avx512bw) {
            using vshort = __vector_type_t<ushort, 32>;
            const auto mask = ((~vshort() >> 8) << b) ^ (~vshort() << 8);
            return __to_storage((reinterpret_cast<vshort>(a.d) << b) & mask);
        } else {
            static_assert(!std::is_same_v<_T, _T>);
        }
    } else {
        return a.d << b;
    }
}

template <class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC __storage<_T, _N> __bit_shift_left(__storage<_T, _N> a, __storage<_T, _N> b)
{
    static_assert(std::is_integral<_T>::value,
                  "__bit_shift_left is only supported for integral types");
    if constexpr (sizeof(_T) == 2 && sizeof(a) == 16 && !__have_avx2) {
        __vector_type_t<int, 4> shift = __storage_bitcast<int>(b).d + (0x03f8'03f8 >> 3);
        return multiplies(
            a,
            __storage<_T, _N>(
                _mm_cvttps_epi32(reinterpret_cast<__m128>(shift << 23)) |
                (_mm_cvttps_epi32(reinterpret_cast<__m128>(shift >> 16 << 23)) << 16)));
    } else if constexpr (sizeof(_T) == 4 && sizeof(a) == 16 && !__have_avx2) {
        return __storage_bitcast<_T>(
            multiplies(a, __storage<_T, _N>(_mm_cvttps_epi32(
                              reinterpret_cast<__m128>((b.d << 23) + 0x3f80'0000)))));
    } else if constexpr (sizeof(_T) == 8 && sizeof(a) == 16 && !__have_avx2) {
        const auto lo = _mm_sll_epi64(a, b);
        const auto hi = _mm_sll_epi64(a, _mm_unpackhi_epi64(b, b));
#ifdef _GLIBCXX_SIMD_HAVE_SSE4_1
        return _mm_blend_epi16(lo, hi, 0xf0);
#else
        // return __make_storage<__llong>(reinterpret_cast<__vector_type_t<__llong, 2>>(lo)[0],
        // reinterpret_cast<__vector_type_t<__llong, 2>>(hi)[1]);
        return __to_storage(
            _mm_move_sd(__intrin_cast<__m128d>(hi), __intrin_cast<__m128d>(lo)));
#endif
    } else if constexpr (__have_avx512f && sizeof(_T) == 8 && _N == 8) {
        return _mm512_sllv_epi64(a, b);
    } else if constexpr (__have_avx2 && sizeof(_T) == 8 && _N == 4) {
        return _mm256_sllv_epi64(a, b);
    } else if constexpr (__have_avx2 && sizeof(_T) == 8 && _N == 2) {
        return _mm_sllv_epi64(a, b);
    } else if constexpr (__have_avx512f && sizeof(_T) == 4 && _N == 16) {
        return _mm512_sllv_epi32(a, b);
    } else if constexpr (__have_avx2 && sizeof(_T) == 4 && _N == 8) {
        return _mm256_sllv_epi32(a, b);
    } else if constexpr (__have_avx2 && sizeof(_T) == 4 && _N == 4) {
        return _mm_sllv_epi32(a, b);
    } else if constexpr (sizeof(_T) == 2) {
        if constexpr (_N == 32 && __have_avx512bw) {
            return _mm512_sllv_epi16(a, b);
        } else if constexpr (_N == 16 && __have_avx512bw_vl) {
            return _mm256_sllv_epi16(a, b);
        } else if constexpr (_N == 16 && __have_avx512bw) {
            return __lo256(
                _mm512_sllv_epi16(_mm512_castsi256_si512(a), _mm512_castsi256_si512(b)));
        } else if constexpr (_N == 16) {
            const auto aa = __vector_cast<unsigned>(a.d);
            const auto bb = __vector_cast<unsigned>(b.d);
            return _mm256_blend_epi16(__auto_cast(aa << (bb & 0x0000ffffu)),
                                      __auto_cast((aa & 0xffff0000u) << (bb >> 16)), 0xaa);
        } else if constexpr (_N == 8 && __have_avx512bw_vl) {
            return _mm_sllv_epi16(a, b);
        } else if constexpr (_N == 8 && __have_avx512bw) {
            return _mm512_sllv_epi16(_mm512_castsi128_si512(a),
                                     _mm512_castsi128_si512(b));
        } else if constexpr (_N == 8) {
            const auto aa = __vector_cast<unsigned>(a.d);
            const auto bb = __vector_cast<unsigned>(b.d);
            return _mm_blend_epi16(__auto_cast(aa << (bb & 0x0000ffffu)),
                                   __auto_cast((aa & 0xffff0000u) << (bb >> 16)), 0xaa);
        } else {
            __assert_unreachable<_T>();
        }
    } else if constexpr (sizeof(_T) == 1) {
        if constexpr (_N == 64 && __have_avx512bw) {
            return concat(_mm512_cvtepi16_epi8(_mm512_sllv_epi16(
                              _mm512_cvtepu8_epi16(__lo256(__vector_cast<__llong>(a))),
                              _mm512_cvtepu8_epi16(__lo256(__vector_cast<__llong>(b))))),
                          _mm512_cvtepi16_epi8(_mm512_sllv_epi16(
                              _mm512_cvtepu8_epi16(__hi256(__vector_cast<__llong>(a))),
                              _mm512_cvtepu8_epi16(__hi256(__vector_cast<__llong>(b))))));
        } else if constexpr (_N == 32 && __have_avx512bw) {
            return _mm512_cvtepi16_epi8(
                _mm512_sllv_epi16(_mm512_cvtepu8_epi16(a), _mm512_cvtepu8_epi16(b)));
        } else if constexpr (_N == 16 && __have_avx512bw_vl) {
            return _mm256_cvtepi16_epi8(
                _mm256_sllv_epi16(_mm256_cvtepu8_epi16(a), _mm256_cvtepu8_epi16(b)));
        } else if constexpr (_N == 16 && __have_avx512bw) {
            return __lo128(_mm512_cvtepi16_epi8(
                _mm512_sllv_epi16(_mm512_cvtepu8_epi16(_mm512_castsi256_si512(a)),
                                  _mm512_cvtepu8_epi16(_mm512_castsi256_si512(b)))));
        } else {
            auto mask_from_bit = [](__vector_type_t<_T, _N> x, int bit) {
                auto y = __vector_cast<short>(x) << bit;
                if constexpr (__have_sse4_1) {
                    return __to_intrin(y);
                } else {
                    return __to_intrin(__vector_cast<__schar>(y) < 0);
                }
            };
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand. left
            // => valid input range for each element of b is [0, 7]
            // => only the 3 low bits of b are relevant
            // do a =<< 4 where b[2] is set
            auto a4 = __vector_cast<__uchar>(__vector_cast<short>(a.d) << 4);
            if constexpr (std::is_unsigned_v<_T>) {
                // shift into or over the sign bit is UB => never spills into a neighbor
                a4 &= 0xf0u;
            }
            a = __blend(mask_from_bit(b, 5), a, __to_intrin(a4));
            // do a =<< 2 where b[1] is set
            // shift into or over the sign bit is UB => never spills into a neighbor
            const auto a2 = std::is_signed_v<_T> ? __to_intrin(__vector_cast<short>(a.d) << 2)
                                                : __to_intrin(a.d << 2);
            a = __blend(mask_from_bit(b, 6), a, a2);
            // do a =<< 1 where b[0] is set
            return __blend(mask_from_bit(b, 7), a, __to_intrin(a.d + a.d));
        }
    } else {
        return a.d << b.d;
    }
}

// }}}
// __bit_shift_right{{{1
template <class _T, class Traits = __vector_traits<_T>> _T __bit_shift_right(_T a, _T b)
{
    using _U = typename Traits::value_type;
    constexpr bool is_byte = sizeof(_U) == 1;
    constexpr bool is_word = sizeof(_U) == 2;
    constexpr bool is_dword = sizeof(_U) == 4;
    constexpr bool is_signed = std::is_signed_v<_U>;
    constexpr bool is_zmm = sizeof(_T) == 64;
    constexpr bool is_ymm = sizeof(_T) == 32;
    constexpr bool is_xmm = sizeof(_T) == 16;

    const auto ai = __to_intrin(a);
    const auto bi = __to_intrin(b);

    if constexpr (is_byte && is_xmm && __have_avx512bw_vl) {  //{{{2
        return is_signed ? _mm256_cvtepi16_epi8(_mm256_srav_epi16(
                               _mm256_cvtepi8_epi16(a), _mm256_cvtepi8_epi16(b)))
                         : _mm256_cvtepi16_epi8(_mm256_srlv_epi16(
                               _mm256_cvtepu8_epi16(a), _mm256_cvtepu8_epi16(b)));
    } else if constexpr (is_byte && is_xmm && __have_sse4_1) {  //{{{2
        if constexpr (is_signed) {
            const auto aa = __vector_cast<short>(a);
            const auto bb = __vector_cast<short>(b);
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of b is [0, 7]
            // => only the 3 low bits of b are relevant
            // do a =>> 4 where b[2] is set
            auto signbit = aa & 0x8080u;
            a = reinterpret_cast<_T>(_mm_blendv_epi8(
                __vector_cast<__llong>(a),
                __vector_cast<__llong>((((signbit << 5) - signbit) | (aa & 0xf8f8u)) >> 4),
                __vector_cast<__llong>(bb << 5)));
            // do a =>> 2 where b[1] is set
            a = reinterpret_cast<_T>(_mm_blendv_epi8(
                __vector_cast<__llong>(a),
                __vector_cast<__llong>((((signbit << 3) - signbit) | (aa & 0xfcfcu)) >> 2),
                __vector_cast<__llong>(bb << 6)));
            // do a =>> 1 where b[0] is set
            return reinterpret_cast<_T>(
                _mm_blendv_epi8(__vector_cast<__llong>(a),
                                __vector_cast<__llong>(signbit | ((aa & 0xfefeu) >> 1)),
                                __vector_cast<__llong>(bb << 7)));
        } else {
            const auto aa = __vector_cast<ushort>(a);
            const auto bb = __vector_cast<ushort>(b);
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of b is [0, 7]
            // => only the 3 low bits of b are relevant
            // do a =>> 4 where b[2] is set
            a = reinterpret_cast<_T>(_mm_blendv_epi8(
                __vector_cast<__llong>(a), __vector_cast<__llong>((aa >> 4) & 0x0f0fu),
                __vector_cast<__llong>(bb << 5)));
            // do a =>> 2 where b[1] is set
            a = reinterpret_cast<_T>(_mm_blendv_epi8(
                __vector_cast<__llong>(a), __vector_cast<__llong>((aa >> 2) & 0x3f3fu),
                __vector_cast<__llong>(bb << 6)));
            // do a =>> 1 where b[0] is set
            return reinterpret_cast<_T>(_mm_blendv_epi8(
                __vector_cast<__llong>(a), __vector_cast<__llong>((aa >> 1) & 0x7f7fu),
                __vector_cast<__llong>(bb << 7)));
        }
    } else if constexpr (is_byte && is_ymm && __have_avx512bw) {  //{{{2
        return _mm512_cvtepi16_epi8(
            is_signed
                ? _mm512_srav_epi16(_mm512_cvtepi8_epi16(ai), _mm512_cvtepi8_epi16(bi))
                : _mm512_srlv_epi16(_mm512_cvtepu8_epi16(ai), _mm512_cvtepu8_epi16(bi)));
    } else if constexpr (is_byte && is_ymm && __have_avx2) {  //{{{2
        if constexpr (is_signed) {
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of b is [0, 7]
            // => only the 3 low bits of b are relevant
            // do a =<< 4 where b[2] is set
            return convert<y_i08>(
                _mm256_srav_epi32(_mm256_cvtepi8_epi32(__lo128(a)),
                                  _mm256_cvtepi8_epi32(__lo128(b))),
                _mm256_srav_epi32(
                    _mm256_cvtepi8_epi32(_mm_unpackhi_epi64(__lo128(a), __lo128(a))),
                    _mm256_cvtepi8_epi32(_mm_unpackhi_epi64(__lo128(b), __lo128(b)))),
                _mm256_srav_epi32(_mm256_cvtepi8_epi32(__hi128(a)),
                                  _mm256_cvtepi8_epi32(__hi128(b))),
                _mm256_srav_epi32(
                    _mm256_cvtepi8_epi32(_mm_unpackhi_epi64(__hi128(a), __hi128(a))),
                    _mm256_cvtepi8_epi32(_mm_unpackhi_epi64(__hi128(b), __hi128(b)))));
        } else {
            const auto aa = __vector_cast<ushort>(a);
            const auto bb = __vector_cast<ushort>(b);
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of b is [0, 7]
            // => only the 3 low bits of b are relevant
            // do a =>> 4 where b[2] is set
            a = reinterpret_cast<_T>(_mm256_blendv_epi8(
                __vector_cast<__llong>(a), __vector_cast<__llong>((aa >> 4) & 0x0f0fu),
                __vector_cast<__llong>(bb << 5)));
            // do a =>> 2 where b[1] is set
            a = reinterpret_cast<_T>(_mm256_blendv_epi8(
                __vector_cast<__llong>(a), __vector_cast<__llong>((aa >> 2) & 0x3f3fu),
                __vector_cast<__llong>(bb << 6)));
            // do a =>> 1 where b[0] is set
            return reinterpret_cast<_T>(_mm256_blendv_epi8(
                __vector_cast<__llong>(a), __vector_cast<__llong>((aa >> 1) & 0x7f7fu),
                __vector_cast<__llong>(bb << 7)));
        }
    } else if constexpr (is_byte && is_zmm && __have_avx512bw) {  //{{{2
        return concat(__bit_shift_right(__lo256(a), __lo256(b)),
                      __bit_shift_right(__hi256(a), __hi256(b)));
    } else if constexpr (is_word && is_xmm && __have_avx512bw_vl) {  //{{{2
        return is_signed ? _mm_srav_epi16(ai, bi) : _mm_srlv_epi16(ai, bi);
    } else if constexpr (is_word && is_xmm && __have_avx2) {  //{{{2
        return is_signed ? x86::__convert_to<x_i16>(y_i32(_mm256_srav_epi32(
                               __convert_to<y_i32>(a), __convert_to<y_i32>(b))))
                         : x86::__convert_to<x_u16>(y_u32(_mm256_srlv_epi32(
                               __convert_to<y_u32>(a), __convert_to<y_u32>(b))));
    } else if constexpr (is_word && is_xmm && __have_sse4_1) {  //{{{2
        if constexpr (is_signed) {
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of b is [0, 15]
            // => only the 4 low bits of b are relevant
            // shift by 4 and duplicate to high byte
            b = (b << 4) | (b << 12);
            // do a =>> 8 where b[3] is set
            a = _mm_blendv_epi8(a, _mm_srai_epi16(a, 8), b);
            // do a =>> 4 where b[2] is set
            a = _mm_blendv_epi8(a, _mm_srai_epi16(a, 4), b = _mm_add_epi16(b, b));
            // do a =>> 2 where b[1] is set
            a = _mm_blendv_epi8(a, _mm_srai_epi16(a, 2), b = _mm_add_epi16(b, b));
            // do a =>> 1 where b[0] is set
            return _mm_blendv_epi8(a, _mm_srai_epi16(a, 1), _mm_add_epi16(b, b));
        } else {
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of b is [0, 15]
            // => only the 4 low bits of b are relevant
            // shift by 4 and duplicate to high byte
            b = (b << 4) | (b << 12);
            // do a =>> 8 where b[3] is set
            a = _mm_blendv_epi8(a, __vector_cast<__llong>(a >> 8), b);
            // do a =>> 4 where b[2] is set
            a = _mm_blendv_epi8(a, __vector_cast<__llong>(a >> 4), b = _mm_add_epi16(b, b));
            // do a =>> 2 where b[1] is set
            a = _mm_blendv_epi8(a, __vector_cast<__llong>(a >> 2), b = _mm_add_epi16(b, b));
            // do a =>> 1 where b[0] is set
            return _mm_blendv_epi8(a, __vector_cast<__llong>(a >> 1), _mm_add_epi16(b, b));
        }
    } else if constexpr (is_word && is_xmm && __have_sse2) {  //{{{2
        auto &&blend = [](_T a, _T b, _T c) { return (~c & a) | (c & b); };
        if constexpr (is_signed) {
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of b is [0, 15]
            // => only the 4 low bits of b are relevant
            // do a =>> 8 where b[3] is set
            a = blend(a, _mm_srai_epi16(a, 8),
                            _mm_cmpgt_epi16(b, __auto_broadcast(0x00070007)));
            // do a =>> 4 where b[2] is set
            a = blend(a, _mm_srai_epi16(a, 4),
                            _mm_cmpgt_epi16(__and(b, __auto_broadcast(0x00040004)),
                                            _mm_setzero_si128()));
            // do a =>> 2 where b[1] is set
            a = blend(a, _mm_srai_epi16(a, 2),
                            _mm_cmpgt_epi16(__and(b, __auto_broadcast(0x00020002)),
                                            _mm_setzero_si128()));
            // do a =>> 1 where b[0] is set
            return blend(a, _mm_srai_epi16(a, 1),
                               _mm_cmpgt_epi16(__and(b, __auto_broadcast(0x00010001)),
                                               _mm_setzero_si128()));
        } else {
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of b is [0, 15]
            // => only the 4 low bits of b are relevant
            // do a =>> 8 where b[3] is set
            a = blend(a, __vector_cast<__llong>(a >> 8),
                            _mm_cmpgt_epi16(b, __auto_broadcast(0x00070007)));
            // do a =>> 4 where b[2] is set
            a = blend(a, __vector_cast<__llong>(a >> 4),
                            _mm_cmpgt_epi16(__and(b, __auto_broadcast(0x00040004)),
                                            _mm_setzero_si128()));
            // do a =>> 2 where b[1] is set
            a = blend(a, __vector_cast<__llong>(a >> 2),
                            _mm_cmpgt_epi16(__and(b, __auto_broadcast(0x00020002)),
                                            _mm_setzero_si128()));
            // do a =>> 1 where b[0] is set
            return blend(a, __vector_cast<__llong>(a >> 1),
                               _mm_cmpgt_epi16(__and(b, __auto_broadcast(0x00010001)),
                                               _mm_setzero_si128()));
        }
    } else if constexpr (is_word && is_ymm && __have_avx512bw_vl) {  //{{{2
        return is_signed ? _mm256_srav_epi16(ai, bi) : _mm256_srlv_epi16(ai, bi);
    } else if constexpr (is_word && is_ymm && __have_avx2) {  //{{{2
        if constexpr (is_signed) {
            auto lo32 = _mm256_srli_epi32(
                _mm256_srav_epi32(_mm256_unpacklo_epi16(_mm256_setzero_si256(), a),
                                  _mm256_unpacklo_epi16(b, _mm256_setzero_si256())),
                16);
            auto hi32 = _mm256_srli_epi32(
                _mm256_srav_epi32(_mm256_unpackhi_epi16(_mm256_setzero_si256(), a),
                                  _mm256_unpackhi_epi16(b, _mm256_setzero_si256())),
                16);
            return _mm256_packs_epi32(lo32, hi32);
        } else {
            return _mm256_blend_epi16(
                (__vector_cast<uint>(a) & 0xffffu) >> (__vector_cast<uint>(b) & 0xffffu),
                __vector_cast<uint>(a) >> (__vector_cast<uint>(b) >> 16), 0xaa);
        }
    } else if constexpr (is_word && is_zmm && __have_avx512bw) {  //{{{2
        return is_signed ? _mm512_srav_epi16(ai, bi) : _mm512_srlv_epi16(ai, bi);
    } else if constexpr (is_dword && is_xmm && !__have_avx2) {  //{{{2
        if constexpr (is_signed) {
            const auto r0 = _mm_sra_epi32(a, _mm_unpacklo_epi32(b, _mm_setzero_si128()));
            const auto r1 = _mm_sra_epi32(a, _mm_srli_epi64(b, 32));
            const auto r2 = _mm_sra_epi32(a, _mm_unpackhi_epi32(b, _mm_setzero_si128()));
            const auto r3 = _mm_sra_epi32(a, _mm_srli_si128(b, 12));
            if constexpr (__have_sse4_1) {
                return _mm_blend_epi16(_mm_blend_epi16(r1, r0, 0x3),
                                       _mm_blend_epi16(r3, r2, 0x30), 0xf0);
            } else {
                return _mm_unpacklo_epi64(_mm_unpacklo_epi32(r0, _mm_srli_si128(r1, 4)),
                                          _mm_unpackhi_epi32(r2, _mm_srli_si128(r3, 4)));
            }
        } else {
            const auto r0 = _mm_srl_epi32(a, _mm_unpacklo_epi32(b, _mm_setzero_si128()));
            const auto r1 = _mm_srl_epi32(a, _mm_srli_epi64(b, 32));
            const auto r2 = _mm_srl_epi32(a, _mm_unpackhi_epi32(b, _mm_setzero_si128()));
            const auto r3 = _mm_srl_epi32(a, _mm_srli_si128(b, 12));
            if constexpr (__have_sse4_1) {
                return _mm_blend_epi16(_mm_blend_epi16(r1, r0, 0x3),
                                       _mm_blend_epi16(r3, r2, 0x30), 0xf0);
            } else {
                return _mm_unpacklo_epi64(_mm_unpacklo_epi32(r0, _mm_srli_si128(r1, 4)),
                                          _mm_unpackhi_epi32(r2, _mm_srli_si128(r3, 4)));
            }
        }
    }  // }}}2
    return a << b;
}
// }}}1
_GLIBCXX_SIMD_END_NAMESPACE

#endif  // __cplusplus >= 201703L
#endif  // _GLIBCXX_EXPERIMENTAL_SIMD_MATH_H
// vim: foldmethod=marker
