#ifndef _GLIBCXX_EXPERIMENTAL_SIMD_MATH_H
#define _GLIBCXX_EXPERIMENTAL_SIMD_MATH_H

#pragma GCC system_header

#if __cplusplus >= 201703L

_GLIBCXX_SIMD_BEGIN_NAMESPACE
// __divides {{{1
template <class _Tp, class = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC _Tp __divides(_Tp __a, _Tp __b)
{
    using _U = typename Traits::value_type;
    constexpr bool is_byte = sizeof(_U) == 1;
    constexpr bool is_word = sizeof(_U) == 2;
    constexpr bool is_dword = sizeof(_U) == 4;
    constexpr bool is_ymm = sizeof(_Tp) == 32;
    constexpr bool is_xmm = sizeof(_Tp) == 16;

    if constexpr (is_dword && ((is_xmm && __have_avx) || (is_ymm && __have_avx512f))) {
        return convert<_U>(convert<double>(__a) / convert<double>(__b));
    } else if constexpr (is_dword) {  // really better with is_xmm?
        auto __x = __convert_all<__vector_type_t<double, Traits::_S_width / 2>>(__a);
        auto __y = __convert_all<__vector_type_t<double, Traits::_S_width / 2>>(__b);
        return convert<_Tp>(__x[0] / __y[0], __x[1] / __y[1]);
    } else if constexpr (is_word) {
        if constexpr ((is_xmm && __have_avx) || (is_ymm && __have_avx512f)) {
            return convert<_Tp>(convert<float>(__a) / convert<float>(__b));
        } else {
            auto __x = __convert_all<__vector_type_t<float, Traits::_S_width / 2>>(__a);
            auto __y = __convert_all<__vector_type_t<float, Traits::_S_width / 2>>(__b);
            return convert<_Tp>(__x[0] / __y[0], __x[1] / __y[1]);
        }
    } else if constexpr (is_byte && is_xmm && __have_avx512f) {
        return convert<_Tp>(convert<float>(__a) / convert<float>(__b));
    } else if constexpr (is_byte && ((is_xmm && __have_avx) || is_ymm && __have_avx512f)) {
        auto __x = __convert_all<__vector_type_t<float, Traits::_S_width / 2>>(__a);
        auto __y = __convert_all<__vector_type_t<float, Traits::_S_width / 2>>(__b);
        return convert<_Tp>(__x[0] / __y[0], __x[1] / __y[1]);
    } else if constexpr (is_byte) {
        auto __x = __convert_all<__vector_type_t<float, Traits::_S_width / 4>>(__a);
        auto __y = __convert_all<__vector_type_t<float, Traits::_S_width / 4>>(__b);
        return convert<_Tp>(__x[0] / __y[0], __x[1] / __y[1], __x[2] / __y[2], __x[3] / __y[3]);
    } else {
        return __a / __b;
    }
}
// __bit_shift_left{{{1
template <class _Tp, size_t _N>
_GLIBCXX_SIMD_INTRINSIC _SimdWrapper<_Tp, _N> constexpr __bit_shift_left(_SimdWrapper<_Tp, _N> __a, int __b)
{
    static_assert(std::is_integral<_Tp>::value, "__bit_shift_left is only supported for integral types");
    if constexpr (sizeof(_Tp) == 1) {
        // (cf. https://gcc.gnu.org/bugzilla/show_bug.cgi?id=83894)
        if (__builtin_constant_p(__b)) {
            if (__b == 0) {
                return __a;
            } else if (__b == 1) {
                return __a._M_data + __a._M_data;
            } else if (__b > 1 && __b < 8) {
                const _UChar mask = (0xff << __b) & 0xff;
                using _V = decltype(__a);
                using In = typename _V::__intrin_type;
                return reinterpret_cast<In>(__storage_bitcast<ushort>(__a)._M_data << __b) &
                       _V::broadcast(mask).__intrin();
            } else {
                __builtin_unreachable();
            }
        }
        if constexpr (_N == 16 && __have_sse2) {
            if constexpr (__have_avx512bw_vl) {
                return _mm256_cvtepi16_epi8(reinterpret_cast<__m256i>(
                    reinterpret_cast<__vector_type_t<ushort, 16>>(_mm256_cvtepi8_epi16(__a))
                    << __b));
            } else {
                using vshort = __vector_type_t<ushort, 8>;
                const auto mask = ((~vshort() >> 8) << __b) ^ (~vshort() << 8);
                return _To_storage((reinterpret_cast<vshort>(__a._M_data) << __b) & mask);
            }
        } else if constexpr (_N == 32 && __have_avx2) {
            if constexpr(__have_avx512bw) {
                return _mm512_cvtepi16_epi8(reinterpret_cast<__m512i>(
                    reinterpret_cast<__vector_type_t<ushort, 32>>(_mm512_cvtepi8_epi16(__a))
                    << __b));
            } else {
                using vshort = __vector_type_t<ushort, 16>;
                const auto mask = ((~vshort() >> 8) << __b) ^ (~vshort() << 8);
                return _To_storage((reinterpret_cast<vshort>(__a._M_data) << __b) & mask);
            }
        } else if constexpr (_N == 64 && __have_avx512bw) {
            using vshort = __vector_type_t<ushort, 32>;
            const auto mask = ((~vshort() >> 8) << __b) ^ (~vshort() << 8);
            return _To_storage((reinterpret_cast<vshort>(__a._M_data) << __b) & mask);
        } else {
            static_assert(!std::is_same_v<_Tp, _Tp>);
        }
    } else {
        return __a._M_data << __b;
    }
}

template <class _Tp, size_t _N>
_GLIBCXX_SIMD_INTRINSIC _SimdWrapper<_Tp, _N> __bit_shift_left(_SimdWrapper<_Tp, _N> __a, _SimdWrapper<_Tp, _N> __b)
{
    static_assert(std::is_integral<_Tp>::value,
                  "__bit_shift_left is only supported for integral types");
    if constexpr (sizeof(_Tp) == 2 && sizeof(__a) == 16 && !__have_avx2) {
        __vector_type_t<int, 4> shift = __storage_bitcast<int>(__b)._M_data + (0x03f8'03f8 >> 3);
        return multiplies(
            __a,
            _SimdWrapper<_Tp, _N>(
                _mm_cvttps_epi32(reinterpret_cast<__m128>(shift << 23)) |
                (_mm_cvttps_epi32(reinterpret_cast<__m128>(shift >> 16 << 23)) << 16)));
    } else if constexpr (sizeof(_Tp) == 4 && sizeof(__a) == 16 && !__have_avx2) {
        return __storage_bitcast<_Tp>(
            multiplies(__a, _SimdWrapper<_Tp, _N>(_mm_cvttps_epi32(
                              reinterpret_cast<__m128>((__b._M_data << 23) + 0x3f80'0000)))));
    } else if constexpr (sizeof(_Tp) == 8 && sizeof(__a) == 16 && !__have_avx2) {
        const auto __lo = _mm_sll_epi64(__a, __b);
        const auto __hi = _mm_sll_epi64(__a, _mm_unpackhi_epi64(__b, __b));
        if constexpr (__have_sse4_1) {
            return _mm_blend_epi16(__lo, __hi, 0xf0);
        } else {
            // return __make_storage<_LLong>(reinterpret_cast<__vector_type_t<_LLong,
            // 2>>(__lo)[0], reinterpret_cast<__vector_type_t<_LLong, 2>>(__hi)[1]);
            return _To_storage(
                _mm_move_sd(__intrin_bitcast<__m128d>(__hi), __intrin_bitcast<__m128d>(__lo)));
        }
    } else if constexpr (__have_avx512f && sizeof(_Tp) == 8 && _N == 8) {
        return _mm512_sllv_epi64(__a, __b);
    } else if constexpr (__have_avx2 && sizeof(_Tp) == 8 && _N == 4) {
        return _mm256_sllv_epi64(__a, __b);
    } else if constexpr (__have_avx2 && sizeof(_Tp) == 8 && _N == 2) {
        return _mm_sllv_epi64(__a, __b);
    } else if constexpr (__have_avx512f && sizeof(_Tp) == 4 && _N == 16) {
        return _mm512_sllv_epi32(__a, __b);
    } else if constexpr (__have_avx2 && sizeof(_Tp) == 4 && _N == 8) {
        return _mm256_sllv_epi32(__a, __b);
    } else if constexpr (__have_avx2 && sizeof(_Tp) == 4 && _N == 4) {
        return _mm_sllv_epi32(__a, __b);
    } else if constexpr (sizeof(_Tp) == 2) {
        if constexpr (_N == 32 && __have_avx512bw) {
            return _mm512_sllv_epi16(__a, __b);
        } else if constexpr (_N == 16 && __have_avx512bw_vl) {
            return _mm256_sllv_epi16(__a, __b);
        } else if constexpr (_N == 16 && __have_avx512bw) {
            return __lo256(
                _mm512_sllv_epi16(_mm512_castsi256_si512(__a), _mm512_castsi256_si512(__b)));
        } else if constexpr (_N == 16) {
            const auto aa = __vector_bitcast<unsigned>(__a._M_data);
            const auto bb = __vector_bitcast<unsigned>(__b._M_data);
            return _mm256_blend_epi16(__auto_bitcast(aa << (bb & 0x0000ffffu)),
                                      __auto_bitcast((aa & 0xffff0000u) << (bb >> 16)), 0xaa);
        } else if constexpr (_N == 8 && __have_avx512bw_vl) {
            return _mm_sllv_epi16(__a, __b);
        } else if constexpr (_N == 8 && __have_avx512bw) {
            return _mm512_sllv_epi16(_mm512_castsi128_si512(__a),
                                     _mm512_castsi128_si512(__b));
        } else if constexpr (_N == 8) {
            const auto aa = __vector_bitcast<unsigned>(__a._M_data);
            const auto bb = __vector_bitcast<unsigned>(__b._M_data);
            return _mm_blend_epi16(__auto_bitcast(aa << (bb & 0x0000ffffu)),
                                   __auto_bitcast((aa & 0xffff0000u) << (bb >> 16)), 0xaa);
        } else {
            __assert_unreachable<_Tp>();
        }
    } else if constexpr (sizeof(_Tp) == 1) {
        if constexpr (_N == 64 && __have_avx512bw) {
            return concat(_mm512_cvtepi16_epi8(_mm512_sllv_epi16(
                              _mm512_cvtepu8_epi16(__lo256(__vector_bitcast<_LLong>(__a))),
                              _mm512_cvtepu8_epi16(__lo256(__vector_bitcast<_LLong>(__b))))),
                          _mm512_cvtepi16_epi8(_mm512_sllv_epi16(
                              _mm512_cvtepu8_epi16(__hi256(__vector_bitcast<_LLong>(__a))),
                              _mm512_cvtepu8_epi16(__hi256(__vector_bitcast<_LLong>(__b))))));
        } else if constexpr (_N == 32 && __have_avx512bw) {
            return _mm512_cvtepi16_epi8(
                _mm512_sllv_epi16(_mm512_cvtepu8_epi16(__a), _mm512_cvtepu8_epi16(__b)));
        } else if constexpr (_N == 16 && __have_avx512bw_vl) {
            return _mm256_cvtepi16_epi8(
                _mm256_sllv_epi16(_mm256_cvtepu8_epi16(__a), _mm256_cvtepu8_epi16(__b)));
        } else if constexpr (_N == 16 && __have_avx512bw) {
            return __lo128(_mm512_cvtepi16_epi8(
                _mm512_sllv_epi16(_mm512_cvtepu8_epi16(_mm512_castsi256_si512(__a)),
                                  _mm512_cvtepu8_epi16(_mm512_castsi256_si512(__b)))));
        } else {
            auto mask_from_bit = [](__vector_type_t<_Tp, _N> __x, int bit) {
                auto __y = __vector_bitcast<short>(__x) << bit;
                if constexpr (__have_sse4_1) {
                    return __to_intrin(__y);
                } else {
                    return __to_intrin(__vector_bitcast<_SChar>(__y) < 0);
                }
            };
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand. left
            // => valid input range for each element of __b is [0, 7]
            // => only the 3 low bits of __b are relevant
            // do __a =<< 4 where __b[2] is set
            auto a4 = __vector_bitcast<_UChar>(__vector_bitcast<short>(__a._M_data) << 4);
            if constexpr (std::is_unsigned_v<_Tp>) {
                // shift into or over the sign bit is UB => never spills into a neighbor
                a4 &= 0xf0u;
            }
            __a = __blend(mask_from_bit(__b, 5), __a, __to_intrin(a4));
            // do __a =<< 2 where __b[1] is set
            // shift into or over the sign bit is UB => never spills into a neighbor
            const auto a2 = std::is_signed_v<_Tp> ? __to_intrin(__vector_bitcast<short>(__a._M_data) << 2)
                                                : __to_intrin(__a._M_data << 2);
            __a = __blend(mask_from_bit(__b, 6), __a, a2);
            // do __a =<< 1 where __b[0] is set
            return __blend(mask_from_bit(__b, 7), __a, __to_intrin(__a._M_data + __a._M_data));
        }
    } else {
        return __a._M_data << __b._M_data;
    }
}

// }}}
// __bit_shift_right{{{1
template <class _Tp, class Traits = __vector_traits<_Tp>> _Tp __bit_shift_right(_Tp __a, _Tp __b)
{
    using _U = typename Traits::value_type;
    constexpr bool is_byte = sizeof(_U) == 1;
    constexpr bool is_word = sizeof(_U) == 2;
    constexpr bool is_dword = sizeof(_U) == 4;
    constexpr bool is_signed = std::is_signed_v<_U>;
    constexpr bool is_zmm = sizeof(_Tp) == 64;
    constexpr bool is_ymm = sizeof(_Tp) == 32;
    constexpr bool is_xmm = sizeof(_Tp) == 16;

    const auto ai = __to_intrin(__a);
    const auto bi = __to_intrin(__b);

    if constexpr (is_byte && is_xmm && __have_avx512bw_vl) {  //{{{2
        return is_signed ? _mm256_cvtepi16_epi8(_mm256_srav_epi16(
                               _mm256_cvtepi8_epi16(__a), _mm256_cvtepi8_epi16(__b)))
                         : _mm256_cvtepi16_epi8(_mm256_srlv_epi16(
                               _mm256_cvtepu8_epi16(__a), _mm256_cvtepu8_epi16(__b)));
    } else if constexpr (is_byte && is_xmm && __have_sse4_1) {  //{{{2
        if constexpr (is_signed) {
            const auto aa = __vector_bitcast<short>(__a);
            const auto bb = __vector_bitcast<short>(__b);
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of __b is [0, 7]
            // => only the 3 low bits of __b are relevant
            // do __a =>> 4 where __b[2] is set
            auto signbit = aa & 0x8080u;
            __a = reinterpret_cast<_Tp>(_mm_blendv_epi8(
                __vector_bitcast<_LLong>(__a),
                __vector_bitcast<_LLong>((((signbit << 5) - signbit) | (aa & 0xf8f8u)) >> 4),
                __vector_bitcast<_LLong>(bb << 5)));
            // do __a =>> 2 where __b[1] is set
            __a = reinterpret_cast<_Tp>(_mm_blendv_epi8(
                __vector_bitcast<_LLong>(__a),
                __vector_bitcast<_LLong>((((signbit << 3) - signbit) | (aa & 0xfcfcu)) >> 2),
                __vector_bitcast<_LLong>(bb << 6)));
            // do __a =>> 1 where __b[0] is set
            return reinterpret_cast<_Tp>(
                _mm_blendv_epi8(__vector_bitcast<_LLong>(__a),
                                __vector_bitcast<_LLong>(signbit | ((aa & 0xfefeu) >> 1)),
                                __vector_bitcast<_LLong>(bb << 7)));
        } else {
            const auto aa = __vector_bitcast<ushort>(__a);
            const auto bb = __vector_bitcast<ushort>(__b);
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of __b is [0, 7]
            // => only the 3 low bits of __b are relevant
            // do __a =>> 4 where __b[2] is set
            __a = reinterpret_cast<_Tp>(_mm_blendv_epi8(
                __vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>((aa >> 4) & 0x0f0fu),
                __vector_bitcast<_LLong>(bb << 5)));
            // do __a =>> 2 where __b[1] is set
            __a = reinterpret_cast<_Tp>(_mm_blendv_epi8(
                __vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>((aa >> 2) & 0x3f3fu),
                __vector_bitcast<_LLong>(bb << 6)));
            // do __a =>> 1 where __b[0] is set
            return reinterpret_cast<_Tp>(_mm_blendv_epi8(
                __vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>((aa >> 1) & 0x7f7fu),
                __vector_bitcast<_LLong>(bb << 7)));
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
            // => valid input range for each element of __b is [0, 7]
            // => only the 3 low bits of __b are relevant
            // do __a =<< 4 where __b[2] is set
            return __vector_convert<_Tp>(
                __vector_bitcast<int>(_mm256_srav_epi32(_mm256_cvtepi8_epi32(__lo128(ai)),
                                                     _mm256_cvtepi8_epi32(__lo128(bi)))),
                __vector_bitcast<int>(_mm256_srav_epi32(
                    _mm256_cvtepi8_epi32(_mm_unpackhi_epi64(__lo128(ai), __lo128(ai))),
                    _mm256_cvtepi8_epi32(_mm_unpackhi_epi64(__lo128(bi), __lo128(bi))))),
                __vector_bitcast<int>(_mm256_srav_epi32(_mm256_cvtepi8_epi32(__hi128(ai)),
                                                     _mm256_cvtepi8_epi32(__hi128(bi)))),
                __vector_bitcast<int>(_mm256_srav_epi32(
                    _mm256_cvtepi8_epi32(_mm_unpackhi_epi64(__hi128(ai), __hi128(ai))),
                    _mm256_cvtepi8_epi32(_mm_unpackhi_epi64(__hi128(bi), __hi128(bi))))));
        } else {
            const auto aa = __vector_bitcast<ushort>(__a);
            const auto bb = __vector_bitcast<ushort>(__b);
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of __b is [0, 7]
            // => only the 3 low bits of __b are relevant
            // do __a =>> 4 where __b[2] is set
            __a = reinterpret_cast<_Tp>(_mm256_blendv_epi8(
                __vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>((aa >> 4) & 0x0f0fu),
                __vector_bitcast<_LLong>(bb << 5)));
            // do __a =>> 2 where __b[1] is set
            __a = reinterpret_cast<_Tp>(_mm256_blendv_epi8(
                __vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>((aa >> 2) & 0x3f3fu),
                __vector_bitcast<_LLong>(bb << 6)));
            // do __a =>> 1 where __b[0] is set
            return reinterpret_cast<_Tp>(_mm256_blendv_epi8(
                __vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>((aa >> 1) & 0x7f7fu),
                __vector_bitcast<_LLong>(bb << 7)));
        }
    } else if constexpr (is_byte && is_zmm && __have_avx512bw) {  //{{{2
        return concat(__bit_shift_right(__lo256(__a), __lo256(__b)),
                      __bit_shift_right(__hi256(__a), __hi256(__b)));
    } else if constexpr (is_word && is_xmm && __have_avx512bw_vl) {  //{{{2
        return is_signed ? _mm_srav_epi16(ai, bi) : _mm_srlv_epi16(ai, bi);
    } else if constexpr (is_word && is_xmm && __have_avx2) {  //{{{2
        return is_signed ? __vector_convert<short>(__vector_convert<int>(__a) >>
                                                   __vector_convert<int>(__b))
                         : __vector_convert<_UShort>(__vector_convert<_UInt>(__a) >>
                                                      __vector_convert<_UInt>(__b));
    } else if constexpr (is_word && is_xmm && __have_sse4_1) {  //{{{2
        if constexpr (is_signed) {
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of __b is [0, 15]
            // => only the 4 low bits of __b are relevant
            // shift by 4 and duplicate to high byte
            __b = (__b << 4) | (__b << 12);
            // do __a =>> 8 where __b[3] is set
            __a = _mm_blendv_epi8(__a, _mm_srai_epi16(__a, 8), __b);
            // do __a =>> 4 where __b[2] is set
            __a = _mm_blendv_epi8(__a, _mm_srai_epi16(__a, 4), __b = _mm_add_epi16(__b, __b));
            // do __a =>> 2 where __b[1] is set
            __a = _mm_blendv_epi8(__a, _mm_srai_epi16(__a, 2), __b = _mm_add_epi16(__b, __b));
            // do __a =>> 1 where __b[0] is set
            return _mm_blendv_epi8(__a, _mm_srai_epi16(__a, 1), _mm_add_epi16(__b, __b));
        } else {
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of __b is [0, 15]
            // => only the 4 low bits of __b are relevant
            // shift by 4 and duplicate to high byte
            __b = (__b << 4) | (__b << 12);
            // do __a =>> 8 where __b[3] is set
            __a = _mm_blendv_epi8(__a, __vector_bitcast<_LLong>(__a >> 8), __b);
            // do __a =>> 4 where __b[2] is set
            __a = _mm_blendv_epi8(__a, __vector_bitcast<_LLong>(__a >> 4), __b = _mm_add_epi16(__b, __b));
            // do __a =>> 2 where __b[1] is set
            __a = _mm_blendv_epi8(__a, __vector_bitcast<_LLong>(__a >> 2), __b = _mm_add_epi16(__b, __b));
            // do __a =>> 1 where __b[0] is set
            return _mm_blendv_epi8(__a, __vector_bitcast<_LLong>(__a >> 1), _mm_add_epi16(__b, __b));
        }
    } else if constexpr (is_word && is_xmm && __have_sse2) {  //{{{2
        auto &&blend = [](_Tp __a, _Tp __b, _Tp __c) { return (~__c & __a) | (__c & __b); };
        if constexpr (is_signed) {
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of __b is [0, 15]
            // => only the 4 low bits of __b are relevant
            // do __a =>> 8 where __b[3] is set
            __a = blend(__a, _mm_srai_epi16(__a, 8),
                            _mm_cmpgt_epi16(__b, __auto_broadcast(0x00070007)));
            // do __a =>> 4 where __b[2] is set
            __a = blend(__a, _mm_srai_epi16(__a, 4),
                            _mm_cmpgt_epi16(__and(__b, __auto_broadcast(0x00040004)),
                                            _mm_setzero_si128()));
            // do __a =>> 2 where __b[1] is set
            __a = blend(__a, _mm_srai_epi16(__a, 2),
                            _mm_cmpgt_epi16(__and(__b, __auto_broadcast(0x00020002)),
                                            _mm_setzero_si128()));
            // do __a =>> 1 where __b[0] is set
            return blend(__a, _mm_srai_epi16(__a, 1),
                               _mm_cmpgt_epi16(__and(__b, __auto_broadcast(0x00010001)),
                                               _mm_setzero_si128()));
        } else {
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of __b is [0, 15]
            // => only the 4 low bits of __b are relevant
            // do __a =>> 8 where __b[3] is set
            __a = blend(__a, __vector_bitcast<_LLong>(__a >> 8),
                            _mm_cmpgt_epi16(__b, __auto_broadcast(0x00070007)));
            // do __a =>> 4 where __b[2] is set
            __a = blend(__a, __vector_bitcast<_LLong>(__a >> 4),
                            _mm_cmpgt_epi16(__and(__b, __auto_broadcast(0x00040004)),
                                            _mm_setzero_si128()));
            // do __a =>> 2 where __b[1] is set
            __a = blend(__a, __vector_bitcast<_LLong>(__a >> 2),
                            _mm_cmpgt_epi16(__and(__b, __auto_broadcast(0x00020002)),
                                            _mm_setzero_si128()));
            // do __a =>> 1 where __b[0] is set
            return blend(__a, __vector_bitcast<_LLong>(__a >> 1),
                               _mm_cmpgt_epi16(__and(__b, __auto_broadcast(0x00010001)),
                                               _mm_setzero_si128()));
        }
    } else if constexpr (is_word && is_ymm && __have_avx512bw_vl) {  //{{{2
        return is_signed ? _mm256_srav_epi16(ai, bi) : _mm256_srlv_epi16(ai, bi);
    } else if constexpr (is_word && is_ymm && __have_avx2) {  //{{{2
        if constexpr (is_signed) {
            auto lo32 = _mm256_srli_epi32(
                _mm256_srav_epi32(_mm256_unpacklo_epi16(_mm256_setzero_si256(), __a),
                                  _mm256_unpacklo_epi16(__b, _mm256_setzero_si256())),
                16);
            auto hi32 = _mm256_srli_epi32(
                _mm256_srav_epi32(_mm256_unpackhi_epi16(_mm256_setzero_si256(), __a),
                                  _mm256_unpackhi_epi16(__b, _mm256_setzero_si256())),
                16);
            return _mm256_packs_epi32(lo32, hi32);
        } else {
            return _mm256_blend_epi16(
                (__vector_bitcast<_UInt>(__a) & 0xffffu) >> (__vector_bitcast<_UInt>(__b) & 0xffffu),
                __vector_bitcast<_UInt>(__a) >> (__vector_bitcast<_UInt>(__b) >> 16), 0xaa);
        }
    } else if constexpr (is_word && is_zmm && __have_avx512bw) {  //{{{2
        return is_signed ? _mm512_srav_epi16(ai, bi) : _mm512_srlv_epi16(ai, bi);
    } else if constexpr (is_dword && is_xmm && !__have_avx2) {  //{{{2
        if constexpr (is_signed) {
            const auto r0 = _mm_sra_epi32(__a, _mm_unpacklo_epi32(__b, _mm_setzero_si128()));
            const auto r1 = _mm_sra_epi32(__a, _mm_srli_epi64(__b, 32));
            const auto r2 = _mm_sra_epi32(__a, _mm_unpackhi_epi32(__b, _mm_setzero_si128()));
            const auto r3 = _mm_sra_epi32(__a, _mm_srli_si128(__b, 12));
            if constexpr (__have_sse4_1) {
                return _mm_blend_epi16(_mm_blend_epi16(r1, r0, 0x3),
                                       _mm_blend_epi16(r3, r2, 0x30), 0xf0);
            } else {
                return _mm_unpacklo_epi64(_mm_unpacklo_epi32(r0, _mm_srli_si128(r1, 4)),
                                          _mm_unpackhi_epi32(r2, _mm_srli_si128(r3, 4)));
            }
        } else {
            const auto r0 = _mm_srl_epi32(__a, _mm_unpacklo_epi32(__b, _mm_setzero_si128()));
            const auto r1 = _mm_srl_epi32(__a, _mm_srli_epi64(__b, 32));
            const auto r2 = _mm_srl_epi32(__a, _mm_unpackhi_epi32(__b, _mm_setzero_si128()));
            const auto r3 = _mm_srl_epi32(__a, _mm_srli_si128(__b, 12));
            if constexpr (__have_sse4_1) {
                return _mm_blend_epi16(_mm_blend_epi16(r1, r0, 0x3),
                                       _mm_blend_epi16(r3, r2, 0x30), 0xf0);
            } else {
                return _mm_unpacklo_epi64(_mm_unpacklo_epi32(r0, _mm_srli_si128(r1, 4)),
                                          _mm_unpackhi_epi32(r2, _mm_srli_si128(r3, 4)));
            }
        }
    }  // }}}2
    return __a << __b;
}
// }}}1
_GLIBCXX_SIMD_END_NAMESPACE

#endif  // __cplusplus >= 201703L
#endif  // _GLIBCXX_EXPERIMENTAL_SIMD_MATH_H
// vim: foldmethod=marker
