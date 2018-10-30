Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// divides {{{1
template <class T, class = builtin_traits<T>>
Vc_INTRINSIC T divides(T a, T b)
{
    using U = typename Traits::value_type;
    constexpr bool is_byte = sizeof(U) == 1;
    constexpr bool is_word = sizeof(U) == 2;
    constexpr bool is_dword = sizeof(U) == 4;
    constexpr bool is_ymm = sizeof(T) == 32;
    constexpr bool is_xmm = sizeof(T) == 16;

    if constexpr (is_dword && ((is_xmm && have_avx) || (is_ymm && have_avx512f))) {
        return convert<U>(convert<double>(a) / convert<double>(b));
    } else if constexpr (is_dword) {  // really better with is_xmm?
        auto x = convert_all<builtin_type_t<double, Traits::width / 2>>(a);
        auto y = convert_all<builtin_type_t<double, Traits::width / 2>>(b);
        return convert<T>(x[0] / y[0], x[1] / y[1]);
    } else if constexpr (is_word) {
        if constexpr ((is_xmm && have_avx) || (is_ymm && have_avx512f)) {
            return convert<T>(convert<float>(a) / convert<float>(b));
        } else {
            auto x = convert_all<builtin_type_t<float, Traits::width / 2>>(a);
            auto y = convert_all<builtin_type_t<float, Traits::width / 2>>(b);
            return convert<T>(x[0] / y[0], x[1] / y[1]);
        }
    } else if constexpr (is_byte && is_xmm && have_avx512f) {
        return convert<T>(convert<float>(a) / convert<float>(b));
    } else if constexpr (is_byte && ((is_xmm && have_avx) || is_ymm && have_avx512f)) {
        auto x = convert_all<builtin_type_t<float, Traits::width / 2>>(a);
        auto y = convert_all<builtin_type_t<float, Traits::width / 2>>(b);
        return convert<T>(x[0] / y[0], x[1] / y[1]);
    } else if constexpr (is_byte) {
        auto x = convert_all<builtin_type_t<float, Traits::width / 4>>(a);
        auto y = convert_all<builtin_type_t<float, Traits::width / 4>>(b);
        return convert<T>(x[0] / y[0], x[1] / y[1], x[2] / y[2], x[3] / y[3]);
    } else {
        return a / b;
    }
}
// bit_shift_left{{{1
template <class T, size_t N>
Vc_INTRINSIC Storage<T, N> constexpr bit_shift_left(Storage<T, N> a, int b)
{
    static_assert(std::is_integral<T>::value, "bit_shift_left is only supported for integral types");
    if constexpr (sizeof(T) == 1) {
        // (cf. https://gcc.gnu.org/bugzilla/show_bug.cgi?id=83894)
        if (__builtin_constant_p(b)) {
            if (b == 0) {
                return a;
            } else if (b == 1) {
                return a.d + a.d;
            } else if (b > 1 && b < 8) {
                const uchar mask = (0xff << b) & 0xff;
                using V = decltype(a);
                using In = typename V::intrin_type;
                return reinterpret_cast<In>(storage_bitcast<ushort>(a).d << b) &
                       V::broadcast(mask).intrin();
            } else {
                return detail::warn_ub(a);
            }
        }
        if constexpr (N == 16 && have_sse2) {
            if constexpr (have_avx512bw_vl) {
                return _mm256_cvtepi16_epi8(reinterpret_cast<__m256i>(
                    reinterpret_cast<builtin_type_t<ushort, 16>>(_mm256_cvtepi8_epi16(a))
                    << b));
            } else {
                using vshort = builtin_type_t<ushort, 8>;
                const auto mask = ((~vshort() >> 8) << b) ^ (~vshort() << 8);
                return detail::to_storage((reinterpret_cast<vshort>(a.d) << b) & mask);
            }
        } else if constexpr (N == 32 && have_avx2) {
            if constexpr(have_avx512bw) {
                return _mm512_cvtepi16_epi8(reinterpret_cast<__m512i>(
                    reinterpret_cast<builtin_type_t<ushort, 32>>(_mm512_cvtepi8_epi16(a))
                    << b));
            } else {
                using vshort = builtin_type_t<ushort, 16>;
                const auto mask = ((~vshort() >> 8) << b) ^ (~vshort() << 8);
                return detail::to_storage((reinterpret_cast<vshort>(a.d) << b) & mask);
            }
        } else if constexpr (N == 64 && have_avx512bw) {
            using vshort = builtin_type_t<ushort, 32>;
            const auto mask = ((~vshort() >> 8) << b) ^ (~vshort() << 8);
            return detail::to_storage((reinterpret_cast<vshort>(a.d) << b) & mask);
        } else {
            static_assert(!std::is_same_v<T, T>);
        }
    } else {
        return a.d << b;
    }
}

template <class T, size_t N>
Vc_INTRINSIC Storage<T, N> bit_shift_left(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value,
                  "bit_shift_left is only supported for integral types");
    if constexpr (sizeof(T) == 2 && sizeof(a) == 16 && !have_avx2) {
        builtin_type_t<int, 4> shift = storage_bitcast<int>(b).d + (0x03f8'03f8 >> 3);
        return multiplies(
            a,
            Storage<T, N>(
                _mm_cvttps_epi32(reinterpret_cast<__m128>(shift << 23)) |
                (_mm_cvttps_epi32(reinterpret_cast<__m128>(shift >> 16 << 23)) << 16)));
    } else if constexpr (sizeof(T) == 4 && sizeof(a) == 16 && !have_avx2) {
        return storage_bitcast<T>(
            multiplies(a, Storage<T, N>(_mm_cvttps_epi32(
                              reinterpret_cast<__m128>((b.d << 23) + 0x3f80'0000)))));
    } else if constexpr (sizeof(T) == 8 && sizeof(a) == 16 && !have_avx2) {
        const auto lo = _mm_sll_epi64(a, b);
        const auto hi = _mm_sll_epi64(a, _mm_unpackhi_epi64(b, b));
#ifdef Vc_HAVE_SSE4_1
        return _mm_blend_epi16(lo, hi, 0xf0);
#else
        // return make_storage<llong>(reinterpret_cast<builtin_type_t<llong, 2>>(lo)[0],
        // reinterpret_cast<builtin_type_t<llong, 2>>(hi)[1]);
        return to_storage(
            _mm_move_sd(intrin_cast<__m128d>(hi), intrin_cast<__m128d>(lo)));
#endif
    } else if constexpr (have_avx512f && sizeof(T) == 8 && N == 8) {
        return _mm512_sllv_epi64(a, b);
    } else if constexpr (have_avx2 && sizeof(T) == 8 && N == 4) {
        return _mm256_sllv_epi64(a, b);
    } else if constexpr (have_avx2 && sizeof(T) == 8 && N == 2) {
        return _mm_sllv_epi64(a, b);
    } else if constexpr (have_avx512f && sizeof(T) == 4 && N == 16) {
        return _mm512_sllv_epi32(a, b);
    } else if constexpr (have_avx2 && sizeof(T) == 4 && N == 8) {
        return _mm256_sllv_epi32(a, b);
    } else if constexpr (have_avx2 && sizeof(T) == 4 && N == 4) {
        return _mm_sllv_epi32(a, b);
    } else if constexpr (sizeof(T) == 2) {
        if constexpr (N == 32 && have_avx512bw) {
            return _mm512_sllv_epi16(a, b);
        } else if constexpr (N == 16 && have_avx512bw_vl) {
            return _mm256_sllv_epi16(a, b);
        } else if constexpr (N == 16 && have_avx512bw) {
            return lo256(
                _mm512_sllv_epi16(_mm512_castsi256_si512(a), _mm512_castsi256_si512(b)));
        } else if constexpr (N == 16) {
            const auto aa = builtin_cast<unsigned>(a.d);
            const auto bb = builtin_cast<unsigned>(b.d);
            return _mm256_blend_epi16(auto_cast(aa << (bb & 0x0000ffffu)),
                                      auto_cast((aa & 0xffff0000u) << (bb >> 16)), 0xaa);
        } else if constexpr (N == 8 && have_avx512bw_vl) {
            return _mm_sllv_epi16(a, b);
        } else if constexpr (N == 8 && have_avx512bw) {
            return _mm512_sllv_epi16(_mm512_castsi128_si512(a),
                                     _mm512_castsi128_si512(b));
        } else if constexpr (N == 8) {
            const auto aa = builtin_cast<unsigned>(a.d);
            const auto bb = builtin_cast<unsigned>(b.d);
            return _mm_blend_epi16(auto_cast(aa << (bb & 0x0000ffffu)),
                                   auto_cast((aa & 0xffff0000u) << (bb >> 16)), 0xaa);
        } else {
            detail::assert_unreachable<T>();
        }
    } else if constexpr (sizeof(T) == 1) {
        if constexpr (N == 64 && have_avx512bw) {
            return concat(_mm512_cvtepi16_epi8(_mm512_sllv_epi16(
                              _mm512_cvtepu8_epi16(lo256(builtin_cast<llong>(a))),
                              _mm512_cvtepu8_epi16(lo256(builtin_cast<llong>(b))))),
                          _mm512_cvtepi16_epi8(_mm512_sllv_epi16(
                              _mm512_cvtepu8_epi16(hi256(builtin_cast<llong>(a))),
                              _mm512_cvtepu8_epi16(hi256(builtin_cast<llong>(b))))));
        } else if constexpr (N == 32 && have_avx512bw) {
            return _mm512_cvtepi16_epi8(
                _mm512_sllv_epi16(_mm512_cvtepu8_epi16(a), _mm512_cvtepu8_epi16(b)));
        } else if constexpr (N == 16 && have_avx512bw_vl) {
            return _mm256_cvtepi16_epi8(
                _mm256_sllv_epi16(_mm256_cvtepu8_epi16(a), _mm256_cvtepu8_epi16(b)));
        } else if constexpr (N == 16 && have_avx512bw) {
            return lo128(_mm512_cvtepi16_epi8(
                _mm512_sllv_epi16(_mm512_cvtepu8_epi16(_mm512_castsi256_si512(a)),
                                  _mm512_cvtepu8_epi16(_mm512_castsi256_si512(b)))));
        } else {
            auto mask_from_bit = [](builtin_type_t<T, N> x, int bit) {
                auto y = builtin_cast<short>(x) << bit;
                if constexpr (have_sse4_1) {
                    return to_intrin(y);
                } else {
                    return to_intrin(builtin_cast<schar>(y) < 0);
                }
            };
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand. left
            // => valid input range for each element of b is [0, 7]
            // => only the 3 low bits of b are relevant
            // do a =<< 4 where b[2] is set
            auto a4 = builtin_cast<uchar>(builtin_cast<short>(a.d) << 4);
            if constexpr (std::is_unsigned_v<T>) {
                // shift into or over the sign bit is UB => never spills into a neighbor
                a4 &= 0xf0u;
            }
            a = x86::blend(mask_from_bit(b, 5), a, to_intrin(a4));
            // do a =<< 2 where b[1] is set
            // shift into or over the sign bit is UB => never spills into a neighbor
            const auto a2 = std::is_signed_v<T> ? to_intrin(builtin_cast<short>(a.d) << 2)
                                                : to_intrin(a.d << 2);
            a = x86::blend(mask_from_bit(b, 6), a, a2);
            // do a =<< 1 where b[0] is set
            return x86::blend(mask_from_bit(b, 7), a, to_intrin(a.d + a.d));
        }
    } else {
        return a.d << b.d;
    }
}

// }}}
// bit_shift_right{{{1
template <class T, class Traits = builtin_traits<T>> T bit_shift_right(T a, T b)
{
    using U = typename Traits::value_type;
    constexpr bool is_byte = sizeof(U) == 1;
    constexpr bool is_word = sizeof(U) == 2;
    constexpr bool is_dword = sizeof(U) == 4;
    constexpr bool is_signed = std::is_signed_v<U>;
    constexpr bool is_zmm = sizeof(T) == 64;
    constexpr bool is_ymm = sizeof(T) == 32;
    constexpr bool is_xmm = sizeof(T) == 16;

    const auto ai = to_intrin(a);
    const auto bi = to_intrin(b);

    if constexpr (is_byte && is_xmm && have_avx512bw_vl) {  //{{{2
        return is_signed ? _mm256_cvtepi16_epi8(_mm256_srav_epi16(
                               _mm256_cvtepi8_epi16(a), _mm256_cvtepi8_epi16(b)))
                         : _mm256_cvtepi16_epi8(_mm256_srlv_epi16(
                               _mm256_cvtepu8_epi16(a), _mm256_cvtepu8_epi16(b)));
    } else if constexpr (is_byte && is_xmm && have_sse4_1) {  //{{{2
        if constexpr (is_signed) {
            const auto aa = builtin_cast<short>(a);
            const auto bb = builtin_cast<short>(b);
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of b is [0, 7]
            // => only the 3 low bits of b are relevant
            // do a =>> 4 where b[2] is set
            auto signbit = aa & 0x8080u;
            a = reinterpret_cast<T>(_mm_blendv_epi8(
                builtin_cast<llong>(a),
                builtin_cast<llong>((((signbit << 5) - signbit) | (aa & 0xf8f8u)) >> 4),
                builtin_cast<llong>(bb << 5)));
            // do a =>> 2 where b[1] is set
            a = reinterpret_cast<T>(_mm_blendv_epi8(
                builtin_cast<llong>(a),
                builtin_cast<llong>((((signbit << 3) - signbit) | (aa & 0xfcfcu)) >> 2),
                builtin_cast<llong>(bb << 6)));
            // do a =>> 1 where b[0] is set
            return reinterpret_cast<T>(
                _mm_blendv_epi8(builtin_cast<llong>(a),
                                builtin_cast<llong>(signbit | ((aa & 0xfefeu) >> 1)),
                                builtin_cast<llong>(bb << 7)));
        } else {
            const auto aa = builtin_cast<ushort>(a);
            const auto bb = builtin_cast<ushort>(b);
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of b is [0, 7]
            // => only the 3 low bits of b are relevant
            // do a =>> 4 where b[2] is set
            a = reinterpret_cast<T>(_mm_blendv_epi8(
                builtin_cast<llong>(a), builtin_cast<llong>((aa >> 4) & 0x0f0fu),
                builtin_cast<llong>(bb << 5)));
            // do a =>> 2 where b[1] is set
            a = reinterpret_cast<T>(_mm_blendv_epi8(
                builtin_cast<llong>(a), builtin_cast<llong>((aa >> 2) & 0x3f3fu),
                builtin_cast<llong>(bb << 6)));
            // do a =>> 1 where b[0] is set
            return reinterpret_cast<T>(_mm_blendv_epi8(
                builtin_cast<llong>(a), builtin_cast<llong>((aa >> 1) & 0x7f7fu),
                builtin_cast<llong>(bb << 7)));
        }
    } else if constexpr (is_byte && is_ymm && have_avx512bw) {  //{{{2
        return _mm512_cvtepi16_epi8(
            is_signed
                ? _mm512_srav_epi16(_mm512_cvtepi8_epi16(ai), _mm512_cvtepi8_epi16(bi))
                : _mm512_srlv_epi16(_mm512_cvtepu8_epi16(ai), _mm512_cvtepu8_epi16(bi)));
    } else if constexpr (is_byte && is_ymm && have_avx2) {  //{{{2
        if constexpr (is_signed) {
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of b is [0, 7]
            // => only the 3 low bits of b are relevant
            // do a =<< 4 where b[2] is set
            return convert<y_i08>(
                _mm256_srav_epi32(_mm256_cvtepi8_epi32(lo128(a)),
                                  _mm256_cvtepi8_epi32(lo128(b))),
                _mm256_srav_epi32(
                    _mm256_cvtepi8_epi32(_mm_unpackhi_epi64(lo128(a), lo128(a))),
                    _mm256_cvtepi8_epi32(_mm_unpackhi_epi64(lo128(b), lo128(b)))),
                _mm256_srav_epi32(_mm256_cvtepi8_epi32(hi128(a)),
                                  _mm256_cvtepi8_epi32(hi128(b))),
                _mm256_srav_epi32(
                    _mm256_cvtepi8_epi32(_mm_unpackhi_epi64(hi128(a), hi128(a))),
                    _mm256_cvtepi8_epi32(_mm_unpackhi_epi64(hi128(b), hi128(b)))));
        } else {
            const auto aa = builtin_cast<ushort>(a);
            const auto bb = builtin_cast<ushort>(b);
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of b is [0, 7]
            // => only the 3 low bits of b are relevant
            // do a =>> 4 where b[2] is set
            a = reinterpret_cast<T>(_mm256_blendv_epi8(
                builtin_cast<llong>(a), builtin_cast<llong>((aa >> 4) & 0x0f0fu),
                builtin_cast<llong>(bb << 5)));
            // do a =>> 2 where b[1] is set
            a = reinterpret_cast<T>(_mm256_blendv_epi8(
                builtin_cast<llong>(a), builtin_cast<llong>((aa >> 2) & 0x3f3fu),
                builtin_cast<llong>(bb << 6)));
            // do a =>> 1 where b[0] is set
            return reinterpret_cast<T>(_mm256_blendv_epi8(
                builtin_cast<llong>(a), builtin_cast<llong>((aa >> 1) & 0x7f7fu),
                builtin_cast<llong>(bb << 7)));
        }
    } else if constexpr (is_byte && is_zmm && have_avx512bw) {  //{{{2
        return concat(bit_shift_right(lo256(a), lo256(b)),
                      bit_shift_right(hi256(a), hi256(b)));
    } else if constexpr (is_word && is_xmm && have_avx512bw_vl) {  //{{{2
        return is_signed ? _mm_srav_epi16(ai, bi) : _mm_srlv_epi16(ai, bi);
    } else if constexpr (is_word && is_xmm && have_avx2) {  //{{{2
        return is_signed ? x86::convert_to<x_i16>(y_i32(_mm256_srav_epi32(
                               convert_to<y_i32>(a), convert_to<y_i32>(b))))
                         : x86::convert_to<x_u16>(y_u32(_mm256_srlv_epi32(
                               convert_to<y_u32>(a), convert_to<y_u32>(b))));
    } else if constexpr (is_word && is_xmm && have_sse4_1) {  //{{{2
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
            a = _mm_blendv_epi8(a, builtin_cast<llong>(a >> 8), b);
            // do a =>> 4 where b[2] is set
            a = _mm_blendv_epi8(a, builtin_cast<llong>(a >> 4), b = _mm_add_epi16(b, b));
            // do a =>> 2 where b[1] is set
            a = _mm_blendv_epi8(a, builtin_cast<llong>(a >> 2), b = _mm_add_epi16(b, b));
            // do a =>> 1 where b[0] is set
            return _mm_blendv_epi8(a, builtin_cast<llong>(a >> 1), _mm_add_epi16(b, b));
        }
    } else if constexpr (is_word && is_xmm && have_sse2) {  //{{{2
        auto &&blend = [](T a, T b, T c) { return (~c & a) | (c & b); };
        if constexpr (is_signed) {
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of b is [0, 15]
            // => only the 4 low bits of b are relevant
            // do a =>> 8 where b[3] is set
            a = blend(a, _mm_srai_epi16(a, 8),
                            _mm_cmpgt_epi16(b, auto_broadcast(0x00070007)));
            // do a =>> 4 where b[2] is set
            a = blend(a, _mm_srai_epi16(a, 4),
                            _mm_cmpgt_epi16(and_(b, auto_broadcast(0x00040004)),
                                            _mm_setzero_si128()));
            // do a =>> 2 where b[1] is set
            a = blend(a, _mm_srai_epi16(a, 2),
                            _mm_cmpgt_epi16(and_(b, auto_broadcast(0x00020002)),
                                            _mm_setzero_si128()));
            // do a =>> 1 where b[0] is set
            return blend(a, _mm_srai_epi16(a, 1),
                               _mm_cmpgt_epi16(and_(b, auto_broadcast(0x00010001)),
                                               _mm_setzero_si128()));
        } else {
            // exploit UB: The behavior is undefined if the right operand is [...] greater
            // than or equal to the length in bits of the promoted left operand.
            // => valid input range for each element of b is [0, 15]
            // => only the 4 low bits of b are relevant
            // do a =>> 8 where b[3] is set
            a = blend(a, builtin_cast<llong>(a >> 8),
                            _mm_cmpgt_epi16(b, auto_broadcast(0x00070007)));
            // do a =>> 4 where b[2] is set
            a = blend(a, builtin_cast<llong>(a >> 4),
                            _mm_cmpgt_epi16(and_(b, auto_broadcast(0x00040004)),
                                            _mm_setzero_si128()));
            // do a =>> 2 where b[1] is set
            a = blend(a, builtin_cast<llong>(a >> 2),
                            _mm_cmpgt_epi16(and_(b, auto_broadcast(0x00020002)),
                                            _mm_setzero_si128()));
            // do a =>> 1 where b[0] is set
            return blend(a, builtin_cast<llong>(a >> 1),
                               _mm_cmpgt_epi16(and_(b, auto_broadcast(0x00010001)),
                                               _mm_setzero_si128()));
        }
    } else if constexpr (is_word && is_ymm && have_avx512bw_vl) {  //{{{2
        return is_signed ? _mm256_srav_epi16(ai, bi) : _mm256_srlv_epi16(ai, bi);
    } else if constexpr (is_word && is_ymm && have_avx2) {  //{{{2
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
                (builtin_cast<uint>(a) & 0xffffu) >> (builtin_cast<uint>(b) & 0xffffu),
                builtin_cast<uint>(a) >> (builtin_cast<uint>(b) >> 16), 0xaa);
        }
    } else if constexpr (is_word && is_zmm && have_avx512bw) {  //{{{2
        return is_signed ? _mm512_srav_epi16(ai, bi) : _mm512_srlv_epi16(ai, bi);
    } else if constexpr (is_dword && is_xmm && !have_avx2) {  //{{{2
        if constexpr (is_signed) {
            const auto r0 = _mm_sra_epi32(a, _mm_unpacklo_epi32(b, _mm_setzero_si128()));
            const auto r1 = _mm_sra_epi32(a, _mm_srli_epi64(b, 32));
            const auto r2 = _mm_sra_epi32(a, _mm_unpackhi_epi32(b, _mm_setzero_si128()));
            const auto r3 = _mm_sra_epi32(a, _mm_srli_si128(b, 12));
            if constexpr (have_sse4_1) {
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
            if constexpr (have_sse4_1) {
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
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

// vim: foldmethod=marker
