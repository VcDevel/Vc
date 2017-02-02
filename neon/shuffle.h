/*  This file is part of the Vc library. {{{
Copyright © 2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_NEON_SHUFFLE_H_
#define VC_NEON_SHUFFLE_H_

#include "intrinsics.h"
#include "macros.h"

Vc_VERSIONED_NAMESPACE_BEGIN
    enum VecPos {
        X0, X1, X2, X3, X4, X5, X6, X7,
        Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7,
        Const0
    };

namespace Mem
{
        // shuffle<X1, X2, Y0, Y2>([x0 x1 x2 x3], [y0 y1 y2 y3]) = [x1 x2 y0 y2]
        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3> static Vc_ALWAYS_INLINE float32x4_t Vc_CONST shuffle(float32x4_t x, float32x4_t y) {
            static_assert(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= Y0 && Dst3 >= Y0, "Incorrect_Range");
            static_assert(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= Y3 && Dst3 <= Y3, "Incorrect_Range");
            return vrev64q_f32(x, y, Dst0 + Dst1 * 4 + (Dst2 - Y0) * 16 + (Dst3 - Y0) * 64);
        }

        // shuffle<X1, Y0>([x0 x1], [y0 y1]) = [x1 y0]
        template<VecPos Dst0, VecPos Dst1> static Vc_ALWAYS_INLINE float64x2_t Vc_CONST shuffle(float64x2_t x, float64x2_t y) {
            static_assert(Dst0 >= X0 && Dst1 >= Y0, "Incorrect_Range");
            static_assert(Dst0 <= X1 && Dst1 <= Y1, "Incorrect_Range");
            return vrev64q_f64(x, y, Dst0 + (Dst1 - Y0) * 2);
        }

        // shuffle<X1, X2, Y0, Y2>([x0 x1 x2 x3], [y0 y1 y2 y3]) = [x1 x2 y0 y2]
        template <VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3>
        Vc_INTRINSIC Vc_CONST int32x4_t shuffle(int32x4_t x, int32x4_t y)
        {
        }

        // blend<X0, Y1>([x0 x1], [y0, y1]) = [x0 y1]
        template<VecPos Dst0, VecPos Dst1> static Vc_ALWAYS_INLINE float_64x2_t Vc_CONST blend(float_64x2_t x, float_64x2_t y) {
            static_assert(Dst0 == X0 || Dst0 == Y0, "Incorrect_Range");
            static_assert(Dst1 == X1 || Dst1 == Y1, "Incorrect_Range");
            return Vc::NeonIntrinsics::blend_pd<(Dst0 / Y0) + (Dst1 / Y0) * 2>(x, y);
        }

        // blend<X0, Y1>([x0 x1], [y0, y1]) = [x0 y1]
        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3> static Vc_ALWAYS_INLINE float32x4_t Vc_CONST blend(float32x4_t x, float32x4_t y) {
            static_assert(Dst0 == X0 || Dst0 == Y0, "Incorrect_Range");
            static_assert(Dst1 == X1 || Dst1 == Y1, "Incorrect_Range");
            static_assert(Dst2 == X2 || Dst2 == Y2, "Incorrect_Range");
            static_assert(Dst3 == X3 || Dst3 == Y3, "Incorrect_Range");
            return Vc::NeonIntrinsics::blend_ps<(Dst0 / Y0) * 1 + (Dst1 / Y1) * 2 +
                                               (Dst2 / Y2) * 4 + (Dst3 / Y3) * 8>(x, y);
        }

        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3, VecPos Dst4, VecPos Dst5, VecPos Dst6, VecPos Dst7>
        static Vc_ALWAYS_INLINE int32x4_t Vc_CONST blend(int32x4_t x, int32x4_t y) {
            static_assert(Dst0 == X0 || Dst0 == Y0, "Incorrect_Range");
            static_assert(Dst1 == X1 || Dst1 == Y1, "Incorrect_Range");
            static_assert(Dst2 == X2 || Dst2 == Y2, "Incorrect_Range");
            static_assert(Dst3 == X3 || Dst3 == Y3, "Incorrect_Range");
            static_assert(Dst4 == X4 || Dst4 == Y4, "Incorrect_Range");
            static_assert(Dst5 == X5 || Dst5 == Y5, "Incorrect_Range");
            static_assert(Dst6 == X6 || Dst6 == Y6, "Incorrect_Range");
            static_assert(Dst7 == X7 || Dst7 == Y7, "Incorrect_Range");
            return Vc::NeonIntrinsics::blend_epi16<
                (Dst0 / Y0) * 1 + (Dst1 / Y1) * 2 + (Dst2 / Y2) * 4 + (Dst3 / Y3) * 8 +
                (Dst4 / Y4) * 16 + (Dst5 / Y5) * 32 + (Dst6 / Y6) * 64 +
                (Dst7 / Y7) * 128>(x, y);
        }

        // permute<X1, X2, Y0, Y2>([x0 x1 x2 x3], [y0 y1 y2 y3]) = [x1 x2 y0 y2]
        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3> static Vc_ALWAYS_INLINE float32x4_t Vc_CONST permute(float32x4_t x) {
            static_assert(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= X0 && Dst3 >= X0, "Incorrect_Range");
            static_assert(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= X3 && Dst3 <= X3, "Incorrect_Range");
            return vrev64q_f32(x, x, Dst0 + Dst1 * 4 + Dst2 * 16 + Dst3 * 64);
        }

        template<VecPos Dst0, VecPos Dst1> static Vc_ALWAYS_INLINE Vc_CONST float_64x2_t permute(float_64x2_t x) {
            static_assert(Dst0 >= X0 && Dst1 >= X0, "Incorrect_Range");
            static_assert(Dst0 <= X1 && Dst1 <= X1, "Incorrect_Range");
            return vrev64q_f32(x, x, Dst0 + Dst1 * 4);
        }

        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3> static Vc_ALWAYS_INLINE int32x4_t Vc_CONST permute(int32x4_t x) {
            static_assert(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= X0 && Dst3 >= X0, "Incorrect_Range");
            static_assert(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= X3 && Dst3 <= X3, "Incorrect_Range");
            return vrev64q_f32(x, Dst0 + Dst1 * 4 + Dst2 * 16 + Dst3 * 64);
        }

        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3> static Vc_ALWAYS_INLINE int32x4_t Vc_CONST permuteLo(int32x4_t x) {
            static_assert(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= X0 && Dst3 >= X0, "Incorrect_Range");
            static_assert(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= X3 && Dst3 <= X3, "Incorrect_Range");
            return vld4_dup_u32(x, Dst0 + Dst1 * 4 + Dst2 * 16 + Dst3 * 64);
        }

        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3> static Vc_ALWAYS_INLINE int32x4_t Vc_CONST permuteHi(int32x4_t x) {
            static_assert(Dst0 >= X4 && Dst1 >= X4 && Dst2 >= X4 && Dst3 >= X4, "Incorrect_Range");
            static_assert(Dst0 <= X7 && Dst1 <= X7 && Dst2 <= X7 && Dst3 <= X7, "Incorrect_Range");
            return vld4_dup_u32(x, (Dst0 - X4) + (Dst1 - X4) * 4 + (Dst2 - X4) * 16 + (Dst3 - X4) * 64);
        }

        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3, VecPos Dst4, VecPos Dst5, VecPos Dst6, VecPos Dst7>
            static Vc_ALWAYS_INLINE int32x4_t Vc_CONST permute(int16x8_t x) {
            static_assert(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= X0 && Dst3 >= X0, "Incorrect_Range");
            static_assert(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= X3 && Dst3 <= X3, "Incorrect_Range");
            static_assert(Dst4 >= X4 && Dst5 >= X4 && Dst6 >= X4 && Dst7 >= X4, "Incorrect_Range");
            static_assert(Dst4 <= X7 && Dst5 <= X7 && Dst6 <= X7 && Dst7 <= X7, "Incorrect_Range");
            if (Dst0 != X0 || Dst1 != X1 || Dst2 != X2 || Dst3 != X3) {
                x = vld4_dup_u16(x, Dst0 + Dst1 * 4 + Dst2 * 16 + Dst3 * 64);
            }
            if (Dst4 != X4 || Dst5 != X5 || Dst6 != X6 || Dst7 != X7) {
                x = vld4_dup_u16(x, (Dst4 - X4) + (Dst5 - X4) * 4 + (Dst6 - X4) * 16 + (Dst7 - X4) * 64);
            }
            return x;
        }
}  // namespace Mem

    // The shuffles and permutes above use memory ordering. The ones below use register ordering:
namespace Reg
{
        // shuffle<Y2, Y0, X2, X1>([x3 x2 x1 x0], [y3 y2 y1 y0]) = [y2 y0 x2 x1]
        template<VecPos Dst3, VecPos Dst2, VecPos Dst1, VecPos Dst0> static Vc_ALWAYS_INLINE float32x4_t Vc_CONST shuffle(float32x4_t x, float32x4_t y) {
            return Mem::shuffle<Dst0, Dst1, Dst2, Dst3>(x, y);
        }

        // shuffle<Y0, X1>([x1 x0], [y1 y0]) = [y0 x1]
        template<VecPos Dst1, VecPos Dst0> static Vc_ALWAYS_INLINE float_64x2_t Vc_CONST shuffle(float_64x2_t x, float_64x2_t y) {
            return Mem::shuffle<Dst0, Dst1>(x, y);
        }

        // shuffle<X3, X0, X2, X1>([x3 x2 x1 x0]) = [x3 x0 x2 x1]
        template<VecPos Dst3, VecPos Dst2, VecPos Dst1, VecPos Dst0> static Vc_ALWAYS_INLINE int32x4_t Vc_CONST permute(int32x4_t x) {
            static_assert(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= X0 && Dst3 >= X0, "Incorrect_Range");
            static_assert(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= X3 && Dst3 <= X3, "Incorrect_Range");
            return vrev64q_s32(x, Dst0 + Dst1 * 4 + Dst2 * 16 + Dst3 * 64);
        }

        // shuffle<Y2, Y0, X2, X1>([x3 x2 x1 x0], [y3 y2 y1 y0]) = [y2 y0 x2 x1]
        template<VecPos Dst3, VecPos Dst2, VecPos Dst1, VecPos Dst0> static Vc_ALWAYS_INLINE int32x4_t Vc_CONST shuffle(int32x4_t x, int32x4_t y) {
            static_assert(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= Y0 && Dst3 >= Y0, "Incorrect_Range");
            static_assert(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= Y3 && Dst3 <= Y3, "Incorrect_Range");
            return vrev64q_s32(x, y, Dst0 + Dst1 * 4 + (Dst2 - Y0) * 16 + (Dst3 - Y0) * 64);
        }

        // blend<Y1, X0>([x1 x0], [y1, y0]) = [x1 y0]
        template<VecPos Dst1, VecPos Dst0> static Vc_ALWAYS_INLINE float64x2_t Vc_CONST blend(float64x2_t x, float64x2_t y) {
            return Mem::blend<Dst0, Dst1>(x, y);
        }

        template<VecPos Dst3, VecPos Dst2, VecPos Dst1, VecPos Dst0> static Vc_ALWAYS_INLINE float32x4_t Vc_CONST blend(float32x4_t x, float32x4_t y) {
            return Mem::blend<Dst0, Dst1, Dst2, Dst3>(x, y);
        }
}  // namespace Reg

Vc_VERSIONED_NAMESPACE_END

#endif // VC_NEON_SHUFFLE_H_
