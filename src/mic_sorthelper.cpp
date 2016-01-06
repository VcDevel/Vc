/*  This file is part of the Vc library. {{{
Copyright © 2013-2015 Matthias Kretz <kretz@kde.org>

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

#include <mic/intrinsics.h>
#include <mic/casts.h>
#include <mic/sorthelper.h>
#include <mic/macros.h>

namespace Vc_VERSIONED_NAMESPACE
{
namespace MIC
{

// reference:
//////////// bitonic merge kernel (Chhugani2008) ///////////////
// There are two outgoing lines from every value-pair in the diagram below. The left line
// signifies the min result, the right line the max result.
//
// Every variable vN, wN, xN, yN signifies a pair of values that is to be split into min
// and max.
//
// a <= b <= c <= d <= e <= f <= g <= h
// p => o => n => m => l => k => j => i
// ap bo cn dm el fk gj hi (or: hi gj fk el dm cn bo ap)
// v0 v1 v2 v3 v4 v5 v6 v7
// │╲ │╲ │╲ │╲ ╱│ ╱│ ╱│ ╱│
// │ ╲│ ╲│ ╲│ ╳ │╱ │╱ │╱ │
// │  ╲  ╲  ╲╱ ╲╱  ╱  ╱  │
// │  │╲ │╲ ╱╲ ╱╲ ╱│ ╱│  │
// │  │ ╲│ ╳│ ╳ │╳ │╱ │  │
// │  │  ╲╱ ╲╱ ╲╱ ╲╱  │  │
// │  │  ╱╲ ╱╲ ╱╲ ╱╲  │  │
// │  │ ╱│ ╳│ ╳ │╳ │╲ │  │
// │  │╱ │╱ ╲╱ ╲╱ ╲│ ╲│  │
// │  ╱  ╱  ╱╲ ╱╲  ╲  ╲  │
// │ ╱│ ╱│ ╱│ ╳ │╲ │╲ │╲ │
// │╱ │╱ │╱ │╱ ╲│ ╲│ ╲│ ╲│
// w0 w1 w2 w3 w4 w5 w6 w7
// │╲ │╲ ╱│ ╱│ │╲ │╲ ╱│ ╱│
// │ ╲│ ╳ │╱ │ │ ╲│ ╳ │╱ │
// │  ╲╱ ╲╱  │ │  ╲╱ ╲╱  │
// │  ╱╲ ╱╲  │ │  ╱╲ ╱╲  │
// │ ╱│ ╳ │╲ │ │ ╱│ ╳ │╲ │
// │╱ │╱ ╲│ ╲│ │╱ │╱ ╲│ ╲│
// x0 x1 x2 x3 x4 x5 x6 x7
// │╲ ╱│ │╲ ╱│ │╲ ╱│ │╲ ╱│
// │ ╳ │ │ ╳ │ │ ╳ │ │ ╳ │
// │╱ ╲│ │╱ ╲│ │╱ ╲│ │╱ ╲│
// y0 y1 y2 y3 y4 y5 y6 y7
// || || || || || || || ||
// ab cd ef gh ij kl mn op

template <> __m512i SortHelper<int>::sort(VectorType in)
{
    //__int64 masks = 0x55559999aaaaf0f0;
    //auto m0x5555 = _mm512_kextract_64(masks, 3);
    //auto m0x9999 = _mm512_kextract_64(masks, 2);
    //auto m0xaaaa = _mm512_kextract_64(masks, 1);
    //auto m0xf0f0 = _mm512_kextract_64(masks, 0);

    auto lh  = in; // dcba
    auto min = _mm512_min_epi32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_CDAB)); // ↓dc ↓dc ↓ba ↓ba
    auto max = _mm512_max_epi32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_CDAB)); // ↑dc ↑dc ↑ba ↑ba
    lh = _mm512_mask_mov_epi32(min, 0xaaaa, max); // ↑dc ↓dc ↑ba ↓ba

    min = _mm512_min_epi32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_BADC)); // ↓↑dc↑ba  ↓dcba  ↓↑dc↑ba  ↓dcba
    max = _mm512_max_epi32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_BADC)); //  ↑dcba  ↑↓dc↓ba  ↑dcba  ↑↓dc↓ba
    lh = _mm512_mask_mov_epi32(min, 0x6666, max);                         //  ↑dcba   ↓dcba  ↓↑dc↑ba ↑↓dc↓ba
    auto tmp = _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_CDAB);               //  ↓dcba   ↑dcba  ↑↓dc↓ba ↓↑dc↑ba
    min = _mm512_min_epi32(lh, tmp); // ↓dcba ↓dcba ↓↑↓dc↓ba↓↑dc↑ba ↓↑↓dc↓ba↓↑dc↑ba  |  aabb
    max = _mm512_max_epi32(lh, tmp); // ↑dcba ↑dcba ↑↑↓dc↓ba↓↑dc↑ba ↑↑↓dc↓ba↓↑dc↑ba  |  ddcc
    lh  = _mm512_mask_mov_epi32(max, 0x6666, _mm512_swizzle_epi32(min, _MM_SWIZ_REG_DACB)); // hfeg|dbac

    tmp = _mm512_permute4f128_epi32(lh, _MM_PERM_CDAB);
    // a <= b <= c <= d
    // e <= f <= g <= h
    // bitonic merge 4+4 -> 8
    min = _mm512_min_epi32(lh, _mm512_swizzle_epi32(tmp, _MM_SWIZ_REG_BADC)); // ↓ed ↓gb ↓ha ↓fc ↓ed ↓gb ↓ha ↓fc
    max = _mm512_max_epi32(lh, _mm512_swizzle_epi32(tmp, _MM_SWIZ_REG_BADC)); // ↑ed ↑gb ↑ha ↑fc ↑ed ↑gb ↑ha ↑fc
    lh  = _mm512_mask_mov_epi32(min, 0xf0f0, max);                         // ↑ed  ↑gb  ↑ha  ↑fc ↓ha  ↓fc  ↓ed  ↓gb
                                                                           //  └╴x3╶┘    └╴x2╶┘   └╴x0╶┘    └╴x1╶┘
    min = _mm512_min_epi32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_CDAB)); // ↓x3 ↓x3 ↓x2 ↓x2 ↓x0 ↓x0 ↓x1 ↓x1
    max = _mm512_max_epi32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_CDAB)); // ↑x3 ↑x3 ↑x2 ↑x2 ↑x0 ↑x0 ↑x1 ↑x1
    lh  = _mm512_mask_mov_epi32(min, 0xaaaa, max);                        // ↑x3 ↓x3 ↑x2 ↓x2 ↑x0 ↓x0 ↑x1 ↓x1
                                                                          //  └──╴y3╶─┘   │   └──╴y1╶─┘   │
                                                                          //      └──╴y2╶─┘       └──╴y0╶─┘
    min = _mm512_min_epi32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_BADC)); // ↓y3 ↓y2 ↓y3 ↓y2 ↓y1 ↓y0 ↓y1 ↓y0
    max = _mm512_max_epi32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_BADC)); // ↑y3 ↑y2 ↑y3 ↑y2 ↑y1 ↑y0 ↑y1 ↑y0
    lh  = _mm512_mask_mov_epi32(min, 0x6666, max);                        // ↓y3 ↑y2 ↑y3 ↓y2 ↓y1 ↑y0 ↑y1 ↓y0 | onpm|kjli|gfhe|cbda (2130)
    tmp = _mm512_permute4f128_epi32(lh, _MM_PERM_ABCD);                   //                                   cbda|gfhe|kjli|onpm
                                                                          // required after swizzle:           bcad|...
    // lh  = [8, 11, 9, 10, 12, 15, 13, 14, 0, 3, 1, 2, 4, 7, 5, 6]
    // tmp = [4, 7, 5, 6, 0, 3, 1, 2, 12, 15, 13, 14, 8, 11, 9, 10]
    // a <= b <= c <= d <= e <= f <= g <= h
    // i <= j <= k <= l <= m <= n <= o <= p
    // bitonic merge 8+8 -> 16
    // needed compares: v7=pa v6=ob v5=nc v4=md v3=le v2=kf v1=jg v0=ih
    min = _mm512_min_epi32(lh, _mm512_swizzle_epi32(tmp, _MM_SWIZ_REG_CDAB)); // ↓v6 ↓v5 ↓v7 ↓v4 ↓v2 ↓v1 ↓v3 ↓v0 ↓v6 ↓v5 ↓v7 ↓v4 ↓v2 ↓v1 ↓v3 ↓v0
    max = _mm512_max_epi32(lh, _mm512_swizzle_epi32(tmp, _MM_SWIZ_REG_CDAB)); // ↑v6 ↑v5 ↑v7 ↑v4 ↑v2 ↑v1 ↑v3 ↑v0 ↑v6 ↑v5 ↑v7 ↑v4 ↑v2 ↑v1 ↑v3 ↑v0
    lh  = _mm512_mask_mov_epi32(min, 0xff00, max);                            // ↑v6 ↑v5 ↑v7 ↑v4 ↑v2 ↑v1 ↑v3 ↑v0 ↓v6 ↓v5 ↓v7 ↓v4 ↓v2 ↓v1 ↓v3 ↓v0
                                                                              //  │   └───│───│w5╶│───┘   │   │   │   └───│───│w1╶│───┘   │   │
                                                                              //  │       │   └───│──╴w4╶─│───┘   │       │   └───│──╴w0╶─│───┘
                                                                              //  └──────╴w6╶─────┘       │       └──────╴w2╶─────┘       │
                                                                              //          └──────╴w7╶─────┘               └──────╴w3╶─────┘
    // lh  = [7, 4, 6, 5, 3, 0, 2, 1, 15, 12, 14, 13, 11, 8, 10, 9]
    tmp = _mm512_permute4f128_epi32(lh, _MM_PERM_CDAB);
    // tmp = [3, 0, 2, 1, 7, 4, 6, 5, 11, 8, 10, 9, 15, 12, 14, 13]

    // bitonic merge 4+4 -> 8
    min = _mm512_min_epi32(lh, _mm512_swizzle_epi32(tmp, _MM_SWIZ_REG_DCBA)); // ↓w6 ↓w5 ↓w7 ↓w4 ↓w6 ↓w5 ↓w7 ↓w4 ...
    max = _mm512_max_epi32(lh, _mm512_swizzle_epi32(tmp, _MM_SWIZ_REG_DCBA)); // ↑w6 ↑w5 ↑w7 ↑w4 ↑w6 ↑w5 ↑w7 ↑w4 ...
    lh  = _mm512_mask_mov_epi32(min, 0xf0f0, max);                            // ↑w6 ↑w5 ↓w7 ↓w4 ↓w6 ↓w5 ↑w7 ↑w4 ...
                                                                              //  │   └╴x7╶┘  │   │   └╴x5╶┘  │
                                                                              //  └────╴x6╶───┘   └────╴x4╶───┘
    lh  = _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_DACB);                        // ↑w6 ↓w4 ↑w5 ↓w7 ↓w6 ↑w4 ↓w5 ↑w7 ...
                                                                              //  └╴x6╶┘  └╴x7╶┘  └╴x4╶┘  └╴x5╶┘
    min = _mm512_min_epi32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_CDAB));  // ↓x6 ↓x6 ↓x7 ↓x7 ↓x4 ↓x4 ↓x5 ↓x5 ...
    max = _mm512_max_epi32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_CDAB));  // ↑x6 ↑x6 ↑x7 ↑x7 ↑x4 ↑x4 ↑x5 ↑x5 ...
    lh  = _mm512_mask_mov_epi32(min, 0xaaaa, max);                            // ↑x6 ↓x6 ↑x7 ↓x7 ↑x4 ↓x4 ↑x5 ↓x5 ...
                                                                              //  └──╴y7╶─┘   │   └──╴y5╶─┘   │
                                                                              //      └──╴y6╶─┘       └──╴y4╶─┘
    min = _mm512_min_epi32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_BADC));  // ↓y7 ↓y6 ↓y7 ↓y6 ...
    max = _mm512_max_epi32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_BADC));  // ↑y7 ↑y6 ↑y7 ↑y6 ...
    lh  = _mm512_mask_mov_epi32(min, 0xcccc, max);

    return _mm512_shuffle_epi32(lh, _MM_PERM_DBCA);
}

template <> __m512i SortHelper<unsigned int>::sort(VectorType in)
{
    //__int64 masks = 0x55559999aaaaf0f0;
    //auto m0x5555 = _mm512_kextract_64(masks, 3);
    //auto m0x9999 = _mm512_kextract_64(masks, 2);
    //auto m0xaaaa = _mm512_kextract_64(masks, 1);
    //auto m0xf0f0 = _mm512_kextract_64(masks, 0);

    auto lh  = in; // dcba
    auto min = _mm512_min_epu32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_CDAB)); // ↓dc ↓dc ↓ba ↓ba
    auto max = _mm512_max_epu32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_CDAB)); // ↑dc ↑dc ↑ba ↑ba
    lh = _mm512_mask_mov_epi32(min, 0xaaaa, max); // ↑dc ↓dc ↑ba ↓ba

    min = _mm512_min_epu32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_BADC)); // ↓↑dc↑ba  ↓dcba  ↓↑dc↑ba  ↓dcba
    max = _mm512_max_epu32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_BADC)); //  ↑dcba  ↑↓dc↓ba  ↑dcba  ↑↓dc↓ba
    lh = _mm512_mask_mov_epi32(min, 0x6666, max);                         //  ↑dcba   ↓dcba  ↓↑dc↑ba ↑↓dc↓ba
    auto tmp = _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_CDAB);               //  ↓dcba   ↑dcba  ↑↓dc↓ba ↓↑dc↑ba
    min = _mm512_min_epu32(lh, tmp); // ↓dcba ↓dcba ↓↑↓dc↓ba↓↑dc↑ba ↓↑↓dc↓ba↓↑dc↑ba  |  aabb
    max = _mm512_max_epu32(lh, tmp); // ↑dcba ↑dcba ↑↑↓dc↓ba↓↑dc↑ba ↑↑↓dc↓ba↓↑dc↑ba  |  ddcc
    lh  = _mm512_mask_mov_epi32(max, 0x6666, _mm512_swizzle_epi32(min, _MM_SWIZ_REG_DACB)); // hfeg|dbac

    tmp = _mm512_permute4f128_epi32(lh, _MM_PERM_CDAB);
    // a <= b <= c <= d
    // e <= f <= g <= h
    // bitonic merge 4+4 -> 8
    min = _mm512_min_epu32(lh, _mm512_swizzle_epi32(tmp, _MM_SWIZ_REG_BADC)); // ↓ed ↓gb ↓ha ↓fc ↓ed ↓gb ↓ha ↓fc
    max = _mm512_max_epu32(lh, _mm512_swizzle_epi32(tmp, _MM_SWIZ_REG_BADC)); // ↑ed ↑gb ↑ha ↑fc ↑ed ↑gb ↑ha ↑fc
    lh  = _mm512_mask_mov_epi32(min, 0xf0f0, max);                         // ↑ed  ↑gb  ↑ha  ↑fc ↓ha  ↓fc  ↓ed  ↓gb
                                                                        //  └╴x3╶┘    └╴x2╶┘   └╴x0╶┘    └╴x1╶┘
    min = _mm512_min_epu32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_CDAB)); // ↓x3 ↓x3 ↓x2 ↓x2 ↓x0 ↓x0 ↓x1 ↓x1
    max = _mm512_max_epu32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_CDAB)); // ↑x3 ↑x3 ↑x2 ↑x2 ↑x0 ↑x0 ↑x1 ↑x1
    lh  = _mm512_mask_mov_epi32(min, 0xaaaa, max);                        // ↑x3 ↓x3 ↑x2 ↓x2 ↑x0 ↓x0 ↑x1 ↓x1
                                                                       //  └──╴y3╶─┘   │   └──╴y1╶─┘   │
                                                                       //      └──╴y2╶─┘       └──╴y0╶─┘
    min = _mm512_min_epu32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_BADC)); // ↓y3 ↓y2 ↓y3 ↓y2 ↓y1 ↓y0 ↓y1 ↓y0
    max = _mm512_max_epu32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_BADC)); // ↑y3 ↑y2 ↑y3 ↑y2 ↑y1 ↑y0 ↑y1 ↑y0
    lh  = _mm512_mask_mov_epi32(min, 0x6666, max);                        // ↓y3 ↑y2 ↑y3 ↓y2 ↓y1 ↑y0 ↑y1 ↓y0 | onpm|kjli|gfhe|cbda (2130)
    tmp = _mm512_permute4f128_epi32(lh, _MM_PERM_ABCD);                   //                                   cbda|gfhe|kjli|onpm
                                                                       // required after swizzle:           bcad|...
    // lh  = [8, 11, 9, 10, 12, 15, 13, 14, 0, 3, 1, 2, 4, 7, 5, 6]
    // tmp = [4, 7, 5, 6, 0, 3, 1, 2, 12, 15, 13, 14, 8, 11, 9, 10]
    // a <= b <= c <= d <= e <= f <= g <= h
    // i <= j <= k <= l <= m <= n <= o <= p
    // bitonic merge 8+8 -> 16
    // needed compares: v7=pa v6=ob v5=nc v4=md v3=le v2=kf v1=jg v0=ih
    min = _mm512_min_epu32(lh, _mm512_swizzle_epi32(tmp, _MM_SWIZ_REG_CDAB)); // ↓v6 ↓v5 ↓v7 ↓v4 ↓v2 ↓v1 ↓v3 ↓v0 ↓v6 ↓v5 ↓v7 ↓v4 ↓v2 ↓v1 ↓v3 ↓v0
    max = _mm512_max_epu32(lh, _mm512_swizzle_epi32(tmp, _MM_SWIZ_REG_CDAB)); // ↑v6 ↑v5 ↑v7 ↑v4 ↑v2 ↑v1 ↑v3 ↑v0 ↑v6 ↑v5 ↑v7 ↑v4 ↑v2 ↑v1 ↑v3 ↑v0
    lh  = _mm512_mask_mov_epi32(min, 0xff00, max);                         // ↑v6 ↑v5 ↑v7 ↑v4 ↑v2 ↑v1 ↑v3 ↑v0 ↓v6 ↓v5 ↓v7 ↓v4 ↓v2 ↓v1 ↓v3 ↓v0
                                                                        //  │   └───│───│w5╶│───┘   │   │   │   └───│───│w1╶│───┘   │   │
                                                                        //  │       │   └───│──╴w4╶─│───┘   │       │   └───│──╴w0╶─│───┘
                                                                        //  └──────╴w6╶─────┘       │       └──────╴w2╶─────┘       │
                                                                        //          └──────╴w7╶─────┘               └──────╴w3╶─────┘
    // lh  = [7, 4, 6, 5, 3, 0, 2, 1, 15, 12, 14, 13, 11, 8, 10, 9]
    tmp = _mm512_permute4f128_epi32(lh, _MM_PERM_CDAB);
    // tmp = [3, 0, 2, 1, 7, 4, 6, 5, 11, 8, 10, 9, 15, 12, 14, 13]

    // bitonic merge 4+4 -> 8
    min = _mm512_min_epu32(lh, _mm512_swizzle_epi32(tmp, _MM_SWIZ_REG_DCBA)); // ↓w6 ↓w5 ↓w7 ↓w4 ↓w6 ↓w5 ↓w7 ↓w4 ...
    max = _mm512_max_epu32(lh, _mm512_swizzle_epi32(tmp, _MM_SWIZ_REG_DCBA)); // ↑w6 ↑w5 ↑w7 ↑w4 ↑w6 ↑w5 ↑w7 ↑w4 ...
    lh  = _mm512_mask_mov_epi32(min, 0xf0f0, max);                         // ↑w6 ↑w5 ↓w7 ↓w4 ↓w6 ↓w5 ↑w7 ↑w4 ...
                                                                        //  │   └╴x7╶┘  │   │   └╴x5╶┘  │
                                                                        //  └────╴x6╶───┘   └────╴x4╶───┘
    lh  = _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_DACB);                     // ↑w6 ↓w4 ↑w5 ↓w7 ↓w6 ↑w4 ↓w5 ↑w7 ...
                                                                        //  └╴x6╶┘  └╴x7╶┘  └╴x4╶┘  └╴x5╶┘
    min = _mm512_min_epu32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_CDAB));  // ↓x6 ↓x6 ↓x7 ↓x7 ↓x4 ↓x4 ↓x5 ↓x5 ...
    max = _mm512_max_epu32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_CDAB));  // ↑x6 ↑x6 ↑x7 ↑x7 ↑x4 ↑x4 ↑x5 ↑x5 ...
    lh  = _mm512_mask_mov_epi32(min, 0xaaaa, max);                         // ↑x6 ↓x6 ↑x7 ↓x7 ↑x4 ↓x4 ↑x5 ↓x5 ...
                                                                        //  └──╴y7╶─┘   │   └──╴y5╶─┘   │
                                                                        //      └──╴y6╶─┘       └──╴y4╶─┘
    min = _mm512_min_epu32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_BADC));  // ↓y7 ↓y6 ↓y7 ↓y6 ...
    max = _mm512_max_epu32(lh, _mm512_swizzle_epi32(lh, _MM_SWIZ_REG_BADC));  // ↑y7 ↑y6 ↑y7 ↑y6 ...
    lh  = _mm512_mask_mov_epi32(min, 0xcccc, max);

    return _mm512_shuffle_epi32(lh, _MM_PERM_DBCA);
}

template <> __m512i SortHelper<short>::sort(VectorType in)
{
    return SortHelper<int>::sort(in);
}

template <> __m512i SortHelper<unsigned short>::sort(VectorType in)
{
    return SortHelper<unsigned int>::sort(
        _mm512_and_epi32(_mm512_set1_epi32(0xffff), in));
}

template <> __m512 SortHelper<float>::sort(VectorType in)
{
    const __int64 masks = 0xaaaa6666f0f0ff00ULL;
    const auto m0xaaaa = _mm512_kextract_64(masks, 3);
    const auto m0x6666 = _mm512_kextract_64(masks, 2);
    const auto m0xf0f0 = _mm512_kextract_64(masks, 1);
    const auto m0xff00 = _mm512_kextract_64(masks, 0);
    const auto m0xcccc = _mm512_kxor(m0xaaaa, m0x6666);

    auto lh  = in; // dcba
    auto min = _mm512_gmin_ps(lh, _mm512_swizzle_ps(lh, _MM_SWIZ_REG_CDAB)); // ↓dc ↓dc ↓ba ↓ba
    auto max = _mm512_gmax_ps(lh, _mm512_swizzle_ps(lh, _MM_SWIZ_REG_CDAB)); // ↑dc ↑dc ↑ba ↑ba
    lh = _mm512_mask_mov_ps(min, m0xaaaa, max); // ↑dc ↓dc ↑ba ↓ba

    min = _mm512_gmin_ps(lh, _mm512_swizzle_ps(lh, _MM_SWIZ_REG_BADC)); // ↓↑dc↑ba  ↓dcba  ↓↑dc↑ba  ↓dcba
    max = _mm512_gmax_ps(lh, _mm512_swizzle_ps(lh, _MM_SWIZ_REG_BADC)); //  ↑dcba  ↑↓dc↓ba  ↑dcba  ↑↓dc↓ba
    lh = _mm512_mask_mov_ps(min, m0x6666, max);                         //  ↑dcba   ↓dcba  ↓↑dc↑ba ↑↓dc↓ba
    auto tmp = _mm512_swizzle_ps(lh, _MM_SWIZ_REG_CDAB);               //  ↓dcba   ↑dcba  ↑↓dc↓ba ↓↑dc↑ba
    min = _mm512_gmin_ps(lh, tmp); // ↓dcba ↓dcba ↓↑↓dc↓ba↓↑dc↑ba ↓↑↓dc↓ba↓↑dc↑ba  |  aabb
    max = _mm512_gmax_ps(lh, tmp); // ↑dcba ↑dcba ↑↑↓dc↓ba↓↑dc↑ba ↑↑↓dc↓ba↓↑dc↑ba  |  ddcc
    lh  = _mm512_mask_mov_ps(max, m0x6666, _mm512_swizzle_ps(min, _MM_SWIZ_REG_DACB)); // hfeg|dbac

    tmp = _mm512_permute4f128_ps(lh, _MM_PERM_CDAB);
    // a <= b <= c <= d
    // e <= f <= g <= h
    // bitonic merge 4+4 -> 8
    min = _mm512_gmin_ps(lh, _mm512_swizzle_ps(tmp, _MM_SWIZ_REG_BADC)); // ↓ed ↓gb ↓ha ↓fc ↓ed ↓gb ↓ha ↓fc
    max = _mm512_gmax_ps(lh, _mm512_swizzle_ps(tmp, _MM_SWIZ_REG_BADC)); // ↑ed ↑gb ↑ha ↑fc ↑ed ↑gb ↑ha ↑fc
    lh  = _mm512_mask_mov_ps(min, m0xf0f0, max);                         // ↑ed  ↑gb  ↑ha  ↑fc ↓ha  ↓fc  ↓ed  ↓gb
                                                                        //  └╴x3╶┘    └╴x2╶┘   └╴x0╶┘    └╴x1╶┘
    min = _mm512_gmin_ps(lh, _mm512_swizzle_ps(lh, _MM_SWIZ_REG_CDAB)); // ↓x3 ↓x3 ↓x2 ↓x2 ↓x0 ↓x0 ↓x1 ↓x1
    max = _mm512_gmax_ps(lh, _mm512_swizzle_ps(lh, _MM_SWIZ_REG_CDAB)); // ↑x3 ↑x3 ↑x2 ↑x2 ↑x0 ↑x0 ↑x1 ↑x1
    lh  = _mm512_mask_mov_ps(min, m0xaaaa, max);                        // ↑x3 ↓x3 ↑x2 ↓x2 ↑x0 ↓x0 ↑x1 ↓x1
                                                                       //  └──╴y3╶─┘   │   └──╴y1╶─┘   │
                                                                       //      └──╴y2╶─┘       └──╴y0╶─┘
    min = _mm512_gmin_ps(lh, _mm512_swizzle_ps(lh, _MM_SWIZ_REG_BADC)); // ↓y3 ↓y2 ↓y3 ↓y2 ↓y1 ↓y0 ↓y1 ↓y0
    max = _mm512_gmax_ps(lh, _mm512_swizzle_ps(lh, _MM_SWIZ_REG_BADC)); // ↑y3 ↑y2 ↑y3 ↑y2 ↑y1 ↑y0 ↑y1 ↑y0
    lh  = _mm512_mask_mov_ps(min, m0x6666, max);                        // ↓y3 ↑y2 ↑y3 ↓y2 ↓y1 ↑y0 ↑y1 ↓y0 | onpm|kjli|gfhe|cbda (2130)
    tmp = _mm512_permute4f128_ps(lh, _MM_PERM_ABCD);                   //                                   cbda|gfhe|kjli|onpm
                                                                       // required after swizzle:           bcad|...
    // lh  = [8, 11, 9, 10, 12, 15, 13, 14, 0, 3, 1, 2, 4, 7, 5, 6]
    // tmp = [4, 7, 5, 6, 0, 3, 1, 2, 12, 15, 13, 14, 8, 11, 9, 10]
    // a <= b <= c <= d <= e <= f <= g <= h
    // i <= j <= k <= l <= m <= n <= o <= p
    // bitonic merge 8+8 -> 16
    // needed compares: v7=pa v6=ob v5=nc v4=md v3=le v2=kf v1=jg v0=ih
    min = _mm512_gmin_ps(lh, _mm512_swizzle_ps(tmp, _MM_SWIZ_REG_CDAB)); // ↓v6 ↓v5 ↓v7 ↓v4 ↓v2 ↓v1 ↓v3 ↓v0 ↓v6 ↓v5 ↓v7 ↓v4 ↓v2 ↓v1 ↓v3 ↓v0
    max = _mm512_gmax_ps(lh, _mm512_swizzle_ps(tmp, _MM_SWIZ_REG_CDAB)); // ↑v6 ↑v5 ↑v7 ↑v4 ↑v2 ↑v1 ↑v3 ↑v0 ↑v6 ↑v5 ↑v7 ↑v4 ↑v2 ↑v1 ↑v3 ↑v0
    lh  = _mm512_mask_mov_ps(min, m0xff00, max);                         // ↑v6 ↑v5 ↑v7 ↑v4 ↑v2 ↑v1 ↑v3 ↑v0 ↓v6 ↓v5 ↓v7 ↓v4 ↓v2 ↓v1 ↓v3 ↓v0
                                                                        //  │   └───│───│w5╶│───┘   │   │   │   └───│───│w1╶│───┘   │   │
                                                                        //  │       │   └───│──╴w4╶─│───┘   │       │   └───│──╴w0╶─│───┘
                                                                        //  └──────╴w6╶─────┘       │       └──────╴w2╶─────┘       │
                                                                        //          └──────╴w7╶─────┘               └──────╴w3╶─────┘
    // lh  = [7, 4, 6, 5, 3, 0, 2, 1, 15, 12, 14, 13, 11, 8, 10, 9]
    tmp = _mm512_permute4f128_ps(lh, _MM_PERM_CDAB);
    // tmp = [3, 0, 2, 1, 7, 4, 6, 5, 11, 8, 10, 9, 15, 12, 14, 13]

    // bitonic merge 4+4 -> 8
    min = _mm512_gmin_ps(lh, _mm512_swizzle_ps(tmp, _MM_SWIZ_REG_DCBA)); // ↓w6 ↓w5 ↓w7 ↓w4 ↓w6 ↓w5 ↓w7 ↓w4 ...
    max = _mm512_gmax_ps(lh, _mm512_swizzle_ps(tmp, _MM_SWIZ_REG_DCBA)); // ↑w6 ↑w5 ↑w7 ↑w4 ↑w6 ↑w5 ↑w7 ↑w4 ...
    lh  = _mm512_mask_mov_ps(min, m0xf0f0, max);                         // ↑w6 ↑w5 ↓w7 ↓w4 ↓w6 ↓w5 ↑w7 ↑w4 ...
                                                                        //  │   └╴x7╶┘  │   │   └╴x5╶┘  │
                                                                        //  └────╴x6╶───┘   └────╴x4╶───┘
    lh  = _mm512_swizzle_ps(lh, _MM_SWIZ_REG_DACB);                     // ↑w6 ↓w4 ↑w5 ↓w7 ↓w6 ↑w4 ↓w5 ↑w7 ...
                                                                        //  └╴x6╶┘  └╴x7╶┘  └╴x4╶┘  └╴x5╶┘
    min = _mm512_gmin_ps(lh, _mm512_swizzle_ps(lh, _MM_SWIZ_REG_CDAB));  // ↓x6 ↓x6 ↓x7 ↓x7 ↓x4 ↓x4 ↓x5 ↓x5 ...
    max = _mm512_gmax_ps(lh, _mm512_swizzle_ps(lh, _MM_SWIZ_REG_CDAB));  // ↑x6 ↑x6 ↑x7 ↑x7 ↑x4 ↑x4 ↑x5 ↑x5 ...
    lh  = _mm512_mask_mov_ps(min, m0xaaaa, max);                         // ↑x6 ↓x6 ↑x7 ↓x7 ↑x4 ↓x4 ↑x5 ↓x5 ...
                                                                        //  └──╴y7╶─┘   │   └──╴y5╶─┘   │
                                                                        //      └──╴y6╶─┘       └──╴y4╶─┘
    min = _mm512_gmin_ps(lh, _mm512_swizzle_ps(lh, _MM_SWIZ_REG_BADC));  // ↓y7 ↓y6 ↓y7 ↓y6 ...
    max = _mm512_gmax_ps(lh, _mm512_swizzle_ps(lh, _MM_SWIZ_REG_BADC));  // ↑y7 ↑y6 ↑y7 ↑y6 ...
    lh  = _mm512_mask_mov_ps(min, m0xcccc, max);

    return mic_cast<__m512>(_mm512_shuffle_epi32(mic_cast<__m512i>(lh), _MM_PERM_DBCA));
}

template <> __m512d SortHelper<double>::sort(VectorType in)
{
    // in = hgfe dcba

    __int64 masks = 0x0055009900aa00f0;
    //auto m0x55 = _mm512_kextract_64(masks, 3);
    auto m0xaa = _mm512_kextract_64(masks, 1);
    auto m0xf0 = _mm512_kextract_64(masks, 0);

    /*
    auto less = _mm512_cmplt_pd_mask(in, _mm512_swizzle_pd(in, _MM_SWIZ_REG_CDAB));
    less ^= 0x55;
    // __mmask8 is converted to GPR anyway... less = _mm512_kxor(less, m0x55);
    auto lh = _mm512_mask_blend_pd(less, in, _mm512_swizzle_pd(in, _MM_SWIZ_REG_CDAB));
    //   lh = ↑hg ↓hg ↑fe ↓fe ↑dc ↓dc ↑ba ↓ba

    auto min = _mm512_gmin_pd(lh, _mm512_swizzle_pd(lh, _MM_SWIZ_REG_BADC));
    auto max = _mm512_gmax_pd(lh, _mm512_swizzle_pd(lh, _MM_SWIZ_REG_BADC));
    auto m0x99 = _mm512_kextract_64(masks, 2);
    auto lh = _mm512_mask_mov_pd(min, m0x99, max);

    min = _mm512_gmin_pd(lh, _mm512_swizzle_pd(lh, _MM_SWIZ_REG_CDAB));
    max = _mm512_gmax_pd(lh, _mm512_swizzle_pd(lh, _MM_SWIZ_REG_CDAB));
    lh  = _mm512_mask_mov_pd(min, m0xaa, max);
    */

    __m512d a, b, c = in;
    __asm__("vgminpd %[c]{{cdab}},%[c],%[b]\n"
            "kextract $1,%[masks],%%k7\n" // k7 = 0xaa
            "vgmaxpd %[c]{{cdab}},%[c],%[b]{{%%k7}}\n"
    // b   = ↑hg ↓hg ↑fe ↓fe ↑dc ↓dc ↑ba ↓ba
            "vgminpd %[b]{{badc}},%[b],%[a]\n"
            "kextract $2,%[masks],%%k6\n" // k6 = 0x99
            "vgmaxpd %[b]{{badc}},%[b],%[a]{{%%k6}}\n"
    // min = ↓↑hg↑fe ↓hgfe ↓↑hg↑fe ↓hgfe ↓↑dc↑ba ↓dcba ↓↑dc↑ba ↓dcba
    // max = ↑hgfe ↑↓hg↓fe ↑hgfe ↑↓hg↓fe ↑dcba ↑↓dc↓ba ↑dcba ↑↓dc↓ba
    // a   = ↑hgfe ↓hgfe ↓↑hg↑fe ↑↓hg↓fe ↑dcba ↓dcba ↓↑dc↑ba ↑↓dc↓ba
    //           ✔      ✔     ✘       ✘      ✔     ✔      ✘       ✘
            "vgminpd %[a]{{cdab}},%[a],%[b]\n"
    // min = ↓hgfe X ↓↓↑hg↑fe↑↓hg↓fe X ↓dcba X ↓↓↑dc↑ba↑↓dc↓ba X
    //         0   0        1        1   0   0        1        1
            "vgmaxpd %[a]{{cdab}},%[a],%[b]{{%%k7}}\n"
    // max = ↑hgfe X ↑↓↑hg↑fe↑↓hg↓fe X ↑dcba X ↑↓↑dc↑ba↑↓dc↓ba X
    //         3   3        2        2   3   3        2        2
    // b   = 3(hgfe) 0(hgfe) 2(hgfe) 1(hgfe) 3(dcba) 0(dcba) 2(dcba) 1(dcba)
    //        h       e       g       f       d       a       c       b

    // bitonic merge 4+4 -> 8
            "vmovapd %[b]{{dacb}},%[b]"               "\n\t" // hfeg dbac
            "vpermf32x4 $0x1b,%[b],%[c]"              "\n\t" // _MM_PERM_ABCD: acdb eghf
            "vgminpd %[b],%[c],%[a]"                  "\n\t" // ↓(hfeg dbac, acdb eghf)
            "kextract $0,%[masks],%%k5"               "\n\t" // k5 = 0xf0
            "vgmaxpd %[b],%[c],%[a]{{%%k5}}"          "\n\t" // ↑ed  ↑gb  ↑ha  ↑fc ↓ha  ↓fc  ↓ed  ↓gb
                                                             //  └╴x3╶┘    └╴x2╶┘   └╴x0╶┘    └╴x1╶┘
            "vgminpd %[a]{{cdab}},%[a],%[b]"          "\n\t" // ↓x3 ↓x3 ↓x2 ↓x2 ↓x0 ↓x0 ↓x1 ↓x1
            "vgmaxpd %[a]{{cdab}},%[a],%[b]{{%%k7}}"  "\n\t" // ↑x3 ↓x3 ↑x2 ↓x2 ↑x0 ↓x0 ↑x1 ↓x1
                                                             //  └──╴y3╶─┘   │   └──╴y1╶─┘   │
                                                             //      └──╴y2╶─┘       └──╴y0╶─┘
            "vgminpd %[b]{{badc}},%[b],%[a]"          "\n\t" // ↓y3 ↓y2 ↓y3 ↓y2 ↓y1 ↓y0 ↓y1 ↓y0
            "vgmaxpd %[b]{{badc}},%[b],%[b]"          "\n\t" // ↑y3 ↑y2 ↑y3 ↑y2 ↑y1 ↑y0 ↑y1 ↑y0
            "vmovapd %[a]{{cdab}},%[a]"               "\n\t" // ↓y2 ↓y3 ↓y2 ↓y3 ↓y0 ↓y1 ↓y0 ↓y1
            "vmovapd %[a]{{dacb}},%[c]"               "\n\t" // ↑y3 ↑y2 ↑y2 ↑y3 ↑y1 ↑y0 ↑y0 ↑y1
            "vmovapd %[b]{{dacb}},%[c]{{%%k7}}"        "\n\t" // 
            : [a]"=x"(a), [b]"=x"(b), [c]"+x"(c) : [masks]"r"(masks) : "k5", "k6", "k7");
    return c;
    /*
    lh = _mm512_swizzle_pd(lh, _MM_SWIZ_REG_DACB); // hfeg dbac
    auto tmp = mic_cast<__m512d>(_mm512_permute4f128_ps(mic_cast<__m512>(lh), _MM_PERM_ABCD)); // acdb eghf
    auto lo  = _mm512_gmin_pd(lh, tmp); // ↓(hfeg dbac, acdb eghf)
    auto hi  = _mm512_gmax_pd(lh, tmp); // ↑(hfeg dbac, acdb eghf)
    lh = _mm512_mask_mov_pd(lo, m0xf0, hi); // ↑ed  ↑gb  ↑ha  ↑fc ↓ha  ↓fc  ↓ed  ↓gb
                                           //   └╴x3╶┘    └╴x2╶┘   └╴x0╶┘    └╴x1╶┘
    lo = _mm512_gmin_pd(lh, _mm512_swizzle_pd(lh, _MM_SWIZ_REG_CDAB)); // ↓x3 ↓x3 ↓x2 ↓x2 ↓x0 ↓x0 ↓x1 ↓x1
    hi = _mm512_gmax_pd(lh, _mm512_swizzle_pd(lh, _MM_SWIZ_REG_CDAB)); // ↑x3 ↑x3 ↑x2 ↑x2 ↑x0 ↑x0 ↑x1 ↑x1
    lh = _mm512_mask_mov_pd(lo, m0xaa, hi); // ↑x3 ↓x3 ↑x2 ↓x2 ↑x0 ↓x0 ↑x1 ↓x1
                                            //  └──╴y3╶─┘   │   └──╴y1╶─┘   │
                                            //      └──╴y2╶─┘       └──╴y0╶─┘

    lo = _mm512_gmin_pd(lh, _mm512_swizzle_pd(lh, _MM_SWIZ_REG_BADC)); // ↓y3 ↓y2 ↓y3 ↓y2 ↓y1 ↓y0 ↓y1 ↓y0
    hi = _mm512_gmax_pd(lh, _mm512_swizzle_pd(lh, _MM_SWIZ_REG_BADC)); // ↑y3 ↑y2 ↑y3 ↑y2 ↑y1 ↑y0 ↑y1 ↑y0
    lh = _mm512_mask_mov_pd(_mm512_swizzle_pd(hi, _MM_SWIZ_REG_DACB), 0x55,
            _mm512_swizzle_pd(_mm512_swizzle_pd(lo, _MM_SWIZ_REG_CDAB), _MM_SWIZ_REG_DACB));
                                                                      // ↑y3 ↓y3 ↑y2 ↓y2 ↑y1 ↓y1 ↑y0 ↓y0
    */
}

}
}
