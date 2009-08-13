/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#include "casts.h"
#include <cstdlib>

// when compiling with optimizations the compiler can use an int parameter as an immediate. When
// compiling without optimizations then the parameter has to be used either as register or memory
// location.
#if defined(NO_OPTIMIZATION) || defined(__INTEL_COMPILER)
#define IMM "r"
#else
#define IMM "n"
#endif

namespace SSE
{
    struct GeneralHelpers
    {
        template<typename Base, typename IndexType, typename EntryType>
        static inline void maskedGatherStructHelper(
                Base &v, const IndexType &indexes, int mask, const EntryType *baseAddr, const int scale
                ) {
#if defined(__GNUC__) && defined (__x86_64__)
            if (sizeof(EntryType) == 2) {
                register unsigned long int bit;
                register unsigned long int index;
                register EntryType value;
                asm volatile(
                        "bsf %1,%0"            "\n\t"
                        "jz 1f"                "\n\t"
                        "0:"                   "\n\t"
                        "movzwl (%5,%0,2),%%ecx""\n\t"
                        "btr %0,%1"            "\n\t"
                        "imul %8,%%ecx"        "\n\t"
                        "movw (%6,%%rcx,1),%3" "\n\t"
                        "movw %3,(%7,%0,2)"    "\n\t"
                        "bsf %1,%0"            "\n\t"
                        "jnz 0b"               "\n\t"
                        "1:"                   "\n\t"
                        : "=&r"(bit), "+r"(mask), "=&r"(index), "=&r"(value), "+m"(v.d)
                        : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), IMM(scale), "m"(indexes.d.v())
                        : "rcx"   );
            } else if (sizeof(EntryType) == 4) {
                if (sizeof(typename IndexType::EntryType) == 4) {
                    register unsigned long int bit;
                    register unsigned long int index;
                    register EntryType value;
                    asm volatile(
                            "bsf %1,%0"            "\n\t"
                            "jz 1f"                "\n\t"
                            "0:"                   "\n\t"
                            "mov (%5,%0,4),%%ecx"  "\n\t"
                            "btr %0,%1"            "\n\t"
                            "imul %8,%%ecx"        "\n\t"
                            "mov (%6,%%rcx,1),%3"  "\n\t"
                            "mov %3,(%7,%0,4)"     "\n\t"
                            "bsf %1,%0"            "\n\t"
                            "jnz 0b"               "\n\t"
                            "1:"                   "\n\t"
                            : "=&r"(bit), "+r"(mask), "=&r"(index), "=&r"(value), "+m"(v.d)
                            : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), IMM(scale), "m"(indexes.d.v())
                            : "rcx"   );
                } else if (sizeof(typename IndexType::EntryType) == 2) {
                    register unsigned long int bit;
                    register unsigned long int index;
                    register EntryType value;
                    asm volatile(
                            "bsf %1,%0"            "\n\t"
                            "jz 1f"                "\n\t"
                            "0:"                   "\n\t"
                            "movzwl (%5,%0,2),%%ecx""\n\t"
                            "btr %0,%1"            "\n\t"
                            "imul %8,%%ecx"        "\n\t"
                            "mov (%6,%%rcx,1),%3"  "\n\t"
                            "mov %3,(%7,%0,4)"     "\n\t"
                            "bsf %1,%0"            "\n\t"
                            "jnz 0b"               "\n\t"
                            "1:"                   "\n\t"
                            : "=&r"(bit), "+r"(mask), "=&r"(index), "=&r"(value), "+m"(v.d)
                            : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), IMM(scale), "m"(indexes.d.v())
                            : "rcx"   );
                } else {
                    abort();
                }
            } else if (sizeof(EntryType) == 8) {
                register unsigned long int bit;
                register unsigned long int index;
                register EntryType value;
                asm volatile(
                        "bsf %1,%0"            "\n\t"
                        "jz 1f"                "\n\t"
                        "0:"                   "\n\t"
                        "mov (%5,%0,4),%%ecx"  "\n\t"
                        "btr %0,%1"            "\n\t"
                        "imul %8,%%ecx"        "\n\t"
                        "mov (%6,%%rcx,1),%3"  "\n\t"
                        "mov %3,(%7,%0,8)"     "\n\t"
                        "bsf %1,%0"            "\n\t"
                        "jnz 0b"               "\n\t"
                        "1:"                   "\n\t"
                        : "=&r"(bit), "+r"(mask), "=&r"(index), "=&r"(value), "+m"(v.d)
                        : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), IMM(scale), "m"(indexes.d.v())
                        : "rcx"   );
            } else {
                abort();
            }
#elif defined(_MSC_VER) || !defined(__x86_64__)
            unrolled_loop16(i, 0, Base::Size,
                    EntryType entry = baseAddr[scale / sizeof(EntryType) * indexes.d.m(i)];
                    register EntryType tmp = v.d.m(i);
                    if (mask & (1 << i)) tmp = entry;
                    v.d.m(i) = tmp;
                    );
#else
#error "Check whether inline asm works, or use else clause"
#endif
        }

        template<typename Base, typename IndexType, typename EntryType>
        static inline void maskedGatherHelper(
                Base &v, const IndexType &indexes, int mask, const EntryType *baseAddr
                ) {
#if defined(__GNUC__) && defined(__x86_64__)
            if (sizeof(EntryType) == 2) {
                register unsigned long int bit;
                register unsigned long int index;
                register EntryType value;
                asm volatile(
                        "bsf %1,%0"            "\n\t"
                        "jz 1f"                "\n\t"
                        "0:"                   "\n\t"
                        "movzwl (%5,%0,2),%%ecx""\n\t"
                        "btr %0,%1"            "\n\t"
                        "movw (%6,%%rcx,2),%3" "\n\t"
                        "movw %3,(%7,%0,2)"    "\n\t"
                        "bsf %1,%0"            "\n\t"
                        "jnz 0b"               "\n\t"
                        "1:"                   "\n\t"
                        : "=&r"(bit), "+r"(mask), "=&r"(index), "=&r"(value), "+m"(v.d)
                        : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), "m"(indexes.d.v())
                        : "rcx"   );
            } else if (sizeof(EntryType) == 4) {
                if (sizeof(typename IndexType::EntryType) == 4) {
                    register unsigned long int bit;
                    register unsigned long int index;
                    register EntryType value;
                    asm volatile(
                            "bsf %1,%0"            "\n\t"
                            "jz 1f"                "\n\t"
                            "0:"                   "\n\t"
                            "mov (%5,%0,4),%%ecx"  "\n\t"
                            "btr %0,%1"            "\n\t"
                            "mov (%6,%%rcx,4),%3"  "\n\t"
                            "mov %3,(%7,%0,4)"     "\n\t"
                            "bsf %1,%0"            "\n\t"
                            "jnz 0b"               "\n\t"
                            "1:"                   "\n\t"
                            : "=&r"(bit), "+r"(mask), "=&r"(index), "=&r"(value), "+m"(v.d)
                            : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), "m"(indexes.d.v())
                            : "rcx"   );
                } else if (sizeof(typename IndexType::EntryType) == 2) {
                    register unsigned long int bit;
                    register unsigned long int index;
                    register EntryType value;
                    asm volatile(
                            "bsf %1,%0"            "\n\t"
                            "jz 1f"                "\n\t"
                            "0:"                   "\n\t"
                            "movzwl (%5,%0,2),%%ecx""\n\t"
                            "btr %0,%1"            "\n\t"
                            "mov (%6,%%rcx,4),%3"  "\n\t"
                            "mov %3,(%7,%0,4)"     "\n\t"
                            "bsf %1,%0"            "\n\t"
                            "jnz 0b"               "\n\t"
                            "1:"                   "\n\t"
                            : "=&r"(bit), "+r"(mask), "=&r"(index), "=&r"(value), "+m"(v.d)
                            : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), "m"(indexes.d.v())
                            : "rcx"   );
                } else {
                    abort();
                }
            } else if (sizeof(EntryType) == 8) {
                register unsigned long int bit;
                register unsigned long int index;
                register EntryType value;
                asm volatile(
                        "bsf %1,%0"            "\n\t"
                        "jz 1f"                "\n\t"
                        "0:"                   "\n\t"
                        "mov (%5,%0,4),%%ecx"  "\n\t"
                        "btr %0,%1"            "\n\t"
                        "mov (%6,%%rcx,8),%3"  "\n\t"
                        "mov %3,(%7,%0,8)"     "\n\t"
                        "bsf %1,%0"            "\n\t"
                        "jnz 0b"               "\n\t"
                        "1:"                   "\n\t"
                        : "=&r"(bit), "+r"(mask), "=&r"(index), "=&r"(value), "+m"(v.d)
                        : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), "m"(indexes.d.v())
                        : "rcx"   );
            } else {
                abort();
            }
#elif defined(_MSC_VER) || !defined(__x86_64__)
            unrolled_loop16(i, 0, Base::Size,
                    EntryType entry = baseAddr[indexes.d.m(i)];
                    register EntryType tmp = v.d.m(i);
                    if (mask & (1 << i)) tmp = entry;
                    v.d.m(i) = tmp;
                    );
#else
#error "Check whether inline asm works, or use else clause"
#endif
        }

        template<typename AliasingT, typename EntryType>
        static inline void maskedScatterHelper(
                const AliasingT &vEntry, const int mask, EntryType &value, const int bitMask
                ) {
#ifdef __GNUC__
            register EntryType t;
            asm(
                    "test %4,%2\n\t"
                    "mov %3,%1\n\t"
                    "cmovne %5,%1\n\t"
                    "mov %1,%0"
                    : "=m"(value), "=&r"(t)
                    : "r"(mask), "m"(value), IMM(bitMask), "m"(vEntry)
               );
#else
            if (mask & bitMask) {
                value = vEntry;
            }
#endif
        }

    };

    ////////////////////////////////////////////////////////
    // Array gathers
    template<typename T> inline void GatherHelper<T>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        for_all_vector_entries(i,
                v.d.m(i) = baseAddr[indexes.d.m(i)];
                );
    }
    template<> inline void GatherHelper<double>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_pd(baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]);
    }
    template<> inline void GatherHelper<float>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_ps(
                baseAddr[indexes.d.m(3)], baseAddr[indexes.d.m(2)],
                baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]);
    }
    template<> inline void GatherHelper<float8>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        v.d.v()[1] = _mm_set_ps(
                baseAddr[indexes.d.m(7)], baseAddr[indexes.d.m(6)],
                baseAddr[indexes.d.m(5)], baseAddr[indexes.d.m(4)]);
        v.d.v()[0] = _mm_set_ps(
                baseAddr[indexes.d.m(3)], baseAddr[indexes.d.m(2)],
                baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]);
    }
    template<> inline void GatherHelper<int>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_epi32(
                baseAddr[indexes.d.m(3)], baseAddr[indexes.d.m(2)],
                baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]);
    }
    template<> inline void GatherHelper<unsigned int>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_epi32(
                baseAddr[indexes.d.m(3)], baseAddr[indexes.d.m(2)],
                baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]);
    }
    template<> inline void GatherHelper<short>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_epi16(
                baseAddr[indexes.d.m(7)], baseAddr[indexes.d.m(6)],
                baseAddr[indexes.d.m(5)], baseAddr[indexes.d.m(4)],
                baseAddr[indexes.d.m(3)], baseAddr[indexes.d.m(2)],
                baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]);
    }
    template<> inline void GatherHelper<unsigned short>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_epi16(
                baseAddr[indexes.d.m(7)], baseAddr[indexes.d.m(6)],
                baseAddr[indexes.d.m(5)], baseAddr[indexes.d.m(4)],
                baseAddr[indexes.d.m(3)], baseAddr[indexes.d.m(2)],
                baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]);
    }

    ////////////////////////////////////////////////////////
    // Struct gathers
    template<typename T> template<typename S1> inline void GatherHelper<T>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        for_all_vector_entries(i,
                v.d.m(i) = baseAddr[indexes.d.m(i)].*(member1);
                );
    }
    template<> template<typename S1> inline void GatherHelper<double>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        v.d.v() = _mm_set_pd(baseAddr[indexes.d.m(1)].*(member1), baseAddr[indexes.d.m(0)].*(member1));
    }
    template<> template<typename S1> inline void GatherHelper<float>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        v.d.v() = _mm_set_ps(
                baseAddr[indexes.d.m(3)].*(member1), baseAddr[indexes.d.m(2)].*(member1),
                baseAddr[indexes.d.m(1)].*(member1), baseAddr[indexes.d.m(0)].*(member1));
    }
    template<> template<typename S1> inline void GatherHelper<float8>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        v.d.v()[1] = _mm_set_ps(
                baseAddr[indexes.d.m(7)].*(member1), baseAddr[indexes.d.m(6)].*(member1),
                baseAddr[indexes.d.m(5)].*(member1), baseAddr[indexes.d.m(4)].*(member1));
        v.d.v()[0] = _mm_set_ps(
                baseAddr[indexes.d.m(3)].*(member1), baseAddr[indexes.d.m(2)].*(member1),
                baseAddr[indexes.d.m(1)].*(member1), baseAddr[indexes.d.m(0)].*(member1));
    }
    template<> template<typename S1> inline void GatherHelper<int>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        v.d.v() = _mm_set_epi32(
                baseAddr[indexes.d.m(3)].*(member1), baseAddr[indexes.d.m(2)].*(member1),
                baseAddr[indexes.d.m(1)].*(member1), baseAddr[indexes.d.m(0)].*(member1));
    }
    template<> template<typename S1> inline void GatherHelper<unsigned int>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        v.d.v() = _mm_set_epi32(
                baseAddr[indexes.d.m(3)].*(member1), baseAddr[indexes.d.m(2)].*(member1),
                baseAddr[indexes.d.m(1)].*(member1), baseAddr[indexes.d.m(0)].*(member1));
    }
    template<> template<typename S1> inline void GatherHelper<short>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        v.d.v() = _mm_set_epi16(
                baseAddr[indexes.d.m(7)].*(member1), baseAddr[indexes.d.m(6)].*(member1),
                baseAddr[indexes.d.m(5)].*(member1), baseAddr[indexes.d.m(4)].*(member1),
                baseAddr[indexes.d.m(3)].*(member1), baseAddr[indexes.d.m(2)].*(member1),
                baseAddr[indexes.d.m(1)].*(member1), baseAddr[indexes.d.m(0)].*(member1));
    }
    template<> template<typename S1> inline void GatherHelper<unsigned short>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        v.d.v() = _mm_set_epi16(
                baseAddr[indexes.d.m(7)].*(member1), baseAddr[indexes.d.m(6)].*(member1),
                baseAddr[indexes.d.m(5)].*(member1), baseAddr[indexes.d.m(4)].*(member1),
                baseAddr[indexes.d.m(3)].*(member1), baseAddr[indexes.d.m(2)].*(member1),
                baseAddr[indexes.d.m(1)].*(member1), baseAddr[indexes.d.m(0)].*(member1));
    }

    ////////////////////////////////////////////////////////
    // Struct of Struct gathers
    template<typename T> template<typename S1, typename S2> inline void GatherHelper<T>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        for_all_vector_entries(i,
                v.d.m(i) = baseAddr[indexes.d.m(i)].*(member1).*(member2);
                );
    }
    template<> template<typename S1, typename S2> inline void GatherHelper<double>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        v.d.v() = _mm_set_pd(baseAddr[indexes.d.m(1)].*(member1).*(member2), baseAddr[indexes.d.m(0)].*(member1).*(member2));
    }
    template<> template<typename S1, typename S2> inline void GatherHelper<float>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        v.d.v() = _mm_set_ps(
                baseAddr[indexes.d.m(3)].*(member1).*(member2), baseAddr[indexes.d.m(2)].*(member1).*(member2),
                baseAddr[indexes.d.m(1)].*(member1).*(member2), baseAddr[indexes.d.m(0)].*(member1).*(member2));
    }
    template<> template<typename S1, typename S2> inline void GatherHelper<float8>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        v.d.v()[1] = _mm_set_ps(
                baseAddr[indexes.d.m(7)].*(member1).*(member2), baseAddr[indexes.d.m(6)].*(member1).*(member2),
                baseAddr[indexes.d.m(5)].*(member1).*(member2), baseAddr[indexes.d.m(4)].*(member1).*(member2));
        v.d.v()[0] = _mm_set_ps(
                baseAddr[indexes.d.m(3)].*(member1).*(member2), baseAddr[indexes.d.m(2)].*(member1).*(member2),
                baseAddr[indexes.d.m(1)].*(member1).*(member2), baseAddr[indexes.d.m(0)].*(member1).*(member2));
    }
    template<> template<typename S1, typename S2> inline void GatherHelper<int>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        v.d.v() = _mm_set_epi32(
                baseAddr[indexes.d.m(3)].*(member1).*(member2), baseAddr[indexes.d.m(2)].*(member1).*(member2),
                baseAddr[indexes.d.m(1)].*(member1).*(member2), baseAddr[indexes.d.m(0)].*(member1).*(member2));
    }
    template<> template<typename S1, typename S2> inline void GatherHelper<unsigned int>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        v.d.v() = _mm_set_epi32(
                baseAddr[indexes.d.m(3)].*(member1).*(member2), baseAddr[indexes.d.m(2)].*(member1).*(member2),
                baseAddr[indexes.d.m(1)].*(member1).*(member2), baseAddr[indexes.d.m(0)].*(member1).*(member2));
    }
    template<> template<typename S1, typename S2> inline void GatherHelper<short>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        v.d.v() = _mm_set_epi16(
                baseAddr[indexes.d.m(7)].*(member1).*(member2), baseAddr[indexes.d.m(6)].*(member1).*(member2),
                baseAddr[indexes.d.m(5)].*(member1).*(member2), baseAddr[indexes.d.m(4)].*(member1).*(member2),
                baseAddr[indexes.d.m(3)].*(member1).*(member2), baseAddr[indexes.d.m(2)].*(member1).*(member2),
                baseAddr[indexes.d.m(1)].*(member1).*(member2), baseAddr[indexes.d.m(0)].*(member1).*(member2));
    }
    template<> template<typename S1, typename S2> inline void GatherHelper<unsigned short>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        v.d.v() = _mm_set_epi16(
                baseAddr[indexes.d.m(7)].*(member1).*(member2), baseAddr[indexes.d.m(6)].*(member1).*(member2),
                baseAddr[indexes.d.m(5)].*(member1).*(member2), baseAddr[indexes.d.m(4)].*(member1).*(member2),
                baseAddr[indexes.d.m(3)].*(member1).*(member2), baseAddr[indexes.d.m(2)].*(member1).*(member2),
                baseAddr[indexes.d.m(1)].*(member1).*(member2), baseAddr[indexes.d.m(0)].*(member1).*(member2));
    }

    ////////////////////////////////////////////////////////
    // Scatters
    //
    // There is no equivalent to the set intrinsics. Therefore the vector entries are copied in
    // memory instead from the xmm register directly.
    //
    // TODO: With SSE 4.1 the extract intrinsics might be an interesting option, though.
    //
    template<typename T> inline void VectorHelperSize<T>::scatter(
            const Base &v, const IndexType &indexes, EntryType *baseAddr) {
        for_all_vector_entries(i,
                baseAddr[indexes.d.m(i)] = v.d.m(i);
                );
    }
    template<> inline void VectorHelperSize<short>::scatter(
            const Base &v, const IndexType &indexes, EntryType *baseAddr) {
        // TODO: verify that using extract is really faster
        for_all_vector_entries(i,
                baseAddr[indexes.d.m(i)] = _mm_extract_epi16(v.d.v(), i);
                );
    }

    template<typename T> inline void VectorHelperSize<T>::scatter(
            const Base &v, const IndexType &indexes, int mask, EntryType *baseAddr) {
        for_all_vector_entries(i,
                GeneralHelpers::maskedScatterHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)], 1 << i * Shift);
                );
    }

    template<typename T> template<typename S1> inline void VectorHelperSize<T>::scatter(
            const Base &v, const IndexType &indexes, S1 *baseAddr, EntryType S1::* member1) {
        for_all_vector_entries(i,
                baseAddr[indexes.d.m(i)].*(member1) = v.d.m(i);
                );
    }

    template<typename T> template<typename S1> inline void VectorHelperSize<T>::scatter(
            const Base &v, const IndexType &indexes, int mask, S1 *baseAddr, EntryType S1::* member1) {
        for_all_vector_entries(i,
                GeneralHelpers::maskedScatterHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)].*(member1), 1 << i * Shift);
                );
    }

    template<typename T> template<typename S1, typename S2> inline void VectorHelperSize<T>::scatter(
            const Base &v, const IndexType &indexes, S1 *baseAddr, S2 S1::* member1, EntryType S2::* member2) {
        for_all_vector_entries(i,
                baseAddr[indexes.d.m(i)].*(member1).*(member2) = v.d.m(i);
                );
    }

    template<typename T> template<typename S1, typename S2> inline void VectorHelperSize<T>::scatter(
            const Base &v, const IndexType &indexes, int mask, S1 *baseAddr, S2 S1::* member1,
            EntryType S2::* member2) {
        for_all_vector_entries(i,
                GeneralHelpers::maskedScatterHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)].*(member1).*(member2), 1 << i * Shift);
                );
    }

    inline void VectorHelperSize<float8>::scatter(const Base &v, const IndexType &indexes, EntryType *baseAddr) {
        for_all_vector_entries(i,
                baseAddr[indexes.d.m(i)] = v.d.m(i);
                );
    }

    inline void VectorHelperSize<float8>::scatter(const Base &v, const IndexType &indexes, int mask, EntryType *baseAddr) {
        for_all_vector_entries(i,
                GeneralHelpers::maskedScatterHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)], 1 << i * Shift);
                );
    }

    template<typename S1> inline void VectorHelperSize<float8>::scatter(const Base &v, const IndexType &indexes,
            S1 *baseAddr, EntryType S1::* member1) {
        for_all_vector_entries(i,
                baseAddr[indexes.d.m(i)].*(member1) = v.d.m(i);
                );
    }

    template<typename S1> inline void VectorHelperSize<float8>::scatter(const Base &v, const IndexType &indexes, int mask,
            S1 *baseAddr, EntryType S1::* member1) {
        for_all_vector_entries(i,
                GeneralHelpers::maskedScatterHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)].*(member1), 1 << i * Shift);
                );
    }

    template<typename S1, typename S2> inline void VectorHelperSize<float8>::scatter(const Base &v, const IndexType &indexes,
            S1 *baseAddr, S2 S1::* member1, EntryType S2::* member2) {
        for_all_vector_entries(i,
                baseAddr[indexes.d.m(i)].*(member1).*(member2) = v.d.m(i);
                );
    }

    template<typename S1, typename S2> inline void VectorHelperSize<float8>::scatter(const Base &v, const IndexType &indexes, int mask,
            S1 *baseAddr, S2 S1::* member1, EntryType S2::* member2) {
        for_all_vector_entries(i,
                GeneralHelpers::maskedScatterHelper(v.d.m(i), mask, baseAddr[indexes.d.m(i)].*(member1).*(member2), 1 << i * Shift);
                );
    }

    // can be used to multiply with a constant. For some special constants it doesn't need an extra
    // vector but can use a shift instead, basically encoding the factor in the instruction.
    template<typename IndexType, unsigned int constant> inline IndexType mulConst(const IndexType &x) {
        typedef VectorHelper<typename IndexType::EntryType> H;
        switch (constant) {
            case    0: return H::zero();
            case    1: return x;
            case    2: return H::slli(x.data(),  1);
            case    4: return H::slli(x.data(),  2);
            case    8: return H::slli(x.data(),  3);
            case   16: return H::slli(x.data(),  4);
            case   32: return H::slli(x.data(),  5);
            case   64: return H::slli(x.data(),  6);
            case  128: return H::slli(x.data(),  7);
            case  256: return H::slli(x.data(),  8);
            case  512: return H::slli(x.data(),  9);
            case 1024: return H::slli(x.data(), 10);
            case 2048: return H::slli(x.data(), 11);
        }
#ifndef __SSE4_1__
        // without SSE 4.1 int multiplication is not so nice
        if (sizeof(typename IndexType::EntryType) == 4) {
            switch (constant) {
                case    3: return H::add(        x.data()    , H::slli(x.data(),  1));
                case    5: return H::add(        x.data()    , H::slli(x.data(),  2));
                case    9: return H::add(        x.data()    , H::slli(x.data(),  3));
                case   17: return H::add(        x.data()    , H::slli(x.data(),  4));
                case   33: return H::add(        x.data()    , H::slli(x.data(),  5));
                case   65: return H::add(        x.data()    , H::slli(x.data(),  6));
                case  129: return H::add(        x.data()    , H::slli(x.data(),  7));
                case  257: return H::add(        x.data()    , H::slli(x.data(),  8));
                case  513: return H::add(        x.data()    , H::slli(x.data(),  9));
                case 1025: return H::add(        x.data()    , H::slli(x.data(), 10));
                case 2049: return H::add(        x.data()    , H::slli(x.data(), 11));
                case    6: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  2));
                case   10: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  3));
                case   18: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  4));
                case   34: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  5));
                case   66: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  6));
                case  130: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  7));
                case  258: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  8));
                case  514: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  9));
                case 1026: return H::add(H::slli(x.data(), 1), H::slli(x.data(), 10));
                case 2050: return H::add(H::slli(x.data(), 1), H::slli(x.data(), 11));
                case   12: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  3));
                case   20: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  4));
                case   36: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  5));
                case   68: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  6));
                case  132: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  7));
                case  260: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  8));
                case  516: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  9));
                case 1028: return H::add(H::slli(x.data(), 2), H::slli(x.data(), 10));
                case 2052: return H::add(H::slli(x.data(), 2), H::slli(x.data(), 11));
                case   24: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  4));
                case   40: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  5));
                case   72: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  6));
                case  136: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  7));
                case  264: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  8));
                case  520: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  9));
                case 1032: return H::add(H::slli(x.data(), 3), H::slli(x.data(), 10));
                case 2056: return H::add(H::slli(x.data(), 3), H::slli(x.data(), 11));
                case   48: return H::add(H::slli(x.data(), 4), H::slli(x.data(),  5));
                case   80: return H::add(H::slli(x.data(), 4), H::slli(x.data(),  6));
                case  144: return H::add(H::slli(x.data(), 4), H::slli(x.data(),  7));
                case  272: return H::add(H::slli(x.data(), 4), H::slli(x.data(),  8));
                case  528: return H::add(H::slli(x.data(), 4), H::slli(x.data(),  9));
                case 1040: return H::add(H::slli(x.data(), 4), H::slli(x.data(), 10));
                case 2064: return H::add(H::slli(x.data(), 4), H::slli(x.data(), 11));
                case   96: return H::add(H::slli(x.data(), 5), H::slli(x.data(),  6));
                case  160: return H::add(H::slli(x.data(), 5), H::slli(x.data(),  7));
                case  288: return H::add(H::slli(x.data(), 5), H::slli(x.data(),  8));
                case  544: return H::add(H::slli(x.data(), 5), H::slli(x.data(),  9));
                case 1056: return H::add(H::slli(x.data(), 5), H::slli(x.data(), 10));
                case 2080: return H::add(H::slli(x.data(), 5), H::slli(x.data(), 11));
                case  192: return H::add(H::slli(x.data(), 6), H::slli(x.data(),  7));
                case  320: return H::add(H::slli(x.data(), 6), H::slli(x.data(),  8));
                case  576: return H::add(H::slli(x.data(), 6), H::slli(x.data(),  9));
                case 1088: return H::add(H::slli(x.data(), 6), H::slli(x.data(), 10));
                case 2112: return H::add(H::slli(x.data(), 6), H::slli(x.data(), 11));
                case  384: return H::add(H::slli(x.data(), 7), H::slli(x.data(),  8));
                case  640: return H::add(H::slli(x.data(), 7), H::slli(x.data(),  9));
                case 1152: return H::add(H::slli(x.data(), 7), H::slli(x.data(), 10));
                case 2176: return H::add(H::slli(x.data(), 7), H::slli(x.data(), 11));
                case  768: return H::add(H::slli(x.data(), 8), H::slli(x.data(),  9));
                case 1280: return H::add(H::slli(x.data(), 8), H::slli(x.data(), 10));
                case 2304: return H::add(H::slli(x.data(), 8), H::slli(x.data(), 11));
                case 1536: return H::add(H::slli(x.data(), 9), H::slli(x.data(), 10));
                case 2560: return H::add(H::slli(x.data(), 9), H::slli(x.data(), 11));
                case 3072: return H::add(H::slli(x.data(),10), H::slli(x.data(), 11));
            }
        }
#endif
        return H::mul(x.data(), H::set(constant));
    }
} // namespace SSE

#undef IMM
