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

namespace SSE
{
    struct GeneralHelpers
    {
        template<typename VectorType, typename EntryType> static inline VectorType set4(
                const EntryType *m, const unsigned long a, const unsigned long b,
                const unsigned long c, const unsigned long d
                ) {
            VectorType v;
            __m128 t1, t2, t3;
            __asm__("movd 0(%4,%5,4), %3\n\t"
                    "movd 0(%4,%6,4), %2\n\t"
                    "movd 0(%4,%7,4), %1\n\t"
                    "movd 0(%4,%8,4), %0\n\t"
                    "unpcklps %3, %2\n\t"
                    "unpcklps %1, %0\n\t"
                    "movlhps %2, %0\n\t"
                    : "=x"(v), "=x"(t1), "=x"(t2), "=x"(t3)
                    : "r"(m), "r"(a), "r"(b), "r"(c), "r"(d)
                   );
            return v;
        }

        template<typename Base, typename IndexType, typename EntryType>
        static inline void maskedGatherStructHelper(
                Base &v, const IndexType &indexes, int mask, const EntryType *baseAddr, const int &scale
                ) {
#ifdef __GNUC__
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
                        : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), "i"(scale)
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
                            : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), "i"(scale)
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
                            : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), "i"(scale)
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
                        : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), "i"(scale)
                        : "rcx"   );
            } else {
                abort();
            }
#else
#error "Check whether inline asm works, or fix else clause"
            for_all_vector_entries(i,
                    if (mask & (1 << i * Shift)) v.d.m(i) = baseAddr[indexes.d.m(i)];
                    );
#endif
        }

        template<typename Base, typename IndexType, typename EntryType>
        static inline void maskedGatherHelper(
                Base &v, const IndexType &indexes, int mask, const EntryType *baseAddr
                ) {
#ifdef __GNUC__
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
                        : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d)
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
                            : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d)
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
                            : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d)
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
                        : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d)
                        : "rcx"   );
            } else {
                abort();
            }
#else
#error "Check whether inline asm works, or fix else clause"
            for_all_vector_entries(i,
                    if (mask & (1 << i * Shift)) v.d.m(i) = baseAddr[indexes.d.m(i)];
                    );
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
                    : "r"(mask), "m"(value),
#ifdef NO_OPTIMIZATION
                    "m"
#else
                    "n"
#endif
                    (bitMask), "m"(vEntry)
               );
#else
            if (mask & bitMask) {
                value = vEntry;
            }
#endif
        }

    };

    template<typename T> inline void VectorHelperSize<T>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr
            ) {
        if (Size == 2) {
            v.d.v() = mm128_reinterpret_cast<VectorType>(_mm_set_pd(
                        baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]));
        } else if (Size == 4) {
            v.d.v() = GeneralHelpers::set4<VectorType, EntryType>(
                    baseAddr, indexes.d.m(3), indexes.d.m(2), indexes.d.m(1), indexes.d.m(0));
        } else if (Size == 8) {
            v.d.v() = mm128_reinterpret_cast<VectorType>(_mm_set_epi16(
                        baseAddr[indexes.d.m(7)], baseAddr[indexes.d.m(6)],
                        baseAddr[indexes.d.m(5)], baseAddr[indexes.d.m(4)],
                        baseAddr[indexes.d.m(3)], baseAddr[indexes.d.m(2)],
                        baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]));
        } else {
            for_all_vector_entries(i,
                    v.d.m(i) = baseAddr[indexes.d.m(i)];
                    );
        }
    }

    template<typename T> template<typename S1> inline void VectorHelperSize<T>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1) {
        for_all_vector_entries(i,
                v.d.m(i) = baseAddr[indexes.d.m(i)].*(member1);
                );
    }

    template<typename T> template<typename S1, typename S2> inline void VectorHelperSize<T>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1,
            const EntryType S2::* member2) {
        for_all_vector_entries(i,
                v.d.m(i) = baseAddr[indexes.d.m(i)].*(member1).*(member2);
                );
    }

    template<typename T> inline void VectorHelperSize<T>::scatter(
            const Base &v, const IndexType &indexes, EntryType *baseAddr) {
        for_all_vector_entries(i,
                baseAddr[indexes.d.m(i)] = v.d.m(i);
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

    inline void VectorHelperSize<float8>::gather(Base &v, const IndexType &indexes, const EntryType *baseAddr) {
        v.d.v()[0] = GeneralHelpers::set4<_M128, EntryType>(baseAddr,
                indexes.d.m(3), indexes.d.m(2), indexes.d.m(1), indexes.d.m(0));
        v.d.v()[1] = GeneralHelpers::set4<_M128, EntryType>(baseAddr,
                indexes.d.m(7), indexes.d.m(6), indexes.d.m(5), indexes.d.m(4));
    }

    template<typename S1> inline void VectorHelperSize<float8>::gather(Base &v, const IndexType &indexes,
            const S1 *baseAddr, const EntryType S1::* member1) {
        for_all_vector_entries(i,
                v.d.m(i) = baseAddr[indexes.d.m(i)].*(member1);
                );
    }

    template<typename S1, typename S2> inline void VectorHelperSize<float8>::gather(Base &v, const IndexType &indexes,
            const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2) {
        for_all_vector_entries(i,
                v.d.m(i) = baseAddr[indexes.d.m(i)].*(member1).*(member2);
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
