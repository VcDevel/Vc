/*{{{
    Copyright © 2014-2015 Matthias Kretz <kretz@kde.org>

    Permission to use, copy, modify, and distribute this software
    and its documentation for any purpose and without fee is hereby
    granted, provided that the above copyright notice appear in all
    copies and that both that the copyright notice and this
    permission notice and warranty disclaimer appear in supporting
    documentation, and that the name of the author not be used in
    advertising or publicity pertaining to distribution of the
    software without specific, written prior permission.

    The author disclaim all warranties with regard to this
    software, including all implied warranties of merchantability
    and fitness.  In no event shall the author be liable for any
    special, indirect or consequential damages or any damages
    whatsoever resulting from loss of use, data or profits, whether
    in an action of contract, negligence or other tortious action,
    arising out of or in connection with the use or performance of
    this software.

}}}*/

#include <Vc/Vc>
#include <iomanip>
#include <random>
#include <vector>
#include <algorithm>
#include "../tsc.h"

using Vc::float_v;
typedef Vc::SimdArray<double, float_v::size()> double_v;
typedef Vc::SimdArray<std::int32_t, float_v::size()> int32_v;
typedef Vc::SimdArray<std::uint32_t, float_v::size()> uint32_v;

typedef typename float_v::mask_type float_m;
typedef typename double_v::mask_type double_m;
typedef typename int32_v::mask_type int32_m;
typedef typename uint32_v::mask_type uint32_m;

namespace std
{
template <typename T>
struct less<Vc::Vector<T>>
    : public binary_function<Vc::Vector<T>, Vc::Vector<T>, Vc::Mask<T>>
{
    Vc::Mask<T> operator()(const Vc::Vector<T> &x, const Vc::Vector<T> &y) const
    {
        return x < y;
    }
};

template <typename T, std::size_t N>
struct less<Vc::SimdArray<T, N>>
    : public binary_function<Vc::SimdArray<T, N>, Vc::SimdArray<T, N>,
                             typename Vc::SimdArray<T, N>::mask_type>
{
    typename Vc::SimdArray<T, N>::mask_type operator()(const Vc::SimdArray<T, N> &x,
                                                       const Vc::SimdArray<T, N> &y) const
    {
        return x < y;
    }
};
}  // namespace std

namespace Vc_VERSIONED_NAMESPACE
{
template <class Iterator, class T, class C>
inline Iterator lower_bound(Iterator first, Iterator last,
                                   const T &value, C comp)
{
    typedef simdize<T> V;
    constexpr auto Size = V::size();
    typedef simdize<C, Size> CV;
    CV comp_v;

#if 1
    const V value_v = value;
    while (first < last) {
        const auto l2 = (std::size_t(last - first) / 2) & ~(Size - 1);
        const auto m = first + l2;
        const auto mask = comp_v(V(&*m, Vc::Aligned), value_v);
        if (all_of(mask)) {
            first = m + Size;
        } else if (none_of(mask)) {
            last = m;
        } else {
            return m + mask.count();
        }
    }
#else
    auto len = std::distance(first, last);
    while (len >= Size) {
        const auto l2 = len / 2;
        Iterator m = first;
        std::advance(m, l2);
        if (comp(*m, value)) {
            first = ++m;
            len -= l2 + 1;
        } else {
            len = l2;
        }
    }
    const auto mask = comp_v(V(&*first), value);
    std::advance(first, all_of(mask) ? Size : (!mask).firstOne());
#endif
    return first;
}

template <class Iterator, class T>
inline Iterator find(Iterator first, Iterator last, const T &value)
{
    typename simdize<Iterator>::value_type value_v = value;
    for (simdize<Iterator> it = first; it < last; ++it) {
        const auto mask = *it == value_v;
        if (any_of(mask)) {
            return static_cast<Iterator>(it) + mask.firstOne();
        }
    }
    return last;
}

template <class Iterator, class V>
inline std::array<Iterator, V::size()> find_parallel(Iterator first, Iterator last,
                                                     const V &value)
{
    std::array<Iterator, V::size()> matches;
    for (auto &x : matches) {
        x = last;
    }
    typename V::mask_type found(false);
    for (; first < last; ++first) {
        const auto mask = *first == value && !found;
        if (any_of(mask)) {
            found |= mask;
            for (std::size_t i : where(mask)) {
                matches[i] = first;
            }
            if (all_of(found)) {
                break;
            }
        }
    }
    return matches;
}
}  // namespace Vc

template <typename _InputIterator, typename _Tp>
inline _InputIterator simple_find(_InputIterator first, _InputIterator last,
                           const _Tp &val) {
  for (; first != last; ++first) {
    if (*first == val) {
      break;
    }
  }
  return first;
}

int Vc_CDECL main()
{
    std::cout << std::setw(15) << "N";
    std::cout << std::setw(15) << "std" << std::setw(15) << "stddev";
    std::cout << std::setw(15) << "Vc" << std::setw(15) << "stddev";
    std::cout << std::setw(15) << "par" << std::setw(15) << "stddev";
    //std::cout << std::setw(15) << "binary" << std::setw(15) << "stddev";
    std::cout << std::setw(15) << "std/Vc" << std::setw(15) << "stddev";
    std::cout << std::setw(15) << "std/par" << std::setw(15) << "stddev" << '\n';

    // create data
    std::vector<float, Vc::Allocator<float>> data;
    constexpr std::size_t NMax = 1024 * 128 * float_v::size();
    data.reserve(NMax);
    std::default_random_engine rne;
    std::uniform_real_distribution<float> uniform_dist(-1000.f, 1000.f);
    for (auto n = data.capacity(); n > 0; --n) {
        data.push_back(uniform_dist(rne));
    }

    auto sorted = data;
    std::sort(sorted.begin(), sorted.end());

    for (std::size_t N = float_v::size() * 2; N <= NMax; N *= 2) {
        const std::size_t Repetitions = 100 + 1024 * 32 / N;

        // create search values
        std::vector<float> search_values;
        search_values.reserve(10000);
        for (auto n = search_values.capacity(); n > 0; --n) {
            search_values.push_back(
                data[std::uniform_int_distribution<std::size_t>(0, N - 1)(rne)]);
        }

        enum { std, vec, par };

        std::vector<decltype(data.begin())> iterators[3];
        iterators[std].resize(search_values.size());
        iterators[vec].resize(search_values.size());
        iterators[par].resize(search_values.size());

        double mean[3] = {};
        double stddev[3] = {};
        do {
            for (int i : {std, vec, par}) {
                mean[i] = 0;
                stddev[i] = 0;
            }
            TimeStampCounter tsc;

            // search (std)
            for (auto n = Repetitions; n; --n) {
                tsc.start();
                for (std::size_t i = 0; i < search_values.size(); ++i) {
                    iterators[std][i] =
                        simple_find(data.begin(), data.begin() + N, search_values[i]);
                }
                tsc.stop();
                double x = tsc.cycles();
                mean[std] += x;
                stddev[std] += x * x;
            }

            // search (vec)
            for (auto n = Repetitions; n; --n) {
                tsc.start();
                for (std::size_t i = 0; i < search_values.size(); ++i) {
                    iterators[vec][i] =
                        Vc::find(data.begin(), data.begin() + N, search_values[i]);
                }
                tsc.stop();
                double x = tsc.cycles();
                mean[vec] += x;
                stddev[vec] += x * x;
            }

            // seach (par)
            for (auto n = Repetitions; n; --n) {
                tsc.start();
                for (std::size_t i = 0; i < search_values.size();) {
                    for (const auto &it :
                         Vc::find_parallel(data.begin(), data.begin() + N,
                                           float_v(&search_values[i]))) {
                        iterators[par][i++] = it;
                    }
                }
                tsc.stop();
                double x = tsc.cycles();
                mean[par] += x;
                stddev[par] += x * x;
            }

            // search (bin)
            /*for (auto n = Repetitions; n; --n) {
                tsc.start();
                for (std::size_t i = 0; i < search_values.size(); ++i) {
                    iterators[bin][i] =
                        std::lower_bound(data.begin(), data.begin() + N, search_values[i]);
                }
                tsc.stop();
                double x = tsc.cycles();
                mean[bin] += x;
                stddev[bin] += x * x;
            }*/

            // test that the results are equal
            for (std::size_t i = 0; i < iterators[vec].size(); ++i) {
                assert(iterators[std][i] == iterators[vec][i]);
                assert(iterators[std][i] == iterators[par][i]);
            }

            // output results
            std::cout << std::setw(15) << N;
            for (int i : {std, vec, par}) {
                mean[i] /= Repetitions;
                stddev[i] /= Repetitions;
                stddev[i] = std::sqrt(stddev[i] - mean[i] * mean[i]);

                std::cout << std::setw(15) << mean[i] / search_values.size();
                std::cout << std::setw(15) << stddev[i] / search_values.size();
            }
            std::cout << std::setw(15) << std::setprecision(4) << mean[std] / mean[vec];
            std::cout << std::setw(15)
                      << mean[std] / mean[vec] *
                             std::sqrt(
                                 stddev[std] * stddev[std] / (mean[std] * mean[std]) +
                                 stddev[vec] * stddev[vec] / (mean[vec] * mean[vec]));
            std::cout << std::setw(15) << std::setprecision(4) << mean[std] / mean[par];
            std::cout << std::setw(15)
                      << mean[std] / mean[par] *
                             std::sqrt(
                                 stddev[std] * stddev[std] / (mean[std] * mean[std]) +
                                 stddev[par] * stddev[par] / (mean[par] * mean[par]));
            std::cout << std::endl;
        } while (stddev[std] * 20 > mean[std] || stddev[vec] * 20 > mean[vec] ||
                 stddev[par] * 20 > mean[par]);
    }

    return 0;
}
