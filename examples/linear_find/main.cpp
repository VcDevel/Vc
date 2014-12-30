/*{{{
    Copyright © 2014 Matthias Kretz <kretz@kde.org>

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
#include "../kdtree/simdize.h"

using Vc::float_v;
typedef Vc::simdarray<double, float_v::size()> double_v;
typedef Vc::simdarray<std::int32_t, float_v::size()> int32_v;
typedef Vc::simdarray<std::uint32_t, float_v::size()> uint32_v;

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
    Vc::Mask<T> operator()(const Vc::Vector<T>& __x, const Vc::Vector<T>& __y) const
    {
        return __x < __y;
    }
};

template <typename T, std::size_t N>
struct less<Vc::simdarray<T, N>>
    : public binary_function<Vc::simdarray<T, N>, Vc::simdarray<T, N>,
                             typename Vc::simdarray<T, N>::mask_type>
{
    typename Vc::simdarray<T, N>::mask_type operator()(const Vc::simdarray<T, N> &__x,
                                                       const Vc::simdarray<T, N> &__y)
        const
    {
        return __x < __y;
    }
};
}  // namespace std

namespace Vc_0
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
            return it.scalar() + mask.firstOne();
        }
    }
    return last;
}
}  // namespace Vc

int main()
{
    std::cout << std::setw(15) << "N" << std::setw(30) << "std" << std::setw(15)
              << "stddev" << std::setw(30) << "Vc" << std::setw(15) << "stddev"
              << std::setw(15) << "speedup" << std::setw(15) << "stddev" << '\n';

    // create data
    std::vector<float, Vc::Allocator<float>> data;
    constexpr std::size_t NMax = 1024 * 128 * float_v::size();
    data.reserve(NMax);
    std::default_random_engine rne;
    std::uniform_real_distribution<float> uniform_dist(-1000.f, 1000.f);
    for (auto n = data.capacity(); n > 0; --n) {
        data.push_back(uniform_dist(rne));
    }

    for (std::size_t N = float_v::size() * 1024; N <= NMax; N *= 2) {
        const std::size_t Repetitions = 100 + 1024 * 32 / N;

        // create search values
        std::vector<float> search_values;
        search_values.reserve(10000);
        for (auto n = search_values.capacity(); n > 0; --n) {
            search_values.push_back(
                data[std::uniform_int_distribution<std::size_t>(0, N - 1)(rne)]);
        }

        enum { std, vec };

        std::vector<decltype(data.begin())> iterators[2];
        iterators[std].resize(search_values.size());
        iterators[vec].resize(search_values.size());
        TimeStampCounter tsc;
        std::vector<decltype(tsc.cycles())> cycles[2];
        cycles[std].resize(Repetitions);
        cycles[vec].resize(Repetitions);

        double mean[2] = {};
        double stddev[2] = {};
        do {
            // search (std)
            for (auto n = Repetitions; n; --n) {
                tsc.start();
                for (std::size_t i = 0; i < search_values.size(); ++i) {
                    iterators[std][i] =
                        std::find(data.begin(), data.begin() + N, search_values[i]);
                }
                tsc.stop();
                cycles[std][Repetitions - n] = tsc.cycles();
            }

            // search (vec)
            for (auto n = Repetitions; n; --n) {
                tsc.start();
                for (std::size_t i = 0; i < search_values.size(); ++i) {
                    iterators[vec][i] =
                        Vc::find(data.begin(), data.begin() + N, search_values[i]);
                }
                tsc.stop();
                cycles[vec][Repetitions - n] = tsc.cycles();
            }

            // test that the results are equal
            for (std::size_t i = 0; i < iterators[vec].size(); ++i) {
                assert(iterators[std][i] == iterators[vec][i]);
            }

            // output results
            std::cout << std::setw(15) << N;
            double median[2] = {};
            for (int i : {std, vec}) {
                mean[i] = 0;
                stddev[i] = 0;
                std::sort(cycles[i].begin(), cycles[i].end());
                median[i] = cycles[i][cycles[i].size() / 2];
                for (double x : cycles[i]) {
                    mean[i] += x;
                    stddev[i] += x * x;
                }
                mean[i] /= cycles[i].size();
                stddev[i] /= cycles[i].size();
                stddev[i] = std::sqrt(stddev[i] - mean[i] * mean[i]);
                // stddev[i] /= std::sqrt(float(cycles[i].size()));  // stddev of mean
            }
            std::cout << std::setw(15) << median[std] / search_values.size();
            std::cout << std::setw(15) << mean[std] / search_values.size();
            std::cout << std::setw(15) << stddev[std] / search_values.size();
            std::cout << std::setw(15) << median[vec] / search_values.size();
            std::cout << std::setw(15) << mean[vec] / search_values.size();
            std::cout << std::setw(15) << stddev[vec] / search_values.size();
            std::cout << std::setw(15) << std::setprecision(4) << mean[std] / mean[vec];
            std::cout << std::setw(15)
                      << mean[std] / mean[vec] *
                             std::sqrt(
                                 stddev[std] * stddev[std] / (mean[std] * mean[std]) +
                                 stddev[vec] * stddev[vec] / (mean[vec] * mean[vec]));
            std::cout << std::endl;
        } while (stddev[std] * 20 > mean[std] || stddev[vec] * 20 > mean[vec]);
    }

    return 0;
}
