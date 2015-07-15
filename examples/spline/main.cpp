/*{{{
    Copyright © 2015 Matthias Kretz <kretz@kde.org>

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
// includes {{{1
#include <array>
#include <iostream>
#include <iomanip>
#include <random>
#include "../tsc.h"
#include "spline.h"
#include "spline2.h"
#include "spline3.h"

// settings {{{1
constexpr int NumberOfEvaluations = 10000;
constexpr int FirstMapSize = 4;
constexpr int MaxMapSize = 256;
constexpr int Repetitions = 100;
constexpr auto StepMultiplier = 1.25;

enum DisabledTests {
    DisabledTestsBegin = -999999,
    Horizontal3,
    DisabledTestsEnd
};
enum EnabledTests {
    Scalar,
    Alice,
    Float4,
    Float16,
    Float12,
    Float12Interleaved,
    Horizontal1,
    Horizontal2,
    Autovectorized,
    NBenchmarks
};

std::string testName(int i)
{
    switch (i) {
    case Scalar:             return "Scalar";
    case Alice:              return "Alice";
    case Float4:             return "Float4";
    case Float16:            return "Float16";
    case Float12:            return "Float12";
    case Float12Interleaved: return "F12Interl.";
    case Horizontal1:        return "Horiz.1";
    case Horizontal2:        return "Horiz.2";
    case Horizontal3:        return "Horiz.3";
    case Autovectorized:     return "Autovec";
    default:                 return "<unknown>";
    }
}

// EnabledTests::operator++ {{{1
EnabledTests &operator++(EnabledTests &x)
{
    return x = static_cast<EnabledTests>(static_cast<int>(x) + 1);
}

// operator<< overloads {{{1
std::ostream &operator<<(std::ostream &s, const Point2 &xyz)
{
    using std::setw;
    return s << '[' << setw(7) << xyz[0] << ", " << setw(7) << xyz[1] << ']';
}
std::ostream &operator<<(std::ostream &s, const Point2V &xyz)
{
    return s << '[' << xyz[0] << ", " << xyz[1] << ']';
}
std::ostream &operator<<(std::ostream &s, const Point3 &xyz)
{
    using std::setw;
    return s << '[' << setw(7) << xyz[0] << ", " << setw(7) << xyz[1] << ", " << setw(7)
             << xyz[2] << ']';
}
std::ostream &operator<<(std::ostream &s, const Point3V &xyz)
{
    return s << '[' << xyz[0] << ", " << xyz[1] << ", " << xyz[2] << ']';
}

// VectorizeBuffer {{{1
template <typename Input> struct VectorizeBuffer
{
    typedef Vc::simdize<Input> InputV;
    InputV input;
    int entries = 0;
    int operator()(Input x)
    {
        assign(input, entries, x);
        entries = (entries + 1) % InputV::size();
        return entries;
    }
};

// TestInfo {{{1
struct TestInfo
{
    bool enabled = false;
    int id = -1;
    TestInfo(EnabledTests t) : enabled(true), id(t) {}
    TestInfo(EnabledTests t, EnabledTests) : enabled(true), id(t) {}
    TestInfo(EnabledTests t, EnabledTests, EnabledTests) : enabled(true), id(t) {}
    TestInfo(EnabledTests t, EnabledTests, DisabledTests) : enabled(true), id(t) {}
    TestInfo(EnabledTests t, DisabledTests) : enabled(true), id(t) {}
    TestInfo(EnabledTests t, DisabledTests, EnabledTests) : enabled(true), id(t) {}
    TestInfo(EnabledTests t, DisabledTests, DisabledTests) : enabled(true), id(t) {}

    TestInfo(DisabledTests) : enabled(false) {}
    TestInfo(DisabledTests, EnabledTests t) : enabled(true), id(t) {}
    TestInfo(DisabledTests, DisabledTests) : enabled(false) {}
    TestInfo(DisabledTests, EnabledTests t, EnabledTests) : enabled(true), id(t) {}
    TestInfo(DisabledTests, DisabledTests, EnabledTests t) : enabled(true), id(t) {}
    TestInfo(DisabledTests, DisabledTests, DisabledTests) : enabled(false) {}

    TestInfo(DisabledTests, EnabledTests t, DisabledTests) : enabled(true), id(t) {}

    operator bool() const { return enabled; }
    operator int() const { return id; }
    operator long() const { return id; }
};
// Runner Lambda {{{1
struct Runner
{
    // data members{{{2
    TimeStampCounter tsc;
    double mean[NBenchmarks] = {};
    double stddev[NBenchmarks] = {};
    const std::vector<Point2> &searchPoints;

    // Runner::Runner{{{2
    Runner(const std::vector<Point2> &p) : searchPoints(p) {}

    void recordTsc(int Test, double norm)  //{{{2
    {
        const double x = tsc.cycles() / norm;
        mean[Test] += x;
        stddev[Test] += x * x;
    }
    template <typename I, typename J> void printRatio(I i, J j)  //{{{2
    {
        if (TestInfo(i) && TestInfo(j)) {
            const auto ratio = mean[i] / mean[j];
            std::cout << std::setprecision(3) << std::setw(9) << ratio;
            std::cout << std::setprecision(3) << std::setw(9)
                      << ratio * std::sqrt(stddev[i] * stddev[i] / (mean[i] * mean[i]) +
                                           stddev[j] * stddev[j] / (mean[j] * mean[j]));
        }
    }
    // benchmarkSearch{{{2
    template <typename F> void benchmark(const TestInfo Test, F &&fun, double err = 20)
    {
        do {
            mean[Test] = 0;
            stddev[Test] = 0;

            for (const auto &p : searchPoints) {
                fun(p);
            }  // one cache warm-up run to remove one outlier
            for (auto rep = Repetitions; rep; --rep) {
                tsc.start();
                for (const auto &p : searchPoints) {
                    fun(p);
                }
                tsc.stop();
                recordTsc(Test, NumberOfEvaluations);
            }

            mean[Test] /= Repetitions;
            stddev[Test] /= Repetitions;
            stddev[Test] = std::sqrt(stddev[Test] - mean[Test] * mean[Test]);
        } while (stddev[Test] * err > mean[Test]);
        std::cout << std::setw(9) << std::setprecision(3) << mean[Test];
        std::cout << std::setw(9) << std::setprecision(3) << stddev[Test];
        std::cout << std::flush;
    }
    //}}}2
};
int main()  // {{{1
{
    // output header {{{2
    using std::cout;
    using std::setw;
    using std::setprecision;
    cout << "NumberOfEvaluations: " << NumberOfEvaluations << '\n';
    cout << "Repetitions: " << Repetitions << '\n';
    cout << setw(8) << "MapSize";
    for (int i = 0; i < NBenchmarks; ++i) {
        cout << setw(18) << testName(i);
    }
    if (TestInfo(Scalar)) {
        for (int i = 0; i < NBenchmarks; ++i) {
            if (i != Scalar) {
                cout << setw(18) << "Scalar/" + testName(i);
            }
        }
    }
    cout << std::endl;

    // random number generator {{{2
    std::default_random_engine randomEngine(1);
    std::uniform_real_distribution<float> uniform(-1.f, 1.f);

    // random search points {{{2
    std::vector<Point2> searchPoints;
    searchPoints.reserve(NumberOfEvaluations);
    searchPoints.emplace_back(Point2{-1.f, -1.f});
    searchPoints.emplace_back(Point2{+1.f, +1.f});
    for (int i = 2; i < NumberOfEvaluations; ++i) {
        searchPoints.emplace_back(Point2{uniform(randomEngine), uniform(randomEngine)});
    }

    // MapSize loop {{{2
    for (int MapSize = FirstMapSize; MapSize <= MaxMapSize; MapSize *= StepMultiplier) {
        Runner runner(searchPoints);
        cout << setw(8) << MapSize * MapSize << std::flush;

        // initialize map with random values {{{2
        Spline spline(-1.f, 1.f, MapSize, -1.f, 1.f, MapSize);
        Spline2 spline2(-1.f, 1.f, MapSize, -1.f, 1.f, MapSize);
        Spline3 spline3(-1.f, 1.f, MapSize, -1.f, 1.f, MapSize);
        for (int i = 0; i < spline.GetNPoints(); ++i) {
            const float xyz[3] = {uniform(randomEngine), uniform(randomEngine),
                                  uniform(randomEngine)};
            spline.Fill(i, xyz);
            spline2.Fill(i, xyz);
            spline3.Fill(i, xyz);
        }

        // run Benchmarks {{{2
        for (EnabledTests i = EnabledTests(0); i < NBenchmarks; ++i) {
            VectorizeBuffer<Point2> vectorizer;
            switch (int(i)) {
            case Scalar:  // {{{3
                runner.benchmark(i, [&](const Point2 &p) {
                    const auto &p2 = spline.GetValueScalar(p);
                    asm("" ::"m"(p2));
                });
                break;
            case Alice:  // {{{3
                runner.benchmark(i, [&](const Point2 &p) {
                    const auto &p2 = spline.GetValueAlice(p);
                    asm("" ::"m"(p2));
                });
                break;
            case Autovectorized:  // {{{3
                runner.benchmark(i, [&](const Point2 &p) {
                    const auto &p2 = spline.GetValueAutovec(p);
                    asm("" ::"m"(p2));
                });
                break;
            case Float4:  // {{{3
                runner.benchmark(i, [&](const Point2 &p) {
                    const auto &p2 = spline.GetValue(p);
                    asm("" ::"m"(p2));
                });
                break;
            case Float16:  // {{{3
                runner.benchmark(i, [&](const Point2 &p) {
                    const auto &p2 = spline.GetValue16(p);
                    asm("" ::"m"(p2));
                });
                break;
            case Float12:  // {{{3
                runner.benchmark(i, [&](const Point2 &p) {
                    const auto &p2 = spline2.GetValue(p);
                    asm("" ::"m"(p2));
                });
                break;
            case Float12Interleaved:  // {{{3
                runner.benchmark(i, [&](const Point2 &p) {
                    const auto &p2 = spline3.GetValue(p);
                    asm("" ::"m"(p2));
                });
                break;
            case Horizontal1:  // {{{3
                runner.benchmark(i, [&](const Point2 &p) {
                    if (0 == vectorizer(p)) {
                        const auto &p2 = spline.GetValue(vectorizer.input);
                        asm("" ::"m"(p2));
                    }
                });
                break;
            case Horizontal2:  // {{{3
                runner.benchmark(i, [&](const Point2 &p) {
                    if (0 == vectorizer(p)) {
                        const auto &p2 = spline2.GetValue(vectorizer.input);
                        asm("" ::"m"(p2));
                    }
                });
                break;
            case Horizontal3:  // {{{3
                runner.benchmark(i, [&](const Point2 &p) {
                    if (0 == vectorizer(p)) {
                        const auto &p2 = spline3.GetValue(vectorizer.input);
                        asm("" ::"m"(p2));
                    }
                });
                break;
            default:  // {{{3
                break;
            }
        }
        // print search timings {{{2
        if (TestInfo(Scalar)) {
            for (EnabledTests i = EnabledTests(0); i < NBenchmarks; ++i) {
                if (i != Scalar) {
                    runner.printRatio(Scalar, i);
                }
            }
        }
        cout << std::flush;

        // verify equivalence {{{2
        {
            bool failed = false;
            VectorizeBuffer<Point2> vectorizer2;
            VectorizeBuffer<Point3> vectorizer3;
            for (const auto &p : searchPoints) {
                const auto &ps = spline.GetValueScalar(p);
                if (TestInfo(Alice)) {  //{{{3
                    const auto &pv = spline.GetValueAlice(p);
                    for (int i = 0; i < 3; ++i) {
                        if (std::abs(ps[i] - pv[i]) > 0.00001f) {
                            std::cout << "\nAlice not equal at " << p << ": " << ps
                                      << " vs. " << pv;
                            failed = true;
                            break;
                        }
                    }
                }
                if (TestInfo(Autovectorized)) {  //{{{3
                    const auto &pv = spline.GetValueAutovec(p);
                    for (int i = 0; i < 3; ++i) {
                        if (std::abs(ps[i] - pv[i]) > 0.00001f) {
                            std::cout << "\nAutovectorized not equal at " << p << ": " << ps
                                      << " vs. " << pv;
                            failed = true;
                            break;
                        }
                    }
                }
                if (TestInfo(Float4)) {  //{{{3
                    const auto &pv = spline.GetValue(p);
                    for (int i = 0; i < 3; ++i) {
                        if (std::abs(ps[i] - pv[i]) > 0.00001f) {
                            std::cout << "\nFloat4 not equal at " << p << ": " << ps
                                      << " vs. " << pv;
                            failed = true;
                            break;
                        }
                    }
                }
                if (TestInfo(Float16)) {  //{{{3
                    const auto &pv = spline.GetValue16(p);
                    for (int i = 0; i < 3; ++i) {
                        if (std::abs(ps[i] - pv[i]) > 0.00001f) {
                            std::cout << "\nFloat16 not equal at " << p << ": " << ps
                                      << " vs. " << pv;
                            failed = true;
                            break;
                        }
                    }
                }
                if (TestInfo(Float12)) {  //{{{3
                    const auto &pv = spline2.GetValue(p);
                    for (int i = 0; i < 3; ++i) {
                        if (std::abs(ps[i] - pv[i]) > 0.00001f) {
                            std::cout << "\nFloat12 not equal at " << p << ": " << ps
                                      << " vs. " << pv;
                            failed = true;
                            break;
                        }
                    }
                }
                if (TestInfo(Float12Interleaved)) {  //{{{3
                    const auto &pv = spline3.GetValue(p);
                    for (int i = 0; i < 3; ++i) {
                        if (std::abs(ps[i] - pv[i]) > 0.00001f) {
                            std::cout << "\nFloat12Interleaved not equal at " << p << ": " << ps
                                      << " vs. " << pv;
                            failed = true;
                            break;
                        }
                    }
                }
                vectorizer3(ps);
                if (0 == vectorizer2(p)) {
                    if (TestInfo(Horizontal1)) {  //{{{3
                        const auto &pv = spline.GetValue(vectorizer2.input);
                        for (int i = 0; i < 3; ++i) {
                            if (any_of(abs(vectorizer3.input[i] - pv[i]) > 0.00001f)) {
                                cout << "\nHorizontal1 not equal at " << vectorizer2.input
                                     << ": " << vectorizer3.input << " vs. " << pv;
                                failed = true;
                                break;
                            }
                        }
                    }
                    if (TestInfo(Horizontal2)) {  //{{{3
                        const auto &pv = spline2.GetValue(vectorizer2.input);
                        for (int i = 0; i < 3; ++i) {
                            if (any_of(abs(vectorizer3.input[i] - pv[i]) > 0.00001f)) {
                                cout << "\nHorizontal2 not equal at \n" << vectorizer2.input
                                     << ":\n" << vectorizer3.input << " vs.\n" << pv;
                                failed = true;
                                break;
                            }
                        }
                    }
                    if (TestInfo(Horizontal3)) {  //{{{3
                        const auto &pv = spline3.GetValue(vectorizer2.input);
                        for (int i = 0; i < 3; ++i) {
                            if (any_of(abs(vectorizer3.input[i] - pv[i]) > 0.00001f)) {
                                cout << "\nHorizontal3 not equal at \n" << vectorizer2.input
                                     << ":\n" << vectorizer3.input << " vs.\n" << pv;
                                failed = true;
                                break;
                            }
                        }
                    }
                }  //{{{3
            }
            if (failed) {
                //std::cout << '\n' << spline. << '\n';
                return 1;
            } else {
                cout << " ✓";
            }
        }
        cout << std::endl;
    }
    return 0;
}  // }}}1

// vim: foldmethod=marker
