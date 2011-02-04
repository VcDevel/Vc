/*
    Copyright (C) 2010 Matthias Kretz <kretz@kde.org>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of
    the License, or (at your option) version 3.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
    02110-1301, USA.

*/

#include "tsc.h"
#include "../examples/mandelbrot/mandel.h"
#include <cstdio>
#include <cstring>

class Output
{
    public:
        Output(int cols) : m_fd(stdout), m_cols(cols), m_col(0) {}
        ~Output();

        void open(const char * = 0);

        Output &operator<<(int);
        Output &operator<<(unsigned long long);
        Output &operator<<(float);
        Output &operator<<(double);
        Output &operator<<(bool);
        Output &operator<<(const char *);

    private:
        Output &foo();

        FILE *m_fd;
        int m_cols;
        int m_col;
};

Output::~Output()
{
    if (m_fd) {
        if (m_fd != stdout) {
            fclose(m_fd);
        }
        m_fd = 0;
    }
}

void Output::open(const char *filename)
{
    if (filename) {
        m_fd = fopen(filename, "w+");
    } else {
        m_fd = stdout;
    }
}

Output &Output::operator<<(int x)
{
    fprintf(m_fd, "%10d", x);
    return foo();
}

Output &Output::operator<<(unsigned long long x)
{
    fprintf(m_fd, "%14lld", x);
    return foo();
}

Output &Output::operator<<(float x)
{
    fprintf(m_fd, "%10f", x);
    return foo();
}

Output &Output::operator<<(double x)
{
    fprintf(m_fd, "%14f", x);
    return foo();
}

Output &Output::operator<<(const char *x)
{
    fprintf(m_fd, "\"%s\"", x);
    return foo();
}

Output &Output::operator<<(bool x)
{
    fprintf(m_fd, "%d", x);
    return foo();
}

Output &Output::foo()
{
    if (++m_col == m_cols) {
        m_col = 0;
        fprintf(m_fd, "\n");
        fflush(m_fd);
    } else {
        fprintf(m_fd, " ");
    }
    return *this;
}

template<MandelImpl Impl>
class MandelHacked : public Mandel<Impl>
{
    public:
        using Mandel<Impl>::mandelMe;
};

void usage(char **argv)
{
    printf("Usage: %s [<options>]\n\n", argv[0]);
    printf("Options:\n");
    printf("  -h|--help           print this message\n");
    printf("  -o|--output <file>  output measurements to file\n\n");
}

int main(int argc, char **argv)
{
#ifdef SCHED_FIFO_BENCHMARKS
    if (SCHED_FIFO != sched_getscheduler(0)) {
        // not realtime priority, check whether the benchmark executable exists
        execv("./benchmark", argv);
        // if the execv call works, great. If it doesn't we just continue, but
        // without realtime prio
    }
#endif

    MandelHacked<VcImpl> mandelVc;
    MandelHacked<ScalarImpl> mandelScalar;
    TimeStampCounter tsc;

    Output out(4);
    for (int i = 1; i < argc; ++i) {
        switch (argv[i][0]) {
        case '-':
            switch (argv[i][1]) {
            case 'o':
                if (argv[i][2] == '\0') {
                    if (++i < argc) {
                        out.open(argv[i]);
                    } else {
                        usage(argv);
                        return 1;
                    }
                } else {
                    out.open(&argv[i][2]);
                }
                break;
            case 'h':
                usage(argv);
                return 0;
            case '-': // long options
                switch (argv[i][2]) {
                case 'h':
                    if (std::strcmp(argv[i], "--help") == 0) {
                        usage(argv);
                        return 0;
                    }
                    usage(argv);
                    return 1;
                case 'o':
                    if (std::strcmp(argv[i], "--output") == 0 && ++i < argc) {
                        out.open(argv[i]);
                        break;
                    }
                    usage(argv);
                    return 1;
                default:
                    usage(argv);
                    return 1;
                }
            default:
                usage(argv);
                return 1;
            }
            break;
        default:
            usage(argv);
            return 1;
        }
    }

    out << "size" << "Vc [cycles]" << "Scalar [cycles]" << "equal";

    for (int size = 25; size <= 700; size += 25) {
        out << size;
        const float x = -2.f;
        const float y = -1.f;
        const float scale = 1.f / size;
        const int maxIterations = 255;

        QImage imageVc(3 * size, 2 * size, QImage::Format_RGB32);
        tsc.Start();
        mandelVc.mandelMe(imageVc, x, y, scale, maxIterations);
        tsc.Stop();
        out << tsc.Cycles();

        QImage imageScalar(3 * size, 2 * size, QImage::Format_RGB32);
        tsc.Start();
        mandelScalar.mandelMe(imageScalar, x, y, scale, maxIterations);
        tsc.Stop();
        out << tsc.Cycles();

        out << (imageVc == imageScalar);
    }
}
