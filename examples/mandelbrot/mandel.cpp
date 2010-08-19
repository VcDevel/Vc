/*
    Copyright (C) 2010 Matthias Kretz <kretz@kde.org>

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

*/

#include "mandel.h"
#include <QMutexLocker>
#include <QtCore/QtDebug>
#include "tsc.h"

using Vc::float_v;
using Vc::int_v;
using Vc::int_m;

template<MandelImpl Impl>
Mandel<Impl>::Mandel(QObject *_parent)
    : MandelBase(_parent)
{
}

MandelBase::MandelBase(QObject *_parent)
    : QThread(_parent),
    m_restart(false), m_abort(false)
{
}

MandelBase::~MandelBase()
{
    m_mutex.lock();
    m_abort = true;
    m_wait.wakeOne();
    m_mutex.unlock();

    wait();
}

void MandelBase::brot(const QSize &size, float x, float y, float scale)
{
    QMutexLocker lock(&m_mutex);

    m_size = size;
    m_x = x;
    m_y = y;
    m_scale = scale;

    if (!isRunning()) {
        start(LowPriority);
    } else {
        m_restart = true;
        m_wait.wakeOne();
    }
}

void MandelBase::run()
{
    while (!m_abort) {
        // first we copy the parameters to our local data so that the main main thread can give a
        // new task while we're working
        m_mutex.lock();
        // destination image, RGB is good - no need for alpha
        QImage image(m_size, QImage::Format_RGB32);
        float x = m_x;
        float y = m_y;
        float scale = m_scale;
        m_mutex.unlock();

        // benchmark the number of cycles it takes
        TimeStampCounter timer;
        timer.Start();

        // calculate the mandelbrot set/image
        mandelMe(image, x, y, scale, 255);

        timer.Stop();

        // if no new set was requested in the meantime - return the finished image
        if (!m_restart) {
            emit ready(image, timer.Cycles());
        }

        // wait for more work
        m_mutex.lock();
        if (!m_restart) {
            m_wait.wait(&m_mutex);
        }
        m_restart = false;
        m_mutex.unlock();
    }
}

static const float S = 4.f;

static int min(int a, int b) { return a < b ? a : b; }

/**
 * std::norm is implemented as std::abs(z) * std::abs(z) for float
 *
 * define our own fast norm implementation ignoring the corner cases
 */
template<typename T>
static inline T fastNorm(const std::complex<T> &z)
{
    return z.real() * z.real() + z.imag() * z.imag();
}

template<typename T> inline T P(T z, T c)
{
    return T(
            z.real() * z.real() + c.real() - z.imag() * z.imag(),
            (z.real() + z.real()) * z.imag() + c.imag()
            );
}

template<> void Mandel<VcImpl>::mandelMe(QImage &image, float x0,
        float y0, float scale, int maxIt)
{
    typedef std::complex<float_v> Z;
    const int height = image.height();
    const int width = image.width();
    for (int y = 0; y < height; ++y) {
        uchar *__restrict__ line = image.scanLine(y);
        const float_v c_imag = y0 + y * scale;
        for (int_v x = int_v::IndexesFromZero(); x[0] < width;
                x += float_v::Size) {
            const Z c(x0 + static_cast<float_v>(x) * scale, c_imag);
            Z z = c;
            Z z2(z.real() * z.real(), z.imag() * z.imag());
            int_v n = int_v::Zero();
            int_m inside = z2.real() + z2.imag() < S;
            while (!(inside && n < maxIt).isEmpty()) {
                z = Z(z2.real() + c.real() - z2.imag(), (z.real() + z.real()) * z.imag() + c.imag());
                z2 = Z(z.real() * z.real(), z.imag() * z.imag());
                ++n(inside);
                inside = z2.real() + z2.imag() < S;
            }
            int_v colorValue = (maxIt - n) * 255 / maxIt;
            const int bound = min(int_v::Size, width - x[0]);
            for (int i = 0; i < bound; ++i) {
                line[0] = colorValue[i];
                line[1] = colorValue[i];
                line[2] = colorValue[i];
                line += 4;
            }
        }
        if (restart()) {
            break;
        }
    }
}

template<> void Mandel<ScalarImpl>::mandelMe(QImage &image, float x0,
        float y0, float scale, int maxIt)
{
    typedef std::complex<float> Z;
    const int height = image.height();
    const int width = image.width();
    for (int y = 0; y < height; ++y) {
        uchar *__restrict__ line = image.scanLine(y);
        const float c_imag = y0 + y * scale;
        for (int x = 0; x < width; ++x) {
            const Z c(x0 + x * scale, c_imag);
            Z z = c;
            int n = 0;
            for (; fastNorm(z) < S && n < maxIt; ++n) {
                z = P(z, c);
            }
            const uchar colorValue = (maxIt - n) * 255 / maxIt;
            line[0] = colorValue;
            line[1] = colorValue;
            line[2] = colorValue;
            line += 4;
        }
        if (restart()) {
            break;
        }
    }
}

template class Mandel<VcImpl>;
template class Mandel<ScalarImpl>;

// vim: sw=4 sts=4 et tw=100
