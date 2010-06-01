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

#include "main.h"
#include "tsc.h"
#include <complex>
#include <cmath>

#include <QApplication>
#include <QTimer>
#include <QtCore/QtDebug>
#include <QPainter>
#include <QProgressBar>

#ifdef Scalar
typedef float float_v;
typedef int int_v;
typedef bool int_m;
#else
#include <Vc/float_v>
#include <Vc/int_v>
#include <Vc/IO>

using Vc::float_v;
using Vc::int_v;
using Vc::int_m;
#endif

MainWindow::MainWindow(QWidget *_parent)
    : QWidget(_parent),
    m_width(width()*0.005f), m_height(height()*0.005f),
    m_dirty(true)
{
    m_x = m_width * -0.667f;
    m_y = m_height * -0.5f;
}

void MainWindow::paintEvent(QPaintEvent *e)
{
    if (m_dirty) {
        QTimer::singleShot(0, this, SLOT(recreateImage()));
    }
    QPainter p(this);
    p.translate(-m_dragDelta);
    p.drawImage(e->rect(), m_image, e->rect());
}

void MainWindow::mousePressEvent(QMouseEvent *e)
{
    m_dragStart = e->pos();
}

void MainWindow::mouseMoveEvent(QMouseEvent *e)
{
    m_dragDelta = m_dragStart - e->pos();
    update();
}

void MainWindow::mouseReleaseEvent(QMouseEvent *e)
{
    m_dragDelta = m_dragStart - e->pos();
    // translate m_x, m_y accordingly and recreate the image
    m_x += m_width * m_dragDelta.x() / m_image.width();
    m_y += m_height * m_dragDelta.y() / m_image.height();
    m_dirty = true;
    m_dragDelta = QPoint();
    update();
}

void MainWindow::wheelEvent(QWheelEvent *e)
{
    if (e->delta() < 0 && m_width > 3.f && m_height > 2.f) {
        return;
    }
    const float constX = m_x + m_width * e->x() / width();
    const float constY = m_y + m_height * e->y() / height();
    if (e->delta() > 0) {
        const float zoom = 1.f / (1.f + e->delta() * 0.001f);
        m_width *= zoom;
        m_height *= zoom;
    } else {
        const float zoom = 1.f - e->delta() * 0.001f;
        m_width *= zoom;
        m_height *= zoom;
    }
    m_x = constX - m_width * e->x() / width();
    m_y = constY - m_height * e->y() / height();
    m_dirty = true;
    update();
}

/*   0                       x
 * 0 +----------------------->
 *   |
 *   |
 *   |
 *   |
 *   |
 *   |
 *   |
 * y V
 */
void MainWindow::resizeEvent(QResizeEvent *e)
{
    m_image = QImage(e->size(), QImage::Format_RGB32);
    const float centerX = m_x + 0.5f * m_width;
    const float centerY = m_y + 0.5f * m_height;
    if (e->oldSize().isValid()) {
        // need to adjust for aspect ratio change
        // width or height may only become smaller
        //
        // m_width/m_height == oldWidth/oldHeight
        // m_width/m_height != newWidth/newHeight

        const float tmp = m_height * e->size().width() / e->size().height();
        if (tmp < m_width) {
            m_width = tmp;
        } else {
            m_height = m_width * e->size().height() / e->size().width();
        }
    } else {
        // initialize to full view
        m_width  = 0.005f * e->size().width();
        m_height = 0.005f * e->size().height();
        const float f = 0.5f * (3.f / m_width + 2.f / m_height);
        m_width *= f;
        m_height *= f;
    }
    m_x = centerX - 0.5f * m_width;
    m_y = centerY - 0.5f * m_height;
    m_dirty = true;
    update();
}

typedef std::complex<float_v> Z;

static inline Z P(Z z, Z c)
{
    return z * z + c;
}

static inline Z::value_type fastNorm(const Z &z)
{
    return z.real() * z.real() + z.imag() * z.imag();
}

static const float reduceOffset = std::sqrt(0.2f);
static const float reduceFactor = 1.f / (std::sqrt(1.2f) - reduceOffset);

static inline float reduce(float x)
{
    return (std::sqrt(x + 0.2f) - reduceOffset) * reduceFactor;
}

template<typename T> static inline T square(T a) { return a * a; }
template<typename T> static inline T minOf(T a, T b) { return a < b ? a : b; }
template<typename T> static inline T maxOf(T a, T b) { return a < b ? b : a; }
template<typename T> static inline T clamp(T min, T value, T max)
{
    if (value > max) {
        return max;
    }
    return value < min ? min : value;
}

struct Pixel
{
    int blue;
    int green;
    int red;
};

void MainWindow::recreateImage()
{
    static bool recursionBarrier = false;
    if (recursionBarrier || !m_dirty) {
        return;
    }
    recursionBarrier = true;
    m_dirty = false;
    QProgressBar progressBar;
    progressBar.setRange(0, 1000);
    progressBar.setValue(0);
    progressBar.show();
    QCoreApplication::processEvents();
    TimeStampCounter timer;
    timer.Start();

    // Parameters Begin
    const float S = 4.f;
    const int upperBound[3] = { 500, 500, 500 };
    const int lowerBound[3] = { 100, 100, 100 };
    const int maxIterations = maxOf(upperBound[0], maxOf(upperBound[1], upperBound[2]));
    const float realMin = -2.102613f;
    const float realMax =  1.200613f;
    const float imagMin = -1.23771f;
    const float imagMax = 1.23971f;
    const float sdFactor[3] = { 4.f, 2.f, 2.f };
    // Parameters End

    // helper constants
    const int iHeight = m_image.height();
    const int iWidth  = m_image.width();
    const float xFact = iWidth / m_width;
    const float yFact = iHeight / m_height;
    const float realStep = (realMax - realMin) / (10 * iWidth);
    const float imagStep = (imagMax - imagMin) / (10 * iHeight);
    const Pixel pixelInit = { 0, 0, 0 };

    std::vector<Pixel> pixels(iHeight * iWidth, pixelInit);
#ifdef Scalar
    for (float real = realMin; real <= realMax; real += realStep) {
        progressBar.setValue(990.f * (real - realMin) / (realMax - realMin));
        QCoreApplication::processEvents();
        for (float imag = imagMin; imag <= imagMax; imag += imagStep) {
            Z c(real, imag);
            Z c2 = Z(1.08f * real + 0.15f, imag);
            if (fastNorm(Z(real + 1.f, imag)) < 0.06f || (std::real(c2) < 0.42f && fastNorm(c2) < 0.417f)) {
                continue;
            }
            Z z = c;
            int n;
            for (n = 0; n < maxIterations && fastNorm(z) < S; ++n) {
                z = P(z, c);
            }
            if (n < maxIterations) {
                // point is outside of the Mandelbrot set
                z = c;
                for (int i = 0; i < maxIterations; ++i) {
                    const int y2 = (std::imag(z) - m_y) * yFact;
                    if (y2 >= 0 && y2 < iHeight) {
                        const int x2 = (std::real(z) - m_x) * xFact;
                        if (x2 >= 0 && x2 < iWidth) {
                            Pixel &p = pixels[x2 + y2 * iWidth];
                            if (i >= lowerBound[2] && i <= upperBound[2]) {
                                p.blue += 1;
                            }
                            if (i >= lowerBound[1] && i <= upperBound[1]) {
                                p.green += 1;
                            }
                            if (i >= lowerBound[0] && i <= upperBound[0]) {
                                p.red += 1;
                            }
                        }
                    }
                    z = P(z, c);
                }
            }
        }
    }
#else
    const float imagStep2 = imagStep * float_v::Size;
    const float_v imagMin2 = imagMin + imagStep * static_cast<float_v>(int_v::IndexesFromZero());
    for (float real = realMin; real <= realMax; real += realStep) {
        progressBar.setValue(990.f * (real - realMin) / (realMax - realMin));
        QCoreApplication::processEvents();
        for (float_v imag = imagMin2; imag <= imagMax; imag += imagStep2) {
            Z c(real, imag);
            Z c2 = Z(1.08f * real + 0.15f, imag);
            if (fastNorm(Z(real + 1.f, imag)) < 0.06f || (std::real(c2) < 0.42f && fastNorm(c2) < 0.417f)) {
                continue;
            }
            Z z = c;
            int_v n(Vc::Zero);
            int_m inside = fastNorm(z) < S;
            while (!(inside && n < maxIterations).isEmpty()) {
                z = P(z, c);
                inside &= fastNorm(z) < S;
                ++n(inside);
            }
            if (inside.isFull()) {
                continue;
            }
            z = c;
            for (int i = 0; i < maxIterations; ++i) {
                const int_v y2 = static_cast<int_v>((std::imag(z) - m_y) * yFact);
                const int_v x2 = static_cast<int_v>((std::real(z) - m_x) * xFact);
                z = P(z, c);
                const int_m drawMask = !inside && y2 >= 0 && x2 >= 0 && y2 < iHeight && x2 < iWidth;

                foreach_bit(int j, drawMask) {
                    Pixel &p = pixels[x2[j] + y2[j] * iWidth];
                    if (i >= lowerBound[2] && i <= upperBound[2]) {
                        p.blue += 1;
                    }
                    if (i >= lowerBound[1] && i <= upperBound[1]) {
                        p.green += 1;
                    }
                    if (i >= lowerBound[0] && i <= upperBound[0]) {
                        p.red += 1;
                    }
                }
            }
        }
    }
#endif
    uchar *line = m_image.scanLine(0);
    int max[3] = { 0, 0, 0 };
    float mean[3] = { 0, 0, 0 };
    for (unsigned int i = 0; i < pixels.size(); ++i) {
        max[0] = maxOf(max[0], pixels[i].red);
        max[1] = maxOf(max[0], pixels[i].green);
        max[2] = maxOf(max[0], pixels[i].blue);
        mean[0] += pixels[i].red;
        mean[1] += pixels[i].green;
        mean[2] += pixels[i].blue;
    }
    mean[0] /= pixels.size();
    mean[1] /= pixels.size();
    mean[2] /= pixels.size();
    float sd[3] = { 0.f, 0.f, 0.f };
    for (unsigned int i = 0; i < pixels.size(); ++i) {
        sd[0] += square(pixels[i].red   - mean[0]);
        sd[1] += square(pixels[i].green - mean[1]);
        sd[2] += square(pixels[i].blue  - mean[2]);
    }
    sd[0] = sqrt(sd[0] / pixels.size());
    sd[1] = sqrt(sd[1] / pixels.size());
    sd[2] = sqrt(sd[2] / pixels.size());
    qDebug() << " max:" << max[0] << max[1] << max[2];
    qDebug() << "mean:" << mean[0] << mean[1] << mean[2];
    qDebug() << "  sd:" << sd[0] << sd[1] << sd[2];

    // colors have the range 0..max at this point
    // they should be transformed such that for the resulting mean and sd:
    //    mean - sd = 0
    //    mean + sd = min(min(2 * mean, max), 255)
    //
    // newColor = (c - mean) * min(min(2 * mean, max), 255) * 0.5 / sd + 127.5

    const float center[3] = { minOf(minOf(2.f * mean[0], static_cast<float>(max[0])), 255.f) * 0.5f,
        minOf(minOf(2.f * mean[1], static_cast<float>(max[1])), 255.f) * 0.5f,
        minOf(minOf(2.f * mean[2], static_cast<float>(max[2])), 255.f) * 0.5f };

    const float redFactor   = center[0] / (sdFactor[0] * sd[0]);
    const float greenFactor = center[1] / (sdFactor[1] * sd[1]);
    const float blueFactor  = center[2] / (sdFactor[2] * sd[2]);
    const Pixel *p = &pixels[0];
    for (int yy = 0; yy < iHeight; ++yy) {
        progressBar.setValue(990.f + 10.f * yy / iHeight);
        QCoreApplication::processEvents();
        for (int xx = 0; xx < iWidth; ++xx) {
            line[0] = clamp(0, static_cast<int>(center[2] + (p->blue  - mean[2]) * blueFactor ), 255);
            line[1] = clamp(0, static_cast<int>(center[1] + (p->green - mean[1]) * greenFactor), 255);
            line[2] = clamp(0, static_cast<int>(center[0] + (p->red   - mean[0]) * redFactor  ), 255);
            line += 4;
            ++p;
        }
    }
    timer.Stop();
    qDebug() << timer.Cycles() << "cycles";
    update();
    recursionBarrier = false;
}

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    MainWindow w;
    w.resize(300, 200);
    w.show();
    return app.exec();
}
