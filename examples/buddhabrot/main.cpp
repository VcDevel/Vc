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
#include <QTextStream>
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
using Vc::float_m;
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
    float blue;
    float green;
    float red;
};

static const Pixel NULL_PIXEL = { 0, 0, 0 };

class Canvas
{
    public:
        Canvas(int h, int w);
        void addDot(float x, float y, int red, int green, int blue);
        void toQImage(QImage *);

    private:
        void addDot(int x, int y, float red, float green, float blue) {
            Pixel &p = m_pixels[x + y * m_width];
            p.blue  += blue;
            p.green += green;
            p.red   += red;
        }
        const int m_width;
        std::vector<Pixel> m_pixels;
};

Canvas::Canvas(int h, int w)
    : m_width(w), m_pixels(h * w, NULL_PIXEL)
{
}

void Canvas::addDot(float x, float y, int red, int green, int blue)
{
    const int x1 = static_cast<int>(std::floor(x));
    const int x2 = static_cast<int>(std::ceil (x));
    const int y1 = static_cast<int>(std::floor(y));
    const int y2 = static_cast<int>(std::ceil (y));
    const float xfrac = x - std::floor(x);
    const float yfrac = y - std::floor(y);
    const float r = red;
    const float g = green;
    const float b = blue;
    const float frac11 = (1.f - xfrac) * (1.f - yfrac);
    const float frac12 = (1.f - xfrac) * yfrac;
    const float frac21 = xfrac * (1.f - yfrac);
    const float frac22 = xfrac * yfrac;
    addDot(x1, y1, r * frac11, g * frac11, b * frac11);
    addDot(x2, y1, r * frac21, g * frac21, b * frac21);
    addDot(x1, y2, r * frac12, g * frac12, b * frac12);
    addDot(x2, y2, r * frac22, g * frac22, b * frac22);
}

#define BUDDHABROT_USE_FUNCTION1

#ifdef BUDDHABROT_USE_FUNCTION2
static inline uchar reduceRange(float x, float m, float h)
{
    /* m: max, h: median
     * +-                              -+
     * |        3        3        2     |
     * |   510 h  + 127 m  - 765 h  m   |
     * |   --------------------------   |
     * |         3    3        2  2     |
     * |      h m  + h  m - 2 h  m      |
     * |                                |
     * |         3        3          2  |
     * |  - 255 h  - 254 m  + 765 h m   |
     * |  ----------------------------  |
     * |        4      2  3    3  2     |
     * |     h m  - 2 h  m  + h  m      |
     * |                                |
     * |                    2        2  |
     * |   - 510 h m + 255 h  + 127 m   |
     * |   ---------------------------  |
     * |         4      2  3    3  2    |
     * |      h m  - 2 h  m  + h  m     |
     * +-                              -+
     */
    const float h2 = h * h;
    const float h3 = h2 * h;
    const float m2 = m * m;
    const float m3 = m2 * m;
    const float denom = h * m * square(m - h);
    return minOf(255.f, 0.5f //rounding
            + x / denom * (
                510.f * h3 + 127.f * m3 - 765.f * h2 * m
                + x / m * (
                    765.f * h * m2 - 255.f * h3 - 254.f * m3
                    + x * (
                        255.f * h2 + 127.f * m2 - 510.f * h * m)
                    )));
}
#elif defined(BUDDHABROT_USE_FUNCTION1)
static inline unsigned int reduceRange(float x, float m, float h)
{
    if (x <= m) {
        return 0.5f // rounding
            + 4.f / 255.f * h * h / m * x
            + square(x) * (h / square(m)) * (4.f - 8.f / 255.f * h);
    } else {
        return 0.5f // rounding
            + 255.f - 4.f * h + 4.f / 255.f * square(h)
            + x / m * (16.f * h - 1020.f - 12.f / 255.f * square(h))
            + square(x / m) * (1020.f - 12.f * h + 8.f / 255.f * square(h));
    }
}
#endif

void Canvas::toQImage(QImage *img)
{
    uchar *line = img->scanLine(0);
    const Pixel *p = &m_pixels[0];
#ifdef BUDDHABROT_USE_FUNCTION2
    float max   [3] = { 0.f, 0.f, 0.f };
    std::vector<float> sorted[3];
    for (int i = 0; i < 3; ++i) {
        sorted[i].reserve(m_pixels.size());
    }
    for (unsigned int i = 0; i < m_pixels.size(); ++i) {
        max[0] = maxOf(max[0], m_pixels[i].red);
        max[1] = maxOf(max[1], m_pixels[i].green);
        max[2] = maxOf(max[2], m_pixels[i].blue);
        if (m_pixels[i].red > 1.f) {
            sorted[0].push_back(m_pixels[i].red);
        }
        if (m_pixels[i].green > 1.f) {
            sorted[1].push_back(m_pixels[i].green);
        }
        if (m_pixels[i].blue > 1.f) {
            sorted[2].push_back(m_pixels[i].blue);
        }
    }
    for (int i = 0; i < 3; ++i) {
        std::sort(sorted[i].begin(), sorted[i].end());
    }
    const float median[3] = {
        sorted[0][sorted[0].size() / 2],
        sorted[1][sorted[1].size() / 2],
        sorted[2][sorted[2].size() / 2]
    };

    /*
    int hist[3][2];
    for (int i = 0; i < 3; ++i) {
        hist[i][0] = hist[i][1] = 0;
    }
    for (unsigned int i = 0; i < m_pixels.size(); ++i) {
        ++hist[0][reduceRange(m_pixels[i].red  , max[0], median[0]) / 128];
        ++hist[1][reduceRange(m_pixels[i].green, max[1], median[1]) / 128];
        ++hist[2][reduceRange(m_pixels[i].blue , max[2], median[2]) / 128];
    }
    qDebug() << "Histogram:\n  red:"
        << median[0] << hist[0][0] << hist[0][1] << "\ngreen:"
        << median[1] << hist[1][0] << hist[1][1] << "\n blue:"
        << median[2] << hist[2][0] << hist[2][1];
    */

    for (int yy = 0; yy < img->height(); ++yy) {
        for (int xx = 0; xx < img->width(); ++xx) {
            line[0] = reduceRange(p->blue , max[2], median[2]);
            line[1] = reduceRange(p->green, max[1], median[1]);
            line[2] = reduceRange(p->red  , max[0], median[0]);
            line += 4;
            ++p;
        }
    }
#elif defined(BUDDHABROT_USE_FUNCTION1)
    float max[3] = { 0.f, 0.f, 0.f };
    for (unsigned int i = 0; i < m_pixels.size(); ++i) {
        max[0] = maxOf(max[0], m_pixels[i].red);
        max[1] = maxOf(max[1], m_pixels[i].green);
        max[2] = maxOf(max[2], m_pixels[i].blue);
    }
    float h[3] = { 220.f, 220.f, 220.f };

    /*
    int hist[3][2];
    for (int i = 0; i < 3; ++i) {
        hist[i][0] = hist[i][1] = 0;
    }
    for (unsigned int i = 0; i < m_pixels.size(); ++i) {
        ++hist[0][reduceRange(m_pixels[i].red  , max[0], h[0]) / 128];
        ++hist[1][reduceRange(m_pixels[i].green, max[1], h[1]) / 128];
        ++hist[2][reduceRange(m_pixels[i].blue , max[2], h[2]) / 128];
    }
    qDebug() << "Histogram:\n  red:"
        << hist[0][0] << hist[0][1] << "\ngreen:"
        << hist[1][0] << hist[1][1] << "\n blue:"
        << hist[2][0] << hist[2][1];
    */

    for (int yy = 0; yy < img->height(); ++yy) {
        for (int xx = 0; xx < img->width(); ++xx) {
            line[0] = reduceRange(p->blue , max[2], h[2]);
            line[1] = reduceRange(p->green, max[1], h[1]);
            line[2] = reduceRange(p->red  , max[0], h[0]);
            line += 4;
            ++p;
        }
    }
#else
    float max   [3] = { 0.f, 0.f, 0.f };
    float mean  [3] = { 0.f, 0.f, 0.f };
    float stddev[3] = { 0.f, 0.f, 0.f };
    for (unsigned int i = 0; i < m_pixels.size(); ++i) {
        max[0] = maxOf(max[0], m_pixels[i].red);
        max[1] = maxOf(max[1], m_pixels[i].green);
        max[2] = maxOf(max[2], m_pixels[i].blue);
        mean[0] += m_pixels[i].red;
        mean[1] += m_pixels[i].green;
        mean[2] += m_pixels[i].blue;
        stddev[0] += square(m_pixels[i].red);
        stddev[1] += square(m_pixels[i].green);
        stddev[2] += square(m_pixels[i].blue);
    }
    const float normalization = 1.f / m_pixels.size();
    mean[0] *= normalization;
    mean[1] *= normalization;
    mean[2] *= normalization;
    stddev[0] = std::sqrt(stddev[0] * normalization - square(mean[0]));
    stddev[1] = std::sqrt(stddev[1] * normalization - square(mean[1]));
    stddev[2] = std::sqrt(stddev[2] * normalization - square(mean[2]));
    qDebug() << "   max:" << max[0] << max[1] << max[2];
    qDebug() << "  mean:" << mean[0] << mean[1] << mean[2];
    qDebug() << "stddev:" << stddev[0] << stddev[1] << stddev[2];

    // colors have the range 0..max at this point
    // they should be transformed such that for the resulting mean and stddev:
    //    mean - stddev = 0
    //    mean + stddev = min(min(2 * mean, max), 255)
    //
    // newColor = (c - mean) * min(min(2 * mean, max), 255) * 0.5 / stddev + 127.5

    const float center[3] = {
        minOf(minOf(2.f * mean[0], max[0]), 255.f) * 0.5f,
        minOf(minOf(2.f * mean[1], max[1]), 255.f) * 0.5f,
        minOf(minOf(2.f * mean[2], max[2]), 255.f) * 0.5f
    };

    const float sdFactor[3] = { 2.f, 2.f, 2.f };
    const float redFactor   = center[0] / (sdFactor[0] * stddev[0]);
    const float greenFactor = center[1] / (sdFactor[1] * stddev[1]);
    const float blueFactor  = center[2] / (sdFactor[2] * stddev[2]);

    for (int yy = 0; yy < img->height(); ++yy) {
        //progressBar.setValue(990.f + 10.f * yy / img->height());
        //QCoreApplication::processEvents();
        for (int xx = 0; xx < img->width(); ++xx) {
            line[0] = clamp(0, static_cast<int>(center[2] + (p->blue  - mean[2]) * blueFactor ), 255);
            line[1] = clamp(0, static_cast<int>(center[1] + (p->green - mean[1]) * greenFactor), 255);
            line[2] = clamp(0, static_cast<int>(center[0] + (p->red   - mean[0]) * redFactor  ), 255);
            line += 4;
            ++p;
        }
    }
#endif
}

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

    const int iHeight = m_image.height();
    const int iWidth  = m_image.width();

    // Parameters Begin
    const float S = 4.f;
    float nSteps[2]   = { 10 * iWidth, 10 * iHeight };
    int upperBound[3] = {    10,     1,    20 };
    int lowerBound[3] = {     2,     0,    11 };
    int overallLowerBound = 10000;
    int maxIterations = 50000;// maxOf(maxOf(overallLowerBound, upperBound[0]), maxOf(upperBound[1], upperBound[2]));
    float realMin = -2.102613f;
    float realMax =  1.200613f;
    float imagMin = 0.f;
    float imagMax = 1.23971f;
    // Parameters End

    const QStringList &args = QCoreApplication::arguments();
    for (int i = 0; i < args.size(); ++i) {
        const QString &arg = args[i];
        bool ok = true;
        if (arg == QLatin1String("--red")) {
            lowerBound[0] = args[++i].toInt(&ok);
            if (ok) {
                upperBound[0] = args[++i].toInt(&ok);
            }
        } else if (arg == QLatin1String("--green")) {
            lowerBound[1] = args[++i].toInt(&ok);
            if (ok) {
                upperBound[1] = args[++i].toInt(&ok);
            }
        } else if (arg == QLatin1String("--blue")) {
            lowerBound[2] = args[++i].toInt(&ok);
            if (ok) {
                upperBound[2] = args[++i].toInt(&ok);
            }
        } else if (arg == QLatin1String("--steps")) {
            nSteps[0] = args[++i].toFloat(&ok);
            if (ok) {
                nSteps[1] = args[++i].toFloat(&ok);
            }
        } else if (arg == QLatin1String("--minIt")) {
            overallLowerBound = args[++i].toInt(&ok);
        } else if (arg == QLatin1String("--maxIt")) {
            maxIterations = args[++i].toInt(&ok);
        }
        if (!ok) {
            QTextStream cout(stdout);
            cout << "incorrect commandline parameters\n";
            cout.flush();
            exit(1);
        }
    }

    TimeStampCounter timer;
    timer.Start();

    // helper constants
    const int overallUpperBound = maxOf(upperBound[0], maxOf(upperBound[1], upperBound[2]));
    const float maxX = static_cast<float>(iWidth ) - 1.f;
    const float maxY = static_cast<float>(iHeight) - 1.f;
    const float xFact = iWidth / m_width;
    const float yFact = iHeight / m_height;
    const float realStep = (realMax - realMin) / nSteps[0];
    const float imagStep = (imagMax - imagMin) / nSteps[1];

    Canvas canvas(iHeight, iWidth);
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
            for (n = 0; n <= maxIterations && fastNorm(z) < S; ++n) {
                z = P(z, c);
            }
            if (n <= maxIterations && n >= overallLowerBound) {
                // point is outside of the Mandelbrot set and required enough (overallLowerBound)
                // iterations to reach the cut-off value S
                Z cn(real, -imag);
                Z zn = cn;
                z = c;
                for (int i = 0; i <= overallUpperBound; ++i) {
                    const float y2 = (std::imag(z) - m_y) * yFact;
                    const float yn2 = (std::imag(zn) - m_y) * yFact;
                    if (y2 >= 0.f && y2 < maxY && yn2 >= 0.f && yn2 < maxY) {
                        const float x2 = (std::real(z) - m_x) * xFact;
                        if (x2 >= 0.f && x2 < maxX) {
                            const int red   = (i >= lowerBound[0] && i <= upperBound[0]) ? 1 : 0;
                            const int green = (i >= lowerBound[1] && i <= upperBound[1]) ? 1 : 0;
                            const int blue  = (i >= lowerBound[2] && i <= upperBound[2]) ? 1 : 0;
                            canvas.addDot(x2, y2 , red, green, blue);
                            canvas.addDot(x2, yn2, red, green, blue);
                        }
                    }
                    z = P(z, c);
                    zn = P(zn, cn);
                    if (fastNorm(z) >= S) { // optimization: skip some useless looping
                        break;
                    }
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
            // FIXME: extra "tracks" if nSteps[1] is not a multiple of float_v::Size
            Z c(real, imag);
            Z c2 = Z(1.08f * real + 0.15f, imag);
            if (fastNorm(Z(real + 1.f, imag)) < 0.06f || (std::real(c2) < 0.42f && fastNorm(c2) < 0.417f)) {
                continue;
            }
            Z z = c;
            int_v n(Vc::Zero);
            int_m inside = fastNorm(z) < S;
            while (!(inside && n <= maxIterations).isEmpty()) {
                z = P(z, c);
                ++n(inside);
                inside &= fastNorm(z) < S;
            }
            inside |= n < overallLowerBound;
            if (inside.isFull()) {
                continue;
            }
            Z cn(real, -imag);
            Z zn = cn;
            z = c;
            for (int i = 0; i <= overallUpperBound; ++i) {
                const float_v y2 = (std::imag(z) - m_y) * yFact;
                const float_v yn2 = (std::imag(zn) - m_y) * yFact;
                const float_v x2 = (std::real(z) - m_x) * xFact;
                z = P(z, c);
                zn = P(zn, cn);
                const float_m drawMask = !inside && y2 >= 0.f && x2 >= 0.f && y2 < maxY && x2 < maxX && yn2 >= 0.f && yn2 < maxY;

                const int red   = (i >= lowerBound[0] && i <= upperBound[0]) ? 1 : 0;
                const int green = (i >= lowerBound[1] && i <= upperBound[1]) ? 1 : 0;
                const int blue  = (i >= lowerBound[2] && i <= upperBound[2]) ? 1 : 0;

                foreach_bit(int j, drawMask) {
                    canvas.addDot(x2[j], y2 [j], red, green, blue);
                    canvas.addDot(x2[j], yn2[j], red, green, blue);
                }
                if (fastNorm(z) >= S) { // optimization: skip some useless looping
                    break;
                }
            }
        }
    }
#endif
    canvas.toQImage(&m_image);

    timer.Stop();
    qDebug() << timer.Cycles() << "cycles";
    update();
    recursionBarrier = false;
}

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    const QStringList &args = QCoreApplication::arguments();
    if (args.contains("--help") || args.contains("-h")) {
        QTextStream out(stdout);
        out << "Usage: " << argv[0] << " [options]\n\n"
            << "Options:\n"
            << "  -h|--help               This message.\n"
            << "  --red   <int> <int>     Specify lower and upper iteration bounds for a red trace.\n"
            << "  --green <int> <int>     Specify lower and upper iteration bounds for a green trace.\n"
            << "  --blue  <int> <int>     Specify lower and upper iteration bounds for a blue trace.\n"
            << "  --steps <float> <float> Specify the steps in real and imaginary direction.\n"
            << "  --minIt <int>           Overall lower iteration bound.\n"
            << "  --maxIt <int>           Overall upper iteration bound.\n"
            ;
        return 0;
    }
    MainWindow w;
    w.resize(300, 200);
    w.show();
    return app.exec();
}
