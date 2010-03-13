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

static const float S = 4.f;
static const int maxIterations = 255;

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
    const float oneOverHeight = 1.f / m_image.height();
    const float oneOverWidth  = 1.f / m_image.width();
#ifdef Scalar
    m_image.fill(0);
    for (int yy = 0; yy < m_image.height(); ++yy) {
        uchar *line = m_image.scanLine(yy);
        int xx;
        const float_v c_imag = m_y + yy * m_height * oneOverHeight;
        int offset = 0;
        for (xx = 0; xx < m_image.width(); ++xx) {
            Z c(m_x + xx * m_width * oneOverWidth, c_imag);
            Z c2 = Z(1.08f * std::real(c) + 0.15f, std::imag(c));
            if (std::norm(c - Z(-1.f, 0.f)) < 0.06f || (std::real(c2) < 0.42f && std::norm(c2) < 0.417f)) {
                offset += 4;
                continue;
            }
            Z z = c;
            int i;
            for (i = 0; i < maxIterations && std::norm(z) < S; ++i) {
                z = P(z, c);
            }
            const uchar value = 255 - i;
            line[offset + 0] = value;
            line[offset + 1] = value;
            line[offset + 2] = value;
            offset += 4;
        }
    }
#else
    const int maxX = m_image.width() / float_v::Size * float_v::Size;
    for (int yy = 0; yy < m_image.height(); ++yy) {
        uchar *line = m_image.scanLine(yy);
        int xx;
        const float_v c_imag = m_y + yy * m_height * oneOverHeight;
        int offset = 0;
        for (xx = 0; xx < maxX; xx += float_v::Size) {
            Z c(m_x + static_cast<float_v>(xx + int_v::IndexesFromZero()) * m_width * oneOverWidth, c_imag);
            Z z = c;
            int_v i(Vc::Zero);
            int_m inside = std::norm(z) < S;
            while (!inside.isEmpty()) {
                z = P(z, c);
                inside &= std::norm(z) < S;
                ++i(inside);
                inside &= i < maxIterations;
            }
            int_v colorValue = 255 - i;
            for (int j = 0; j < int_v::Size; ++j) {
                const uchar value = colorValue[j];
                line[offset + 0] = value;
                line[offset + 1] = value;
                line[offset + 2] = value;
                offset += 4;
            }
        }
        if (maxX != m_image.width()) {
            Z c(m_x + static_cast<float_v>(xx + int_v::IndexesFromZero()) * m_width * oneOverWidth, c_imag);
            Z z = c;
            int_v i(Vc::Zero);
            int_m inside = std::norm(z) < S;
            while (!inside.isEmpty()) {
                z = P(z, c);
                inside &= std::norm(z) < S;
                ++i(inside);
                inside &= i < maxIterations;
            }
            int_v colorValue = 255 - i;
            colorValue(inside) = 0;
            for (int j = 0; j < m_image.width() - maxX; ++j) {
                const uchar value = colorValue[j];
                line[offset + 0] = value;
                line[offset + 1] = value;
                line[offset + 2] = value;
                offset += 4;
            }
        }
    }
#endif
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
