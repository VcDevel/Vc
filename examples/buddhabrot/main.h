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

#ifndef MAIN_H
#define MAIN_H

#include <QImage>
#include <QTextStream>
#include <QString>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QResizeEvent>
#include <QWheelEvent>
#include <QObject>

class ProgressWriter
{
    public:
        ProgressWriter();
        void setValue(float v);
        void done();

    private:
        QTextStream m_out;
};

class MainWindow : public QObject
{
    public:
        MainWindow(QObject *parent = 0);
        void setSize(int w, int h);
        void setFilename(const QString &);
        void recreateImage();

    private:
        float m_x; // left
        float m_y; // top
        float m_width;
        float m_height;
        QImage m_image;
        QString m_filename;
        ProgressWriter m_progress;
};
#endif // MAIN_H
