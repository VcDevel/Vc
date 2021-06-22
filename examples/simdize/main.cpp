/*  This file is part of the Vc project
    Copyright (C) 2009-2020 Matthias Kretz <kretz@kde.org>
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

#include <Vc/simdize>
#include <Vc/IO>

template <typename T> struct PointTemplate {
    T x, y, z;

    Vc_SIMDIZE_INTERFACE((x, y, z));
};

using Point = PointTemplate<float>;

template <typename T> class Vectorizer
{
public:
    using value_type = Vc::simdize<T>;

    void append(const T &x) { decorate(data)[count++] = x; }
    bool isFull() const { return count == data.size(); }
    const value_type &contents() const { return data; }

private:
    value_type data;
    std::size_t count = 0;
};

int Vc_CDECL main()
{
    Vectorizer<Point> v;
    float x = 0.f;
    while (!v.isFull()) {
        v.append(Point{x, x + 1, x + 2});
        x += 10;
    }
    const auto &vec = v.contents();
    std::cout << vec.x << '\n' << vec.y << '\n' << vec.z << '\n';
    return 0;
}
