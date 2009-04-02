#include "../vector.h"
#include <iostream>

template<typename T>
std::ostream &operator<<(std::ostream &out, const Vc::Vector<T> &v)
{
    out << "[";
    for (int i = 0; i < v.Size; ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << v[i];
    }
    out << "]";
    return out;
}

#ifdef USE_SSE
template<typename T>
std::ostream &operator<<(std::ostream &out, const Vc::Mask &m)
{
    const int *const mm = reinterpret_cast<const int *>(&m);
    out << "m[";
    for (int i = 0; i < m.Size; ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << mm[i];
    }
    out << "]";
    return out;
}
#endif
