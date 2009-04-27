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

template<unsigned int VectorSize>
std::ostream &operator<<(std::ostream &out, const Vc::Mask<VectorSize> &m)
{
    out << "m[";
    for (unsigned int i = 0; i < VectorSize; ++i) {
        if (i > 0 && (i % 4) == 0) {
            out << " ";
        }
        out << m[i];
    }
    out << "]";
    return out;
}
