#include <cstdio>

template<typename T> struct SincosReference
{
    const T x, s, c;
};

template<typename T> struct Data
{
    static const SincosReference<T> sincosReference[];
};

template<> const SincosReference<float> Data<float>::sincosReference[] = {
#include "sincos-reference-single.h"
};
template<> const SincosReference<double> Data<double>::sincosReference[] = {
#include "sincos-reference-double.h"
};

template<typename T> static inline const char *filenameOut();
template<> inline const char *filenameOut<float >() { return "sincos-reference-single.dat"; }
template<> inline const char *filenameOut<double>() { return "sincos-reference-double.dat"; }

template<typename T>
static void convert()
{
    FILE *file = fopen(filenameOut<T>(), "wb");
    fwrite(&Data<T>::sincosReference[0], sizeof(SincosReference<T>), sizeof(Data<T>::sincosReference) / sizeof(SincosReference<T>), file);
    fclose(file);
}

int main()
{
    convert<float>();
    convert<double>();
    return 0;
}
