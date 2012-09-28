#include <cstdio>

template<typename T> struct SincosReference
{
    const T x, s, c;
};

template<typename T> struct Reference
{
    const T x, ref;
};

template<typename T> struct Data
{
    static const SincosReference<T> sincosReference[];
    static const Reference<T> asinReference[];
    static const Reference<T> acosReference[];
    static const Reference<T> atanReference[];
};

template<> const SincosReference<float> Data<float>::sincosReference[] = {
#include "sincos-reference-single.h"
};
template<> const SincosReference<double> Data<double>::sincosReference[] = {
#include "sincos-reference-double.h"
};
template<> const Reference<float> Data<float>::asinReference[] = {
#include "asin-reference-single.h"
};
template<> const Reference<double> Data<double>::asinReference[] = {
#include "asin-reference-double.h"
};
template<> const Reference<float> Data<float>::acosReference[] = {
#include "acos-reference-single.h"
};
template<> const Reference<double> Data<double>::acosReference[] = {
#include "acos-reference-double.h"
};
template<> const Reference<float> Data<float>::atanReference[] = {
#include "atan-reference-single.h"
};
template<> const Reference<double> Data<double>::atanReference[] = {
#include "atan-reference-double.h"
};

enum Function {
    Sincos, Atan, Asin, Acos
};
template<typename T, Function F> static inline const char *filenameOut();
template<> inline const char *filenameOut<float , Sincos>() { return "sincos-reference-single.dat"; }
template<> inline const char *filenameOut<double, Sincos>() { return "sincos-reference-double.dat"; }
template<> inline const char *filenameOut<float , Atan  >() { return "atan-reference-single.dat"; }
template<> inline const char *filenameOut<double, Atan  >() { return "atan-reference-double.dat"; }
template<> inline const char *filenameOut<float , Asin  >() { return "asin-reference-single.dat"; }
template<> inline const char *filenameOut<double, Asin  >() { return "asin-reference-double.dat"; }
template<> inline const char *filenameOut<float , Acos  >() { return "acos-reference-single.dat"; }
template<> inline const char *filenameOut<double, Acos  >() { return "acos-reference-double.dat"; }

template<typename T>
static void convert()
{
    FILE *file = fopen(filenameOut<T, Sincos>(), "wb");
    fwrite(&Data<T>::sincosReference[0], sizeof(SincosReference<T>), sizeof(Data<T>::sincosReference) / sizeof(SincosReference<T>), file);
    fclose(file);

    file = fopen(filenameOut<T, Atan>(), "wb");
    fwrite(&Data<T>::atanReference[0], sizeof(Reference<T>), sizeof(Data<T>::atanReference) / sizeof(Reference<T>), file);
    fclose(file);

    file = fopen(filenameOut<T, Asin>(), "wb");
    fwrite(&Data<T>::asinReference[0], sizeof(Reference<T>), sizeof(Data<T>::asinReference) / sizeof(Reference<T>), file);
    fclose(file);

    file = fopen(filenameOut<T, Acos>(), "wb");
    fwrite(&Data<T>::acosReference[0], sizeof(Reference<T>), sizeof(Data<T>::acosReference) / sizeof(Reference<T>), file);
    fclose(file);
}

int main()
{
    convert<float>();
    convert<double>();
    return 0;
}
