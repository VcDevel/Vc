namespace
{
template<typename Cond, typename T, typename Except = void> struct EnableIfUnsignedInteger : public EnableIf<!IsEqualType<Except, Cond>::Value && IsUnsignedInteger<Cond>::Value, T> {};
template<typename Cond, typename T, typename Except = void> struct EnableIfInteger         : public EnableIf<!IsEqualType<Except, Cond>::Value &&!IsReal<Cond>::Value && CanConvertToInt<Cond>::Value, T> {};
template<typename Cond, typename T, typename Except = void> struct EnableIfSignedInteger   : public EnableIf<!IsEqualType<Except, Cond>::Value &&!IsReal<Cond>::Value && CanConvertToInt<Cond>::Value && !IsUnsignedInteger<Cond>::Value, T> {};
template<typename Cond, typename T> struct EnableIfUnsignedInteger<Cond, T, void> : public EnableIf< IsUnsignedInteger<Cond>::Value, T> {};
template<typename Cond, typename T> struct EnableIfInteger        <Cond, T, void> : public EnableIf<!IsReal<Cond>::Value && CanConvertToInt<Cond>::Value, T> {};
template<typename Cond, typename T> struct EnableIfSignedInteger  <Cond, T, void> : public EnableIf<!IsReal<Cond>::Value && CanConvertToInt<Cond>::Value && !IsUnsignedInteger<Cond>::Value, T> {};
template<typename Cond, typename T> struct EnableIf_short_v;
template<               typename T> struct EnableIf_short_v< short_v, T> : public EnableIf<true, T> {};
template<               typename T> struct EnableIf_short_v<ushort_v, T> : public EnableIf<true, T> {};
template<typename Cond, typename T> struct EnableIf_int_v;
template<               typename T> struct EnableIf_int_v  <   int_v, T> : public EnableIf<true, T> {};
template<               typename T> struct EnableIf_int_v  <  uint_v, T> : public EnableIf<true, T> {};
template<typename Cond, typename T> struct EnableIfNeitherIntegerNorVector : public EnableIf<!CanConvertToInt<Cond>::Value, T> {};
template<typename Cond, typename T> struct EnableIfNeitherIntegerNorVector<Vector<Cond>, T>;
}

// float-int arithmetic operators //{{{1
#define VC_OPERATOR_FORWARD_(ret, op) \
template<typename Scalar> static inline VC_EXACT_TYPE(Scalar, float, double_##ret) operator op(Scalar x, double_v::AsArg y) { return double_v(x) op y; } \
template<typename Scalar> static inline VC_EXACT_TYPE(Scalar, float, double_##ret) operator op(double_v::AsArg x, Scalar y) { return x op double_v(y); } \
template<typename Scalar> static inline typename EnableIfInteger<Scalar, double_##ret>::Value operator op(Scalar x, double_v::AsArg y) { return double_v(x) op y; } \
template<typename Scalar> static inline typename EnableIfInteger<Scalar, double_##ret>::Value operator op(double_v::AsArg x, Scalar y) { return x op double_v(y); } \
\
template<typename V> static inline typename EnableIf_int_v<V, float_##ret>::Value operator op(const V &x, float_v::AsArg y) { return float_v(x) op y; } \
template<typename V> static inline typename EnableIf_int_v<V, float_##ret>::Value operator op(float_v::AsArg x, const V &y) { return x op float_v(y); } \
template<typename V, typename Scalar> static inline typename EnableIf<IsEqualType<V,  int_v>::Value && IsEqualType<Scalar, float>::Value, float_##ret>::Value operator op(const V &x, Scalar y) { return float_v(x) op float_v(y); } \
template<typename V, typename Scalar> static inline typename EnableIf<IsEqualType<V, uint_v>::Value && IsEqualType<Scalar, float>::Value, float_##ret>::Value operator op(const V &x, Scalar y) { return float_v(x) op float_v(y); } \
template<typename V, typename Scalar> static inline typename EnableIf<IsEqualType<V,  int_v>::Value && IsEqualType<Scalar, float>::Value, float_##ret>::Value operator op(Scalar x, const V &y) { return float_v(x) op float_v(y); } \
template<typename V, typename Scalar> static inline typename EnableIf<IsEqualType<V, uint_v>::Value && IsEqualType<Scalar, float>::Value, float_##ret>::Value operator op(Scalar x, const V &y) { return float_v(x) op float_v(y); } \
template<typename Scalar> static inline typename EnableIfInteger<Scalar, float_##ret>::Value operator op(Scalar x, float_v::AsArg y) { return float_v(x) op y; } \
template<typename Scalar> static inline typename EnableIfInteger<Scalar, float_##ret>::Value operator op(float_v::AsArg x, Scalar y) { return x op float_v(y); } \
\
template<typename V> static inline typename EnableIf_short_v<V, sfloat_##ret>::Value operator op(const V &x, sfloat_v::AsArg y) { return sfloat_v(x) op y; } \
template<typename V> static inline typename EnableIf_short_v<V, sfloat_##ret>::Value operator op(sfloat_v::AsArg x, const V &y) { return x op sfloat_v(y); } \
template<typename V, typename Scalar> static inline typename EnableIf<IsEqualType<V,  short_v>::Value && IsEqualType<Scalar, float>::Value, sfloat_##ret>::Value operator op(const V &x, Scalar y) { return sfloat_v(x) op sfloat_v(y); } \
template<typename V, typename Scalar> static inline typename EnableIf<IsEqualType<V, ushort_v>::Value && IsEqualType<Scalar, float>::Value, sfloat_##ret>::Value operator op(const V &x, Scalar y) { return sfloat_v(x) op sfloat_v(y); } \
template<typename V, typename Scalar> static inline typename EnableIf<IsEqualType<V,  short_v>::Value && IsEqualType<Scalar, float>::Value, sfloat_##ret>::Value operator op(Scalar x, const V &y) { return sfloat_v(x) op sfloat_v(y); } \
template<typename V, typename Scalar> static inline typename EnableIf<IsEqualType<V, ushort_v>::Value && IsEqualType<Scalar, float>::Value, sfloat_##ret>::Value operator op(Scalar x, const V &y) { return sfloat_v(x) op sfloat_v(y); } \
template<typename Scalar> static inline typename EnableIfInteger<Scalar, sfloat_##ret>::Value operator op(Scalar x, sfloat_v::AsArg y) { return sfloat_v(x) op y; } \
template<typename Scalar> static inline typename EnableIfInteger<Scalar, sfloat_##ret>::Value operator op(sfloat_v::AsArg x, Scalar y) { return x op sfloat_v(y); } \
\
static inline   uint_##ret operator op(      int_v::AsArg x,      uint_v::AsArg y) { return uint_v(x) op        y ; } \
static inline   uint_##ret operator op(     uint_v::AsArg x,       int_v::AsArg y) { return        x  op uint_v(y); } \
template<typename Scalar> static inline typename EnableIfUnsignedInteger<Scalar,   uint_##ret              >::Value operator op(Scalar x,    int_v::AsArg y) { return   uint_v(x) op   uint_v(y); } \
template<typename Scalar> static inline typename EnableIfSignedInteger  <Scalar,    int_##ret,          int>::Value operator op(Scalar x,    int_v::AsArg y) { return    int_v(x) op          y ; } \
template<typename Scalar> static inline typename EnableIfInteger        <Scalar,   uint_##ret, unsigned int>::Value operator op(Scalar x,   uint_v::AsArg y) { return   uint_v(x) op          y ; } \
template<typename Scalar> static inline typename EnableIfUnsignedInteger<Scalar,   uint_##ret              >::Value operator op(   int_v::AsArg x, Scalar y) { return   uint_v(x) op   uint_v(y); } \
template<typename Scalar> static inline typename EnableIfSignedInteger  <Scalar,    int_##ret,          int>::Value operator op(   int_v::AsArg x, Scalar y) { return          x  op    int_v(y); } \
template<typename Scalar> static inline typename EnableIfInteger        <Scalar,   uint_##ret, unsigned int>::Value operator op(  uint_v::AsArg x, Scalar y) { return          x  op   uint_v(y); } \
\
static inline ushort_##ret operator op(    short_v::AsArg x,    ushort_v::AsArg y) { return ushort_v(x) op          y ; } \
static inline ushort_##ret operator op(   ushort_v::AsArg x,     short_v::AsArg y) { return          x  op ushort_v(y); } \
template<typename Scalar> static inline typename EnableIfUnsignedInteger<Scalar, ushort_##ret                >::Value operator op(Scalar x,  short_v::AsArg y) { return ushort_v(x) op ushort_v(y); } \
template<typename Scalar> static inline typename EnableIfSignedInteger  <Scalar,  short_##ret,          short>::Value operator op(Scalar x,  short_v::AsArg y) { return  short_v(x) op          y ; } \
template<typename Scalar> static inline typename EnableIfInteger        <Scalar, ushort_##ret, unsigned short>::Value operator op(Scalar x, ushort_v::AsArg y) { return ushort_v(x) op          y ; } \
template<typename Scalar> static inline typename EnableIfUnsignedInteger<Scalar, ushort_##ret                >::Value operator op( short_v::AsArg x, Scalar y) { return ushort_v(x) op ushort_v(y); } \
template<typename Scalar> static inline typename EnableIfSignedInteger  <Scalar,  short_##ret,          short>::Value operator op( short_v::AsArg x, Scalar y) { return          x  op  short_v(y); } \
template<typename Scalar> static inline typename EnableIfInteger        <Scalar, ushort_##ret, unsigned short>::Value operator op(ushort_v::AsArg x, Scalar y) { return          x  op ushort_v(y); }

// break incorrect combinations
#define VC_OPERATOR_INTENTIONAL_ERROR_1(V, op) \
template<typename T> static inline typename EnableIfNeitherIntegerNorVector<T, Vc::Error::invalid_operands_of_types<V, T> >::Value operator op(const V &, const T &) { return Vc::Error::invalid_operands_of_types<V, T>(); } \
template<typename T> static inline typename EnableIfNeitherIntegerNorVector<T, Vc::Error::invalid_operands_of_types<T, V> >::Value operator op(const T &, const V &) { return Vc::Error::invalid_operands_of_types<T, V>(); }

#define VC_OPERATOR_INTENTIONAL_ERROR_2(V1, V2, op) \
static inline Vc::Error::invalid_operands_of_types<V1, V2> operator op(V1::AsArg, V2::AsArg) { return Vc::Error::invalid_operands_of_types<V1, V2>(); } \
static inline Vc::Error::invalid_operands_of_types<V2, V1> operator op(V2::AsArg, V1::AsArg) { return Vc::Error::invalid_operands_of_types<V2, V1>(); }

#define VC_OPERATOR_INTENTIONAL_ERROR_3(V, _T, op) \
template<typename T> static inline typename EnableIf<IsEqualType<T, _T>::Value, Vc::Error::invalid_operands_of_types<V, T> >::Value operator op(const V &, const T &) { return Vc::Error::invalid_operands_of_types<V, T>(); } \
template<typename T> static inline typename EnableIf<IsEqualType<T, _T>::Value, Vc::Error::invalid_operands_of_types<T, V> >::Value operator op(const T &, const V &) { return Vc::Error::invalid_operands_of_types<T, V>(); }

//#define VC_EXTRA_CHECKING
#ifdef VC_EXTRA_CHECKING
#define VC_OPERATOR_INTENTIONAL_ERROR(op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v, sfloat_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v,  float_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v,    int_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v,   uint_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v,  short_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v, ushort_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(   int_v,  short_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(  uint_v,  short_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(   int_v, ushort_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(  uint_v, ushort_v, op) \
    VC_APPLY_1(VC_LIST_VECTOR_TYPES, VC_OPERATOR_INTENTIONAL_ERROR_1, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2( float_v,  short_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2( float_v, ushort_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(sfloat_v,  float_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(sfloat_v,    int_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(sfloat_v,   uint_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_3( float_v,   double, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_3(sfloat_v,   double, op)
#else
#define VC_OPERATOR_INTENTIONAL_ERROR(op)
#endif

#define VC_OPERATOR_FORWARD_COMMUTATIVE(ret, op, op2) \
template<typename T> static inline VC_EXACT_TYPE(T,         double, double_##ret) operator op(T x, double_v::AsArg y) { return y op2 x; } \
template<typename T> static inline VC_EXACT_TYPE(T,          float, sfloat_##ret) operator op(T x, sfloat_v::AsArg y) { return y op2 x; } \
template<typename T> static inline VC_EXACT_TYPE(T,          float,  float_##ret) operator op(T x,  float_v::AsArg y) { return y op2 x; } \
template<typename T> static inline VC_EXACT_TYPE(T,            int,    int_##ret) operator op(T x,    int_v::AsArg y) { return y op2 x; } \
template<typename T> static inline VC_EXACT_TYPE(T,   unsigned int,   uint_##ret) operator op(T x,   uint_v::AsArg y) { return y op2 x; } \
template<typename T> static inline VC_EXACT_TYPE(T,          short,  short_##ret) operator op(T x,  short_v::AsArg y) { return y op2 x; } \
template<typename T> static inline VC_EXACT_TYPE(T, unsigned short, ushort_##ret) operator op(T x, ushort_v::AsArg y) { return y op2 x; } \
VC_OPERATOR_FORWARD_(ret, op) \
VC_OPERATOR_INTENTIONAL_ERROR(op)

#define VC_OPERATOR_FORWARD(ret, op) \
template<typename T> static inline VC_EXACT_TYPE(T,         double, double_##ret) operator op(T x, double_v::AsArg y) { return double_v(x) op y; } \
template<typename T> static inline VC_EXACT_TYPE(T,          float, sfloat_##ret) operator op(T x, sfloat_v::AsArg y) { return sfloat_v(x) op y; } \
template<typename T> static inline VC_EXACT_TYPE(T,          float,  float_##ret) operator op(T x,  float_v::AsArg y) { return  float_v(x) op y; } \
template<typename T> static inline VC_EXACT_TYPE(T,            int,    int_##ret) operator op(T x,    int_v::AsArg y) { return    int_v(x) op y; } \
template<typename T> static inline VC_EXACT_TYPE(T,   unsigned int,   uint_##ret) operator op(T x,   uint_v::AsArg y) { return   uint_v(x) op y; } \
template<typename T> static inline VC_EXACT_TYPE(T,          short,  short_##ret) operator op(T x,  short_v::AsArg y) { return  short_v(x) op y; } \
template<typename T> static inline VC_EXACT_TYPE(T, unsigned short, ushort_##ret) operator op(T x, ushort_v::AsArg y) { return ushort_v(x) op y; } \
VC_OPERATOR_FORWARD_(ret, op) \
VC_OPERATOR_INTENTIONAL_ERROR(op)

VC_OPERATOR_FORWARD_COMMUTATIVE(v, *, *)
VC_OPERATOR_FORWARD(v, /)
VC_OPERATOR_FORWARD_COMMUTATIVE(v, +, +)
VC_OPERATOR_FORWARD(v, -)
VC_OPERATOR_FORWARD_COMMUTATIVE(v, |, |)
VC_OPERATOR_FORWARD_COMMUTATIVE(v, &, &)
VC_OPERATOR_FORWARD_COMMUTATIVE(v, ^, ^)
VC_OPERATOR_FORWARD_COMMUTATIVE(m, <, >)
VC_OPERATOR_FORWARD_COMMUTATIVE(m, >, <)
VC_OPERATOR_FORWARD_COMMUTATIVE(m, <=, >=)
VC_OPERATOR_FORWARD_COMMUTATIVE(m, >=, <=)
VC_OPERATOR_FORWARD_COMMUTATIVE(m, ==, ==)
VC_OPERATOR_FORWARD_COMMUTATIVE(m, !=, !=)

#undef VC_OPERATOR_FORWARD_
#undef VC_OPERATOR_INTENTIONAL_ERROR_1
#undef VC_OPERATOR_INTENTIONAL_ERROR_2
#undef VC_OPERATOR_INTENTIONAL_ERROR
#undef VC_OPERATOR_FORWARD_COMMUTATIVE
#undef VC_OPERATOR_FORWARD

// }}}1
