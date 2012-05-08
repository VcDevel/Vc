// float-int arithmetic operators //{{{1
#define VC_OPERATOR_FORWARD_(ret, op) \
static inline double_##ret operator op(         long long x,    double_v::AsArg y) { return double_v(x) op          y ; } \
static inline double_##ret operator op(unsigned long long x,    double_v::AsArg y) { return double_v(x) op          y ; } \
static inline double_##ret operator op(              long x,    double_v::AsArg y) { return double_v(x) op          y ; } \
static inline double_##ret operator op(     unsigned long x,    double_v::AsArg y) { return double_v(x) op          y ; } \
static inline double_##ret operator op(               int x,    double_v::AsArg y) { return double_v(x) op          y ; } \
static inline double_##ret operator op(      unsigned int x,    double_v::AsArg y) { return double_v(x) op          y ; } \
static inline double_##ret operator op(             short x,    double_v::AsArg y) { return double_v(x) op          y ; } \
static inline double_##ret operator op(    unsigned short x,    double_v::AsArg y) { return double_v(x) op          y ; } \
static inline double_##ret operator op(             float x,    double_v::AsArg y) { return double_v(x) op          y ; } \
static inline double_##ret operator op(   double_v::AsArg x,          long long y) { return          x  op double_v(y); } \
static inline double_##ret operator op(   double_v::AsArg x, unsigned long long y) { return          x  op double_v(y); } \
static inline double_##ret operator op(   double_v::AsArg x,               long y) { return          x  op double_v(y); } \
static inline double_##ret operator op(   double_v::AsArg x,      unsigned long y) { return          x  op double_v(y); } \
static inline double_##ret operator op(   double_v::AsArg x,                int y) { return          x  op double_v(y); } \
static inline double_##ret operator op(   double_v::AsArg x,       unsigned int y) { return          x  op double_v(y); } \
static inline double_##ret operator op(   double_v::AsArg x,              short y) { return          x  op double_v(y); } \
static inline double_##ret operator op(   double_v::AsArg x,     unsigned short y) { return          x  op double_v(y); } \
static inline double_##ret operator op(   double_v::AsArg x,              float y) { return          x  op double_v(y); } \
\
static inline  float_##ret operator op(    float_v::AsArg x,       int_v::AsArg y) { return          x  op  float_v(y); } \
static inline  float_##ret operator op(    float_v::AsArg x,      uint_v::AsArg y) { return          x  op  float_v(y); } \
static inline  float_##ret operator op(    float_v::AsArg x,          long long y) { return          x  op  float_v(y); } \
static inline  float_##ret operator op(    float_v::AsArg x, unsigned long long y) { return          x  op  float_v(y); } \
static inline  float_##ret operator op(    float_v::AsArg x,               long y) { return          x  op  float_v(y); } \
static inline  float_##ret operator op(    float_v::AsArg x,      unsigned long y) { return          x  op  float_v(y); } \
static inline  float_##ret operator op(    float_v::AsArg x,                int y) { return          x  op  float_v(y); } \
static inline  float_##ret operator op(    float_v::AsArg x,       unsigned int y) { return          x  op  float_v(y); } \
static inline  float_##ret operator op(    float_v::AsArg x,              short y) { return          x  op  float_v(y); } \
static inline  float_##ret operator op(    float_v::AsArg x,     unsigned short y) { return          x  op  float_v(y); } \
static inline  float_##ret operator op(     uint_v::AsArg x,     float_v::AsArg y) { return  float_v(x) op          y ; } \
static inline  float_##ret operator op(     uint_v::AsArg x,              float y) { return  float_v(x) op  float_v(y); } \
static inline  float_##ret operator op(      int_v::AsArg x,     float_v::AsArg y) { return  float_v(x) op          y ; } \
static inline  float_##ret operator op(      int_v::AsArg x,              float y) { return  float_v(x) op  float_v(y); } \
static inline  float_##ret operator op(             float x,       int_v::AsArg y) { return  float_v(x) op  float_v(y); } \
static inline  float_##ret operator op(             float x,      uint_v::AsArg y) { return  float_v(x) op  float_v(y); } \
static inline  float_##ret operator op(         long long x,     float_v::AsArg y) { return  float_v(x) op          y ; } \
static inline  float_##ret operator op(unsigned long long x,     float_v::AsArg y) { return  float_v(x) op          y ; } \
static inline  float_##ret operator op(              long x,     float_v::AsArg y) { return  float_v(x) op          y ; } \
static inline  float_##ret operator op(     unsigned long x,     float_v::AsArg y) { return  float_v(x) op          y ; } \
static inline  float_##ret operator op(               int x,     float_v::AsArg y) { return  float_v(x) op          y ; } \
static inline  float_##ret operator op(      unsigned int x,     float_v::AsArg y) { return  float_v(x) op          y ; } \
static inline  float_##ret operator op(             short x,     float_v::AsArg y) { return  float_v(x) op          y ; } \
static inline  float_##ret operator op(    unsigned short x,     float_v::AsArg y) { return  float_v(x) op          y ; } \
\
static inline sfloat_##ret operator op(   sfloat_v::AsArg x,     short_v::AsArg y) { return          x  op sfloat_v(y); } \
static inline sfloat_##ret operator op(   sfloat_v::AsArg x,    ushort_v::AsArg y) { return          x  op sfloat_v(y); } \
static inline sfloat_##ret operator op(   ushort_v::AsArg x,    sfloat_v::AsArg y) { return sfloat_v(x) op          y ; } \
static inline sfloat_##ret operator op(   ushort_v::AsArg x,              float y) { return sfloat_v(x) op sfloat_v(y); } \
static inline sfloat_##ret operator op(    short_v::AsArg x,    sfloat_v::AsArg y) { return sfloat_v(x) op          y ; } \
static inline sfloat_##ret operator op(    short_v::AsArg x,              float y) { return sfloat_v(x) op sfloat_v(y); } \
static inline sfloat_##ret operator op(             float x,     short_v::AsArg y) { return sfloat_v(x) op sfloat_v(y); } \
static inline sfloat_##ret operator op(             float x,    ushort_v::AsArg y) { return sfloat_v(x) op sfloat_v(y); } \
\
static inline sfloat_##ret operator op(   sfloat_v::AsArg x,          long long y) { return          x  op sfloat_v(y); } \
static inline sfloat_##ret operator op(   sfloat_v::AsArg x, unsigned long long y) { return          x  op sfloat_v(y); } \
static inline sfloat_##ret operator op(   sfloat_v::AsArg x,               long y) { return          x  op sfloat_v(y); } \
static inline sfloat_##ret operator op(   sfloat_v::AsArg x,      unsigned long y) { return          x  op sfloat_v(y); } \
static inline sfloat_##ret operator op(   sfloat_v::AsArg x,                int y) { return          x  op sfloat_v(y); } \
static inline sfloat_##ret operator op(   sfloat_v::AsArg x,       unsigned int y) { return          x  op sfloat_v(y); } \
static inline sfloat_##ret operator op(   sfloat_v::AsArg x,              short y) { return          x  op sfloat_v(y); } \
static inline sfloat_##ret operator op(   sfloat_v::AsArg x,     unsigned short y) { return          x  op sfloat_v(y); } \
static inline sfloat_##ret operator op(         long long x,    sfloat_v::AsArg y) { return sfloat_v(x) op          y ; } \
static inline sfloat_##ret operator op(unsigned long long x,    sfloat_v::AsArg y) { return sfloat_v(x) op          y ; } \
static inline sfloat_##ret operator op(              long x,    sfloat_v::AsArg y) { return sfloat_v(x) op          y ; } \
static inline sfloat_##ret operator op(     unsigned long x,    sfloat_v::AsArg y) { return sfloat_v(x) op          y ; } \
static inline sfloat_##ret operator op(               int x,    sfloat_v::AsArg y) { return sfloat_v(x) op          y ; } \
static inline sfloat_##ret operator op(      unsigned int x,    sfloat_v::AsArg y) { return sfloat_v(x) op          y ; } \
static inline sfloat_##ret operator op(             short x,    sfloat_v::AsArg y) { return sfloat_v(x) op          y ; } \
static inline sfloat_##ret operator op(    unsigned short x,    sfloat_v::AsArg y) { return sfloat_v(x) op          y ; } \
\
static inline   uint_##ret operator op(      int_v::AsArg x,      uint_v::AsArg y) { return uint_v(x) op        y ; } \
static inline   uint_##ret operator op(     uint_v::AsArg x,       int_v::AsArg y) { return        x  op uint_v(y); } \
\
static inline   uint_##ret operator op(    unsigned short x,      uint_v::AsArg y) { return uint_v(x) op        y ; } \
static inline   uint_##ret operator op(             short x,      uint_v::AsArg y) { return uint_v(x) op        y ; } \
static inline   uint_##ret operator op(    unsigned short x,       int_v::AsArg y) { return uint_v(x) op uint_v(y); } \
static inline    int_##ret operator op(             short x,       int_v::AsArg y) { return  int_v(x) op        y ; } \
static inline   uint_##ret operator op(               int x,      uint_v::AsArg y) { return uint_v(x) op        y ; } \
static inline   uint_##ret operator op(      unsigned int x,       int_v::AsArg y) { return uint_v(x) op uint_v(y); } \
static inline   uint_##ret operator op(              long x,      uint_v::AsArg y) { return uint_v(x) op        y ; } \
static inline   uint_##ret operator op(     unsigned long x,      uint_v::AsArg y) { return uint_v(x) op        y ; } \
static inline    int_##ret operator op(              long x,       int_v::AsArg y) { return  int_v(x) op        y ; } \
static inline   uint_##ret operator op(     unsigned long x,       int_v::AsArg y) { return uint_v(x) op uint_v(y); } \
static inline   uint_##ret operator op(         long long x,      uint_v::AsArg y) { return uint_v(x) op        y ; } \
static inline   uint_##ret operator op(unsigned long long x,      uint_v::AsArg y) { return uint_v(x) op        y ; } \
static inline    int_##ret operator op(         long long x,       int_v::AsArg y) { return  int_v(x) op        y ; } \
static inline   uint_##ret operator op(unsigned long long x,       int_v::AsArg y) { return uint_v(x) op uint_v(y); } \
static inline   uint_##ret operator op(     uint_v::AsArg x,     unsigned short y) { return        x  op uint_v(y); } \
static inline   uint_##ret operator op(     uint_v::AsArg x,              short y) { return        x  op uint_v(y); } \
static inline   uint_##ret operator op(      int_v::AsArg x,     unsigned short y) { return uint_v(x) op uint_v(y); } \
static inline    int_##ret operator op(      int_v::AsArg x,              short y) { return        x  op  int_v(y); } \
static inline   uint_##ret operator op(      int_v::AsArg x,       unsigned int y) { return uint_v(x) op uint_v(y); } \
static inline   uint_##ret operator op(     uint_v::AsArg x,                int y) { return        x  op uint_v(y); } \
static inline   uint_##ret operator op(     uint_v::AsArg x,      unsigned long y) { return        x  op uint_v(y); } \
static inline   uint_##ret operator op(     uint_v::AsArg x,               long y) { return        x  op uint_v(y); } \
static inline   uint_##ret operator op(      int_v::AsArg x,      unsigned long y) { return uint_v(x) op uint_v(y); } \
static inline    int_##ret operator op(      int_v::AsArg x,               long y) { return        x  op  int_v(y); } \
static inline   uint_##ret operator op(     uint_v::AsArg x, unsigned long long y) { return        x  op uint_v(y); } \
static inline   uint_##ret operator op(     uint_v::AsArg x,          long long y) { return        x  op uint_v(y); } \
static inline   uint_##ret operator op(      int_v::AsArg x, unsigned long long y) { return uint_v(x) op uint_v(y); } \
static inline    int_##ret operator op(      int_v::AsArg x,          long long y) { return        x  op  int_v(y); } \
\
static inline ushort_##ret operator op(    short_v::AsArg x,    ushort_v::AsArg y) { return ushort_v(x) op          y ; } \
static inline ushort_##ret operator op(   ushort_v::AsArg x,     short_v::AsArg y) { return          x  op ushort_v(y); } \
static inline ushort_##ret operator op(             short x,    ushort_v::AsArg y) { return ushort_v(x) op          y ; } \
static inline ushort_##ret operator op(    unsigned short x,     short_v::AsArg y) { return ushort_v(x) op ushort_v(y); } \
static inline ushort_##ret operator op(               int x,    ushort_v::AsArg y) { return ushort_v(x) op          y ; } \
static inline ushort_##ret operator op(      unsigned int x,    ushort_v::AsArg y) { return ushort_v(x) op          y ; } \
static inline  short_##ret operator op(               int x,     short_v::AsArg y) { return  short_v(x) op          y ; } \
static inline ushort_##ret operator op(      unsigned int x,     short_v::AsArg y) { return ushort_v(x) op ushort_v(y); } \
static inline ushort_##ret operator op(              long x,    ushort_v::AsArg y) { return ushort_v(x) op          y ; } \
static inline ushort_##ret operator op(     unsigned long x,    ushort_v::AsArg y) { return ushort_v(x) op          y ; } \
static inline  short_##ret operator op(              long x,     short_v::AsArg y) { return  short_v(x) op          y ; } \
static inline ushort_##ret operator op(     unsigned long x,     short_v::AsArg y) { return ushort_v(x) op ushort_v(y); } \
static inline ushort_##ret operator op(         long long x,    ushort_v::AsArg y) { return ushort_v(x) op          y ; } \
static inline ushort_##ret operator op(unsigned long long x,    ushort_v::AsArg y) { return ushort_v(x) op          y ; } \
static inline  short_##ret operator op(         long long x,     short_v::AsArg y) { return  short_v(x) op          y ; } \
static inline ushort_##ret operator op(unsigned long long x,     short_v::AsArg y) { return ushort_v(x) op ushort_v(y); } \
static inline ushort_##ret operator op(    short_v::AsArg x,     unsigned short y) { return ushort_v(x) op ushort_v(y); } \
static inline ushort_##ret operator op(   ushort_v::AsArg x,              short y) { return          x  op ushort_v(y); } \
static inline ushort_##ret operator op(   ushort_v::AsArg x,                int y) { return          x  op ushort_v(y); } \
static inline ushort_##ret operator op(   ushort_v::AsArg x,       unsigned int y) { return          x  op ushort_v(y); } \
static inline  short_##ret operator op(    short_v::AsArg x,                int y) { return          x  op  short_v(y); } \
static inline ushort_##ret operator op(    short_v::AsArg x,       unsigned int y) { return ushort_v(x) op ushort_v(y); } \
static inline ushort_##ret operator op(   ushort_v::AsArg x,               long y) { return          x  op ushort_v(y); } \
static inline ushort_##ret operator op(   ushort_v::AsArg x,      unsigned long y) { return          x  op ushort_v(y); } \
static inline  short_##ret operator op(    short_v::AsArg x,               long y) { return          x  op  short_v(y); } \
static inline ushort_##ret operator op(    short_v::AsArg x,      unsigned long y) { return ushort_v(x) op ushort_v(y); } \
static inline ushort_##ret operator op(   ushort_v::AsArg x,          long long y) { return          x  op ushort_v(y); } \
static inline ushort_##ret operator op(   ushort_v::AsArg x, unsigned long long y) { return          x  op ushort_v(y); } \
static inline  short_##ret operator op(    short_v::AsArg x,          long long y) { return          x  op  short_v(y); } \
static inline ushort_##ret operator op(    short_v::AsArg x, unsigned long long y) { return ushort_v(x) op ushort_v(y); }

// break incorrect combinations
#define VC_OPERATOR_INTENTIONAL_ERROR_1(V, op) \
template<typename T> static inline Vc::Error::invalid_operands_of_types<V, T> operator op(const V &, const T &) { return Vc::Error::invalid_operands_of_types<V, T>(); } \
template<typename T> static inline Vc::Error::invalid_operands_of_types<T, V> operator op(const T &, const V &) { return Vc::Error::invalid_operands_of_types<T, V>(); }

#define VC_OPERATOR_INTENTIONAL_ERROR_2(V1, V2, op) \
static inline Vc::Error::invalid_operands_of_types<V1, V2> operator op(V1::AsArg, V2::AsArg) { return Vc::Error::invalid_operands_of_types<V1, V2>(); } \
static inline Vc::Error::invalid_operands_of_types<V2, V1> operator op(V2::AsArg, V1::AsArg) { return Vc::Error::invalid_operands_of_types<V2, V1>(); }

#define VC_OPERATOR_INTENTIONAL_ERROR(op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v,  float_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v,    int_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v,   uint_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v,  short_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v, ushort_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(   int_v,  short_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(  uint_v,  short_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(   int_v, ushort_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(  uint_v, ushort_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_1(double_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_1(sfloat_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_1( float_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_1(   int_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_1(  uint_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_1( short_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_1(ushort_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2( float_v,  short_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2( float_v, ushort_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(sfloat_v, double_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(sfloat_v,  float_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(sfloat_v,    int_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(sfloat_v,   uint_v, op)

#define VC_OPERATOR_FORWARD_COMMUTATIVE(ret, op, op2) \
static inline double_##ret operator op(         double x, double_v::AsArg y) { return y op2 x; } \
static inline sfloat_##ret operator op(          float x, sfloat_v::AsArg y) { return y op2 x; } \
static inline  float_##ret operator op(          float x,  float_v::AsArg y) { return y op2 x; } \
static inline    int_##ret operator op(            int x,    int_v::AsArg y) { return y op2 x; } \
static inline   uint_##ret operator op(   unsigned int x,   uint_v::AsArg y) { return y op2 x; } \
static inline  short_##ret operator op(          short x,  short_v::AsArg y) { return y op2 x; } \
static inline ushort_##ret operator op( unsigned short x, ushort_v::AsArg y) { return y op2 x; } \
VC_OPERATOR_FORWARD_(ret, op) \
VC_OPERATOR_INTENTIONAL_ERROR(op)

#define VC_OPERATOR_FORWARD(ret, op) \
static inline double_##ret operator op(         double x, double_v::AsArg y) { return double_v(x) op y; } \
static inline sfloat_##ret operator op(          float x, sfloat_v::AsArg y) { return sfloat_v(x) op y; } \
static inline  float_##ret operator op(          float x,  float_v::AsArg y) { return  float_v(x) op y; } \
static inline    int_##ret operator op(            int x,    int_v::AsArg y) { return    int_v(x) op y; } \
static inline   uint_##ret operator op(   unsigned int x,   uint_v::AsArg y) { return   uint_v(x) op y; } \
static inline  short_##ret operator op(          short x,  short_v::AsArg y) { return  short_v(x) op y; } \
static inline ushort_##ret operator op( unsigned short x, ushort_v::AsArg y) { return ushort_v(x) op y; } \
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
