#ifndef VC_ICC
// ICC ICEs if the following type-traits are in the anonymous namespace
namespace
{
#endif

template <bool C, typename T, typename F>
using conditional = typename std::conditional<C, T, F>::type;

using std::is_convertible;
using std::is_floating_point;
using std::is_integral;
using std::is_unsigned;
using std::is_same;

template <typename T> struct IsVectorInternal             { static constexpr bool value = false; };
template <typename T> struct IsVectorInternal<Vector<T> > { static constexpr bool value =  true; };
template <typename T> constexpr bool isVector() { return IsVectorInternal<T>::value; }

/* § 8.5.4 p7:
 * A narrowing conversion is an implicit conversion
 * — from a floating-point type to an integer type, or
 * — from long double to double or float, or from double to float, except where the source is a constant
 *   expression and the actual value after conversion is within the range of values that can be represented
 *   (even if it cannot be represented exactly), or
 * — from an integer type or unscoped enumeration type to a floating-point type, except where the source
 *   is a constant expression and the actual value after conversion will fit into the target type and will
 *   produce the original value when converted back to the original type, or
 * — from an integer type or unscoped enumeration type to an integer type that cannot represent all the
 *   values of the original type, except where the source is a constant expression and the actual value after
 *   conversion will fit into the target type and will produce the original value when converted back to the
 *   original type.
 */
template <typename From, typename To> constexpr bool isNarrowingFloatConversion()
{
    return is_floating_point<From>::value &&
           (is_integral<To>::value || (is_floating_point<To>::value && sizeof(From) > sizeof(To)));
}

static_assert(isNarrowingFloatConversion<double, float>(), "");
static_assert(isNarrowingFloatConversion<long double, float>(), "");
static_assert(isNarrowingFloatConversion<long double, double>(), "");
static_assert(is_convertible<double, float_v>::value, "");
static_assert(false == ((is_convertible<double, float_v>::value ||
                         (isVector<double>() && is_convertible<float_v, double>::value)) &&
                        !isNarrowingFloatConversion<double, float_v::EntryType>()),
              "");

template<typename V, typename W> struct DetermineReturnType  { using type = V; };
template<> struct DetermineReturnType<   int_v,    float>    { using type =  float_v; };
template<> struct DetermineReturnType<  uint_v,    float>    { using type =  float_v; };
template<> struct DetermineReturnType<   int_v,  float_v>    { using type =  float_v; };
template<> struct DetermineReturnType<  uint_v,  float_v>    { using type =  float_v; };
template<> struct DetermineReturnType<   int_v,   uint_v>    { using type =   uint_v; };
template<> struct DetermineReturnType< short_v, ushort_v>    { using type = ushort_v; };
template<typename T> struct DetermineReturnType<   int_v, T> { using type = conditional<!is_same<bool, T>::value && is_unsigned<T>::value,   uint_v,   int_v>; };
template<typename T> struct DetermineReturnType< short_v, T> { using type = conditional<!is_same<bool, T>::value && is_unsigned<T>::value, ushort_v, short_v>; };

template <typename V, bool = isVector<V>()> struct VectorEntryTypeOfInternal
{
    using type = typename V::EntryType;
};
template <typename V> struct VectorEntryTypeOfInternal<V, false>
{
    using type = void*;
};
template <typename V> using VectorEntryTypeOf = typename VectorEntryTypeOfInternal<V>::type;

template <typename L,
          typename R,
          typename V = conditional<isVector<L>(), L, R>,
          typename W = conditional<!isVector<L>(), L, R>,
          bool = isVector<V>() &&          // one operand has to be a vector
                 !is_same<V, W>::value &&  // if they're the same type it's already covered by
                                           // Vector::operatorX
                 (is_convertible<W, V>::value || (isVector<W>() && is_convertible<V, W>::value)) &&
                 !isNarrowingFloatConversion<W, VectorEntryTypeOf<V>>()>
struct TypesForOperatorInternal
{
    // Vector<decltype(V::EntryType() + W::EntryType())> is not what we want since short_v * int should result in short_v not int_v
    using type = typename DetermineReturnType<V, W>::type;
};

template <typename L, typename R, typename V, typename W>
struct TypesForOperatorInternal<L, R, V, W, false>
{
    static_assert(!(isVector<V>() &&          // one operand has to be a vector
                    !is_same<V, W>::value &&  // if they're the same type it's allowed
                    (isNarrowingFloatConversion<W, VectorEntryTypeOf<V>>() ||
                     (isVector<W>() && !is_convertible<V, W>::value &&
                      !is_convertible<W, V>::value) ||
                     (std::is_arithmetic<W>::value && !is_convertible<W, V>::value))),
                  "Invalid operands to binary expression. Vc does not allow operands that could "
                  "possibly have different Vector::Size.");
};

template <typename T> using GetMaskType = typename T::Mask;

template <typename L, typename R>
using TypesForOperator = typename TypesForOperatorInternal<typename std::decay<L>::type,
                                                           typename std::decay<R>::type>::type;

#ifndef VC_ICC
}
#endif

#define VC_GENERIC_OPERATOR(op)                                                                    \
    template <typename L, typename R>                                                              \
    Vc_ALWAYS_INLINE TypesForOperator<L, R> operator op(L &&x, R &&y)                              \
    {                                                                                              \
        using V = TypesForOperator<L, R>;                                                          \
        return V(std::forward<L>(x)) op V(std::forward<R>(y));                                     \
    }

#define VC_COMPARE_OPERATOR(op)                                                                    \
    template <typename L, typename R>                                                              \
    Vc_ALWAYS_INLINE GetMaskType<TypesForOperator<L, R>> operator op(L &&x, R &&y)                 \
    {                                                                                              \
        using V = TypesForOperator<L, R>;                                                          \
        return V(std::forward<L>(x)) op V(std::forward<R>(y));                                     \
    }

VC_ALL_LOGICAL    (VC_GENERIC_OPERATOR)
VC_ALL_BINARY     (VC_GENERIC_OPERATOR)
VC_ALL_ARITHMETICS(VC_GENERIC_OPERATOR)
VC_ALL_COMPARES   (VC_COMPARE_OPERATOR)

#undef VC_GENERIC_OPERATOR
#undef VC_COMPARE_OPERATOR

// }}}1
