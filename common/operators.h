#ifndef VC_ICC
// ICC ICEs if the following type-traits are in the anonymous namespace
namespace
{
#endif

using std::conditional;
using std::integral_constant;
using std::is_convertible;
using std::is_floating_point;
using std::is_integral;
using std::is_unsigned;
using std::is_same;
using std::remove_cv;
using std::remove_reference;

template<typename T> struct IsVector             { static constexpr bool value = false; };
template<typename T> struct IsVector<Vector<T> > { static constexpr bool value =  true; };

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
template<typename From, typename To> struct is_narrowing_float_conversion : public integral_constant<bool,
    is_floating_point<From>::value && (is_integral<To>::value || (is_floating_point<To>::value && sizeof(From) > sizeof(To)))>
{};
static_assert(is_narrowing_float_conversion<double, float>::value, "");
static_assert(is_narrowing_float_conversion<long double, float>::value, "");
static_assert(is_narrowing_float_conversion<long double, double>::value, "");
static_assert(is_convertible<double, float_v>::value, "");
static_assert(false ==
           ((is_convertible<double, float_v>::value || (IsVector<double>::value && is_convertible<float_v, double>::value))
        && !is_narrowing_float_conversion<double, float_v::EntryType>::value), "");

template<typename V, typename W> struct DetermineReturnType { typedef V type; };
template<> struct DetermineReturnType<   int_v,    float> { typedef  float_v type; };
template<> struct DetermineReturnType<  uint_v,    float> { typedef  float_v type; };
template<> struct DetermineReturnType<   int_v,  float_v> { typedef  float_v type; };
template<> struct DetermineReturnType<  uint_v,  float_v> { typedef  float_v type; };
template<> struct DetermineReturnType<   int_v,   uint_v> { typedef   uint_v type; };
template<> struct DetermineReturnType< short_v, ushort_v> { typedef ushort_v type; };
template<typename T> struct DetermineReturnType<   int_v, T> { typedef typename conditional<!is_same<bool, T>::value && is_unsigned<T>::value,   uint_v,   int_v>::type type; };
template<typename T> struct DetermineReturnType< short_v, T> { typedef typename conditional<!is_same<bool, T>::value && is_unsigned<T>::value, ushort_v, short_v>::type type; };

template<typename L, typename R,
    typename V = typename remove_cv<typename remove_reference<typename conditional< IsVector<typename remove_cv<typename remove_reference<L>::type>::type>::value, L, R>::type>::type>::type,
    typename W = typename remove_cv<typename remove_reference<typename conditional<!IsVector<typename remove_cv<typename remove_reference<L>::type>::type>::value, L, R>::type>::type>::type,
    bool = IsVector<V>::value // one operand has to be a vector
        && !is_same<V, W>::value // if they're the same type it's already covered by Vector::operatorX
        && (is_convertible<W, V>::value || (IsVector<W>::value && is_convertible<V, W>::value))
        && !is_narrowing_float_conversion<W, typename V::EntryType>::value
        > struct TypesForOperator
{
    // Vector<decltype(V::EntryType() + W::EntryType())> is not what we want since short_v * int should result in short_v not int_v
    typedef typename DetermineReturnType<V, W>::type type;
};

template<typename L, typename R, typename V, typename W> struct TypesForOperator<L, R, V, W, false> {
    static_assert(!(
            IsVector<V>::value // one operand has to be a vector
            && !is_same<V, W>::value // if they're the same type it's allowed
            && (is_narrowing_float_conversion<W, typename V::EntryType>::value
                || (IsVector<W>::value && !is_convertible<V, W>::value && !is_convertible<W, V>::value)
                || (std::is_arithmetic<W>::value && !is_convertible<W, V>::value)
               )),
            "invalid operands to binary expression. Vc does not allow operands that can have a differing size on some targets.");
};

template<typename L, typename R,
    typename V = typename remove_cv<typename remove_reference<typename conditional< IsVector<typename remove_cv<typename remove_reference<L>::type>::type>::value, L, R>::type>::type>::type,
    typename W = typename remove_cv<typename remove_reference<typename conditional<!IsVector<typename remove_cv<typename remove_reference<L>::type>::type>::value, L, R>::type>::type>::type,
    bool IsIncorrect = IsVector<V>::value // one operand has to be a vector
        && !is_same<V, W>::value // if they're the same type it's allowed
        && (is_narrowing_float_conversion<W, typename V::EntryType>::value
            || (!is_convertible<W, V>::value && !(IsVector<W>::value && is_convertible<V, W>::value))
           )
    > struct IsIncorrectVectorOperands
{
    typedef void type;
    static constexpr bool correct = !IsIncorrect;
};
template<typename L, typename R, typename V, typename W> struct IsIncorrectVectorOperands<L, R, V, W, false> {
    static constexpr bool correct = true;
};

template<typename T> struct GetMaskType { typedef typename T::Mask type; };

#ifndef VC_ICC
}
#endif

#define VC_GENERIC_OPERATOR(op) \
template<typename L, typename R> Vc_ALWAYS_INLINE typename TypesForOperator<L, R>::type operator op(L &&x, R &&y) { typedef typename TypesForOperator<L, R>::type V; return V(std::forward<L>(x)) op V(std::forward<R>(y)); }

#define VC_COMPARE_OPERATOR(op) \
template<typename L, typename R> Vc_ALWAYS_INLINE typename GetMaskType<typename TypesForOperator<L, R>::type>::type operator op(L &&x, R &&y) { typedef typename TypesForOperator<L, R>::type V; return V(std::forward<L>(x)) op V(std::forward<R>(y)); }

#define VC_INVALID_OPERATOR(op) \
template<typename L, typename R> typename IsIncorrectVectorOperands<L, R>::type operator op(L &&, R &&) { static_assert(IsIncorrectVectorOperands<L, R>::correct, "invalid operands to binary expression operator"#op". Vc does not allow operands that can have a differing size on some targets."); }

VC_ALL_LOGICAL    (VC_GENERIC_OPERATOR)
VC_ALL_BINARY     (VC_GENERIC_OPERATOR)
VC_ALL_ARITHMETICS(VC_GENERIC_OPERATOR)
VC_ALL_COMPARES   (VC_COMPARE_OPERATOR)

//VC_ALL_LOGICAL    (VC_INVALID_OPERATOR)
//VC_ALL_BINARY     (VC_INVALID_OPERATOR)
//VC_ALL_ARITHMETICS(VC_INVALID_OPERATOR)
//VC_ALL_COMPARES   (VC_INVALID_OPERATOR)

#undef VC_GENERIC_OPERATOR
#undef VC_COMPARE_OPERATOR
#undef VC_INVALID_OPERATOR

// }}}1
