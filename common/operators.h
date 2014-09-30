#ifndef VC_ICC
// ICC ICEs if the following type-traits are in the anonymous namespace
namespace
{
#endif

template <bool C, typename T, typename F>
using Conditional = typename std::conditional<C, T, F>::type;

using std::is_convertible;
using std::is_floating_point;
using std::is_integral;
using std::is_same;

template <typename T> constexpr bool isUnsigned()
{
    return !std::is_same<Traits::decay<T>, bool>::value && Traits::is_unsigned<T>::value;
}
template <typename T> constexpr bool isIntegral()
{
    return Traits::is_integral<T>::value;
}
template <typename T> constexpr bool isVector()
{
    return Traits::is_simd_vector_internal<Traits::decay<T>>::value;
}

template <typename T, bool = isIntegral<T>()> struct MakeUnsignedInternal;
template <typename T> struct MakeUnsignedInternal<Vector<T>, true > { using type = Vector<typename std::make_unsigned<T>::type>; };
template <typename T> struct MakeUnsignedInternal<Vector<T>, false> { using type = Vector<T>; };

template <typename Test, typename T>
using CopyUnsigned = typename MakeUnsignedInternal<T, isIntegral<T>() && isUnsigned<Test>()>::type;

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

template <typename T> static constexpr bool convertsToSomeVector()
{
    return is_convertible<T, double_v>::value || is_convertible<T, float_v>::value ||
           is_convertible<T, int_v>::value || is_convertible<T, uint_v>::value ||
           is_convertible<T, short_v>::value || is_convertible<T, ushort_v>::value;
}

static_assert(isNarrowingFloatConversion<double, float>(), "");
static_assert(isNarrowingFloatConversion<long double, float>(), "");
static_assert(isNarrowingFloatConversion<long double, double>(), "");
static_assert(is_convertible<double, float_v>::value, "");
static_assert(false == ((is_convertible<double, float_v>::value ||
                         (isVector<double>() && is_convertible<float_v, double>::value)) &&
                        !isNarrowingFloatConversion<double, float_v::EntryType>()),
              "");

template <typename V, typename W>
using DetermineReturnType =
    Conditional<(is_same<V, int_v>::value || is_same<V, uint_v>::value) &&
                    (is_same<W, float>::value || is_same<W, float_v>::value),
                float_v,
                CopyUnsigned<W, V>>;

template <typename V, typename W> constexpr bool participateInOverloadResolution()
{
    return isVector<V>() &&            // one operand has to be a vector
           !is_same<V, W>::value &&    // if they're the same type it's already
                                       // covered by Vector::operatorX
           convertsToSomeVector<W>();  // if the other operand is not convertible to a SIMD vector
                                       // type at all then don't use our operator in overload
                                       // resolution at all
}

template <typename V, typename W> constexpr enable_if<isVector<V>(), bool> isValidOperandTypes()
{
    // Vc does not allow operands that could possibly have different Vector::Size.
    return isVector<W>()
               ? (is_convertible<V, W>::value || is_convertible<W, V>::value)
               : (is_convertible<W, DetermineReturnType<V, W>>::value &&
                  !isNarrowingFloatConversion<W, typename DetermineReturnType<V, W>::EntryType>());
}

template <
    typename V,
    typename W,
    bool VectorOperation = participateInOverloadResolution<V, W>() && isValidOperandTypes<V, W>()>
struct TypesForOperatorInternal
{
};

template <typename V, typename W> struct TypesForOperatorInternal<V, W, true>
{
    using type = DetermineReturnType<V, W>;
};

template <typename L, typename R>
using TypesForOperator = typename TypesForOperatorInternal<
    Traits::decay<Conditional<isVector<L>(), L, R>>,
    Traits::decay<Conditional<!isVector<L>(), L, R>>>::type;

template <
    typename V,
    typename W,
    bool IsIncorrect = participateInOverloadResolution<V, W>() && !isValidOperandTypes<V, W>()>
struct IsIncorrectVectorOperands
{
};
template <typename V, typename W> struct IsIncorrectVectorOperands<V, W, true>
{
    using type = void;
};

template <typename L, typename R>
using Vc_does_not_allow_operands_to_a_binary_operator_which_can_have_different_SIMD_register_sizes_on_some_targets_and_thus_enforces_portability =
    typename IsIncorrectVectorOperands<
        Traits::decay<Conditional<isVector<L>(), L, R>>,
        Traits::decay<Conditional<!isVector<L>(), L, R>>>::type;

#ifndef VC_ICC
}  // unnamed namespace
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
    Vc_ALWAYS_INLINE typename TypesForOperator<L, R>::Mask operator op(L &&x, R &&y)               \
    {                                                                                              \
        using V = TypesForOperator<L, R>;                                                          \
        return V(std::forward<L>(x)) op V(std::forward<R>(y));                                     \
    }

#define VC_INVALID_OPERATOR(op)                                                                    \
    template <typename L, typename R>                                                              \
    Vc_does_not_allow_operands_to_a_binary_operator_which_can_have_different_SIMD_register_sizes_on_some_targets_and_thus_enforces_portability<L, R> operator op(L &&, R &&) = delete;
// invalid operands to binary expression. Vc does not allow operands that can have a differing size
// on some targets.

VC_ALL_LOGICAL    (VC_GENERIC_OPERATOR)
VC_ALL_BINARY     (VC_GENERIC_OPERATOR)
VC_ALL_ARITHMETICS(VC_GENERIC_OPERATOR)
VC_ALL_COMPARES   (VC_COMPARE_OPERATOR)

VC_ALL_LOGICAL    (VC_INVALID_OPERATOR)
VC_ALL_BINARY     (VC_INVALID_OPERATOR)
VC_ALL_ARITHMETICS(VC_INVALID_OPERATOR)
VC_ALL_COMPARES   (VC_INVALID_OPERATOR)

#undef VC_GENERIC_OPERATOR
#undef VC_COMPARE_OPERATOR
#undef VC_INVALID_OPERATOR

// }}}1
