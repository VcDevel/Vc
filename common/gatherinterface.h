/*  This file is part of the Vc library. {{{
Copyright © 2014-2015 Matthias Kretz <kretz@kde.org>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef Vc_CURRENT_CLASS_NAME
#error "incorrect use of common/gatherinterface.h: Vc_CURRENT_CLASS_NAME must be defined to the current class name for declaring constructors."
#endif

///////////////////////////////////////////////////////////////////////////////////////////
// gathers
// A gather takes the following arguments:
// 1. A const pointer to memory of any type that can convert to EntryType
// 2. An indexes “vector”. The requirement is that the type implements the subscript operator,
//    stores «Size» valid index values, and each offset to the pointer above yields a valid
//    memory location for reading.
// 3. Optionally the third argument may be a mask. The mask disables several memory reads and
//    thus removes the requirements in (2.) for the disabled entries.

private:
    /**\internal
     * This function implements a gather given a pointer to memory \p mem and some
     * container object storing the gather \p indexes.
     *
     * \param mem This pointer must be aligned correctly for the type \p MT. This is the
     * natural behavior of C++, so this is typically the case.
     * \param indexes This object contains at least \VSize{T} indexes that denote the
     * offset in \p mem where the components for the current vector should be copied from.
     * The offset is not in Bytes, but in multiples of `sizeof(MT)`.
     */
    // enable_if<std::can_convert<MT, EntryType>::value &&
    // has_subscript_operator<IT>::value>
    template <typename MT, typename IT>
    inline void gatherImplementation(const MT *mem, IT &&indexes);

    /**\internal
     * This overload of the above function adds a \p mask argument to disable memory
     * accesses at the \p indexes offsets where \p mask is \c false.
     */
    template <typename MT, typename IT>
    inline void gatherImplementation(const MT *mem, IT &&indexes, MaskArgument mask);

    /**\internal
     * Overload for the case of C-arrays or %Vc vector objects.
     *
     * In this case the \p indexes parameter is usable without adjustment.
     *
     * \param indexes An object to be used for gather or scatter.
     * \returns Forwards the \p indexes parameter.
     */
    template <typename IT, typename = enable_if<std::is_pointer<IT>::value ||
                                                Traits::is_simd_vector<IT>::value>>
    static Vc_INTRINSIC IT adjustIndexParameter(IT &&indexes)
    {
        return std::forward<IT>(indexes);
    }

    /**\internal
     * Overload for the case of a container that returns an lvalue reference from its
     * subscript operator.
     *
     * In this case the container is assumed to use contiguous storage and therefore the
     * \p indexes object is converted to a C-array interface.
     *
     * \param indexes An object to be used for gather or scatter.
     * \returns A pointer to the first object in the \p indexes container.
     */
    template <typename IT,
              typename = enable_if<
                  !std::is_pointer<IT>::value && !Traits::is_simd_vector<IT>::value &&
                  std::is_lvalue_reference<decltype(std::declval<IT>()[0])>::value>>
    static Vc_INTRINSIC decltype(std::addressof(std::declval<IT>()[0]))
        adjustIndexParameter(IT &&i)
    {
        return std::addressof(i[0]);
    }

    /**\internal
     * Overload for the case of a container that returns an rvalue from its
     * subscript operator.
     *
     * \param indexes An object to be used for gather or scatter.
     * \returns Forwards the \p indexes parameter.
     */
    template <typename IT>
    static Vc_INTRINSIC
        enable_if<!std::is_pointer<IT>::value && !Traits::is_simd_vector<IT>::value &&
                      !std::is_lvalue_reference<decltype(std::declval<IT>()[0])>::value,
                  IT>
            adjustIndexParameter(IT &&i)
    {
        return std::forward<IT>(i);
    }

public:
#define Vc_ASSERT_GATHER_PARAMETER_TYPES_                                                \
    static_assert(                                                                       \
        std::is_convertible<MT, EntryType>::value,                                       \
        "The memory pointer needs to point to a type that can be converted to the "      \
        "EntryType of this SIMD vector type.");                                          \
    static_assert(                                                                       \
        Vc::Traits::has_subscript_operator<IT>::value,                                   \
        "The indexes argument must be a type that implements the subscript operator.");  \
    static_assert(                                                                       \
        !Traits::is_simd_vector<IT>::value ||                                            \
            Traits::simd_vector_size<IT>::value >= Size,                                 \
        "If you use a SIMD vector for the indexes parameter, the index vector must "     \
        "have at least as many entries as this SIMD vector.");                           \
    static_assert(                                                                       \
        !std::is_array<T>::value ||                                                      \
            (std::rank<T>::value == 1 &&                                                 \
             (std::extent<T>::value == 0 || std::extent<T>::value >= Size)),             \
        "If you use a simple array for the indexes parameter, the array must have "      \
        "at least as many entries as this SIMD vector.")

    /**
     * \name Gather constructors and member functions
     *
     * Constructs or loads a vector from the objects at `mem[indexes[0]]`,
     * `mem[indexes[1]]`, `mem[indexes[2]]`, ...
     *
     * All gather functions optionally take a mask as last argument. In that case only the
     * entries that are selected in the mask are accessed in memory and copied to the
     * vector. This enables invalid indexes in the \p indexes vector if those are masked
     * off in \p mask.
     *
     * Gathers from structured data (AoS: arrays of struct) are possible via a special
     * subscript operator of the container (array). You can use \ref Vc::array and \ref
     * Vc::vector as drop-in replacements for \c std::array and \c std::vector. These
     * container classes contain the necessary subscript operator overload. Example:
     * \code
     * Vc::vector<float> data(100);
     * std::iota(data.begin(), data.end(), 0.f);  // fill with values 0, 1, 2, ...
     * auto indexes = float_v::IndexType::IndexesFromZero();
     * float_v gathered = data[indexes];  // gathered == [0, 1, 2, ...]
     * \endcode
     *
     * Alternatively, you can use Vc::Common::AdaptSubscriptOperator to extend a given
     * container class with the necessary subscript operator. Example:
     * \code
     * template <typename T, typename Allocator = std::allocator<T>>
     * using my_vector = Vc::Common::AdaptSubscriptOperator<std::vector<T, Allocator>>;
     * \endcode
     *
     * \param mem A pointer to memory which contains objects of type \p MT at the offsets
     *            given by \p indexes.
     * \param indexes A container/vector of offsets into \p mem.
     *                The type of \p indexes (\p IT) may either be a pointer to integers
     *                (C-array) or a vector of integers (preferrably IndexType).
     * \param mask If a mask is given, only the active entries will be copied from memory.
     *
     * \note If you use a masked gather constructor the masked-off entries of the vector
     * are zero-initilized.
     */
    ///@{

    /// Gather constructor
    template <typename MT, typename IT,
              typename = enable_if<Traits::has_subscript_operator<IT>::value>>
    Vc_INTRINSIC Vc_CURRENT_CLASS_NAME(const MT *mem, IT &&indexes)
    {
        Vc_ASSERT_GATHER_PARAMETER_TYPES_;
        gatherImplementation(mem, adjustIndexParameter(std::forward<IT>(indexes)));
    }

    /// Masked gather constructor
    template <typename MT, typename IT,
              typename = enable_if<Vc::Traits::has_subscript_operator<IT>::value>>
    Vc_INTRINSIC Vc_CURRENT_CLASS_NAME(const MT *mem, IT &&indexes, MaskArgument mask)
    {
        Vc_ASSERT_GATHER_PARAMETER_TYPES_;
        gatherImplementation(mem, adjustIndexParameter(std::forward<IT>(indexes)), mask);
    }

    /// Gather function
    template <typename MT,
              typename IT,
              typename = enable_if<Vc::Traits::has_subscript_operator<IT>::value>>
    Vc_INTRINSIC void gather(const MT *mem, IT &&indexes)
    {
        Vc_ASSERT_GATHER_PARAMETER_TYPES_;
        gatherImplementation(mem, adjustIndexParameter(std::forward<IT>(indexes)));
    }

    /// Masked gather function
    template <typename MT,
              typename IT,
              typename = enable_if<Vc::Traits::has_subscript_operator<IT>::value>>
    Vc_INTRINSIC void gather(const MT *mem, IT &&indexes, MaskArgument mask)
    {
        Vc_ASSERT_GATHER_PARAMETER_TYPES_;
        gatherImplementation(mem, adjustIndexParameter(std::forward<IT>(indexes)), mask);
    }
    ///@}

    /// \name Deprecated Members
    ///@{

    /**
     * \deprecated Use Vc::array or Vc::vector subscripting instead.
     *
     * \param array   A pointer into memory (without alignment restrictions).
     * \param member1 If \p array points to a struct, \p member1 determines the member in the struct to
     *                be read. Thus the offsets in \p indexes are relative to the \p array and not to
     *                the size of the gathered type (i.e. array[i].*member1 is accessed instead of
     *                (&(array->*member1))[i])
     * \param indexes Determines the offsets into \p array where the values are gathered from/scattered
     *                to. The type of indexes can either be an integer vector or a type that supports
     *                operator[] access.
     */
    template <typename S1, typename IT>
    Vc_DEPRECATED("use the subscript operator to Vc::array or Vc::vector "
                  "instead.") inline Vc_CURRENT_CLASS_NAME(const S1 *array,
                                                           const EntryType S1::*member1,
                                                           IT indexes)
    {
        gather(Common::SubscriptOperation<S1, IT, std::ratio<1, 1>, true>(
                   array, indexes)[member1]
                   .gatherArguments());
    }

    /**
     * \deprecated Use Vc::array or Vc::vector subscripting instead.
     *
     * \param array   A pointer into memory (without alignment restrictions).
     * \param member1 If \p array points to a struct, \p member1 determines the member in the struct to
     *                be read. Thus the offsets in \p indexes are relative to the \p array and not to
     *                the size of the gathered type (i.e. array[i].*member1 is accessed instead of
     *                (&(array->*member1))[i])
     * \param indexes Determines the offsets into \p array where the values are gathered from/scattered
     *                to. The type of indexes can either be an integer vector or a type that supports
     *                operator[] access.
     * \param mask    If a mask is given only the active entries will be gathered/scattered.
     */
    template <typename S1, typename IT>
    Vc_DEPRECATED("use the subscript operator to Vc::array or Vc::vector "
                  "instead.") inline Vc_CURRENT_CLASS_NAME(const S1 *array,
                                                           const EntryType S1::*member1,
                                                           IT indexes, MaskArgument mask)
    {
        gather(Common::SubscriptOperation<S1, IT, std::ratio<1, 1>, true>(
                   array, indexes)[member1]
                   .gatherArguments(),
               mask);
    }

    /**
     * \deprecated Use Vc::array or Vc::vector subscripting instead.
     *
     * \param array   A pointer into memory (without alignment restrictions).
     * \param member1 If \p array points to a struct, \p member1 determines the member in the struct to
     *                be read. Thus the offsets in \p indexes are relative to the \p array and not to
     *                the size of the gathered type (i.e. array[i].*member1 is accessed instead of
     *                (&(array->*member1))[i])
     * \param member2 If \p member1 is a struct then \p member2 selects the member to be read from that
     *                struct (i.e. array[i].*member1.*member2 is read).
     * \param indexes Determines the offsets into \p array where the values are gathered from/scattered
     *                to. The type of indexes can either be an integer vector or a type that supports
     *                operator[] access.
     */
    template <typename S1, typename S2, typename IT>
    Vc_DEPRECATED("use the subscript operator to Vc::array or Vc::vector "
                  "instead.") inline Vc_CURRENT_CLASS_NAME(const S1 *array,
                                                           const S2 S1::*member1,
                                                           const EntryType S2::*member2,
                                                           IT indexes)
    {
        gather(Common::SubscriptOperation<S1, IT, std::ratio<1, 1>, true>(
                   array, indexes)[member1][member2]
                   .gatherArguments());
    }

    /**
     * \deprecated Use Vc::array or Vc::vector subscripting instead.
     *
     * \param array   A pointer into memory (without alignment restrictions).
     * \param member1 If \p array points to a struct, \p member1 determines the member in the struct to
     *                be read. Thus the offsets in \p indexes are relative to the \p array and not to
     *                the size of the gathered type (i.e. array[i].*member1 is accessed instead of
     *                (&(array->*member1))[i])
     * \param member2 If \p member1 is a struct then \p member2 selects the member to be read from that
     *                struct (i.e. array[i].*member1.*member2 is read).
     * \param indexes Determines the offsets into \p array where the values are gathered from/scattered
     *                to. The type of indexes can either be an integer vector or a type that supports
     *                operator[] access.
     * \param mask    If a mask is given only the active entries will be gathered/scattered.
     */
    template <typename S1, typename S2, typename IT>
    Vc_DEPRECATED("use the subscript operator to Vc::array or Vc::vector "
                  "instead.") inline Vc_CURRENT_CLASS_NAME(const S1 *array,
                                                           const S2 S1::*member1,
                                                           const EntryType S2::*member2,
                                                           IT indexes, MaskArgument mask)
    {
        gather(Common::SubscriptOperation<S1, IT, std::ratio<1, 1>, true>(
                   array, indexes)[member1][member2]
                   .gatherArguments(),
               mask);
    }

    /**
     * \deprecated Use Vc::array or Vc::vector subscripting instead.
     *
     * \param array   A pointer into memory (without alignment restrictions).
     * \param ptrMember1 If \p array points to a struct, \p member1 determines the member in the struct to
     *                be read. Thus the offsets in \p indexes are relative to the \p array and not to
     *                the size of the gathered type (i.e. array[i].*member1 is accessed instead of
     *                (&(array->*member1))[i])
     * \param outerIndexes
     * \param innerIndexes
     */
    template <typename S1, typename IT1, typename IT2>
    Vc_DEPRECATED(
        "use the subscript operator to Vc::array or Vc::vector "
        "instead.") inline Vc_CURRENT_CLASS_NAME(const S1 *array,
                                                 const EntryType *const S1::*ptrMember1,
                                                 IT1 outerIndexes, IT2 innerIndexes)
    {
        gather(Common::SubscriptOperation<S1, IT1, std::ratio<1, 1>, true>(
                   array, outerIndexes)[ptrMember1][innerIndexes]
                   .gatherArguments());
    }

    /**
     * \deprecated Use Vc::array or Vc::vector subscripting instead.
     *
     * \param array   A pointer into memory (without alignment restrictions).
     * \param ptrMember1 If \p array points to a struct, \p member1 determines the member in the struct to
     *                be read. Thus the offsets in \p indexes are relative to the \p array and not to
     *                the size of the gathered type (i.e. array[i].*member1 is accessed instead of
     *                (&(array->*member1))[i])
     * \param outerIndexes
     * \param innerIndexes
     * \param mask    If a mask is given only the active entries will be gathered/scattered.
     */
    template <typename S1, typename IT1, typename IT2>
    Vc_DEPRECATED(
        "use the subscript operator to Vc::array or Vc::vector "
        "instead.") inline Vc_CURRENT_CLASS_NAME(const S1 *array,
                                                 const EntryType *const S1::*ptrMember1,
                                                 IT1 outerIndexes, IT2 innerIndexes,
                                                 MaskArgument mask)
    {
        gather(Common::SubscriptOperation<S1, IT1, std::ratio<1, 1>, true>(
                   array, outerIndexes)[ptrMember1][innerIndexes]
                   .gatherArguments(),
               mask);
    }

    /**
     * \deprecated Use Vc::array or Vc::vector subscripting instead.
     *
     * \param array   A pointer into memory (without alignment restrictions).
     * \param member1 If \p array points to a struct, \p member1 determines the member in the struct to
     *                be read. Thus the offsets in \p indexes are relative to the \p array and not to
     *                the size of the gathered type (i.e. array[i].*member1 is accessed instead of
     *                (&(array->*member1))[i])
     * \param indexes Determines the offsets into \p array where the values are gathered from/scattered
     *                to. The type of indexes can either be an integer vector or a type that supports
     *                operator[] access.
     */
    template <typename S1, typename IT>
    Vc_DEPRECATED("use the subscript operator to Vc::array or Vc::vector "
                  "instead.") inline void gather(const S1 *array,
                                                 const EntryType S1::*member1, IT indexes)
    {
        gather(Common::SubscriptOperation<S1, IT, std::ratio<1, 1>, true>(
                   array, indexes)[member1]
                   .gatherArguments());
    }

    /**
     * \deprecated Use Vc::array or Vc::vector subscripting instead.
     *
     * \param array   A pointer into memory (without alignment restrictions).
     * \param member1 If \p array points to a struct, \p member1 determines the member in the struct to
     *                be read. Thus the offsets in \p indexes are relative to the \p array and not to
     *                the size of the gathered type (i.e. array[i].*member1 is accessed instead of
     *                (&(array->*member1))[i])
     * \param indexes Determines the offsets into \p array where the values are gathered from/scattered
     *                to. The type of indexes can either be an integer vector or a type that supports
     *                operator[] access.
     * \param mask    If a mask is given only the active entries will be gathered/scattered.
     */
    template <typename S1, typename IT>
    Vc_DEPRECATED("use the subscript operator to Vc::array or Vc::vector "
                  "instead.") inline void gather(const S1 *array,
                                                 const EntryType S1::*member1,
                                                 IT indexes,
                                                 MaskArgument mask)
    {
        gather(Common::SubscriptOperation<S1, IT, std::ratio<1, 1>, true>(
                   array, indexes)[member1]
                   .gatherArguments(),
               mask);
    }

    /**
     * \deprecated Use Vc::array or Vc::vector subscripting instead.
     *
     * \param array   A pointer into memory (without alignment restrictions).
     * \param member1 If \p array points to a struct, \p member1 determines the member in the struct to
     *                be read. Thus the offsets in \p indexes are relative to the \p array and not to
     *                the size of the gathered type (i.e. array[i].*member1 is accessed instead of
     *                (&(array->*member1))[i])
     * \param member2 If \p member1 is a struct then \p member2 selects the member to be read from that
     *                struct (i.e. array[i].*member1.*member2 is read).
     * \param indexes Determines the offsets into \p array where the values are gathered from/scattered
     *                to. The type of indexes can either be an integer vector or a type that supports
     *                operator[] access.
     */
    template <typename S1, typename S2, typename IT>
    Vc_DEPRECATED("use the subscript operator to Vc::array or Vc::vector "
                  "instead.") inline void gather(const S1 *array, const S2 S1::*member1,
                                                 const EntryType S2::*member2, IT indexes)
    {
        gather(Common::SubscriptOperation<S1, IT, std::ratio<1, 1>, true>(
                   array, indexes)[member1][member2]
                   .gatherArguments());
    }

    /**
     * \deprecated Use Vc::array or Vc::vector subscripting instead.
     *
     * \param array   A pointer into memory (without alignment restrictions).
     * \param member1 If \p array points to a struct, \p member1 determines the member in the struct to
     *                be read. Thus the offsets in \p indexes are relative to the \p array and not to
     *                the size of the gathered type (i.e. array[i].*member1 is accessed instead of
     *                (&(array->*member1))[i])
     * \param member2 If \p member1 is a struct then \p member2 selects the member to be read from that
     *                struct (i.e. array[i].*member1.*member2 is read).
     * \param indexes Determines the offsets into \p array where the values are gathered from/scattered
     *                to. The type of indexes can either be an integer vector or a type that supports
     *                operator[] access.
     * \param mask    If a mask is given only the active entries will be gathered/scattered.
     */
    template <typename S1, typename S2, typename IT>
    Vc_DEPRECATED("use the subscript operator to Vc::array or Vc::vector "
                  "instead.") inline void gather(const S1 *array, const S2 S1::*member1,
                                                 const EntryType S2::*member2, IT indexes,
                                                 MaskArgument mask)
    {
        gather(Common::SubscriptOperation<S1, IT, std::ratio<1, 1>, true>(
                   array, indexes)[member1][member2]
                   .gatherArguments(),
               mask);
    }

    /**
     * \deprecated Use Vc::array or Vc::vector subscripting instead.
     *
     * \param array   A pointer into memory (without alignment restrictions).
     * \param ptrMember1 If \p array points to a struct, \p member1 determines the member in the struct to
     *                be read. Thus the offsets in \p indexes are relative to the \p array and not to
     *                the size of the gathered type (i.e. array[i].*member1 is accessed instead of
     *                (&(array->*member1))[i])
     * \param outerIndexes
     * \param innerIndexes
     */
    template <typename S1, typename IT1, typename IT2>
    Vc_DEPRECATED("use the subscript operator to Vc::array or Vc::vector "
                  "instead.") inline void gather(const S1 *array,
                                                 const EntryType *const S1::*ptrMember1,
                                                 IT1 outerIndexes, IT2 innerIndexes)
    {
        gather(Common::SubscriptOperation<S1, IT1, std::ratio<1, 1>, true>(
                   array, outerIndexes)[ptrMember1][innerIndexes]
                   .gatherArguments());
    }

    /**
     * \deprecated Use Vc::array or Vc::vector subscripting instead.
     *
     * \param array   A pointer into memory (without alignment restrictions).
     * \param ptrMember1 If \p array points to a struct, \p member1 determines the member in the struct to
     *                be read. Thus the offsets in \p indexes are relative to the \p array and not to
     *                the size of the gathered type (i.e. array[i].*member1 is accessed instead of
     *                (&(array->*member1))[i])
     * \param outerIndexes
     * \param innerIndexes
     * \param mask    If a mask is given only the active entries will be gathered/scattered.
     */
    template <typename S1, typename IT1, typename IT2>
    Vc_DEPRECATED("use the subscript operator to Vc::array or Vc::vector "
                  "instead.") inline void gather(const S1 *array,
                                                 const EntryType *const S1::*ptrMember1,
                                                 IT1 outerIndexes, IT2 innerIndexes,
                                                 MaskArgument mask)
    {
        gather(Common::SubscriptOperation<S1, IT1, std::ratio<1, 1>, true>(
                   array, outerIndexes)[ptrMember1][innerIndexes]
                   .gatherArguments(),
               mask);
    }
    ///@}

    /**\internal
     * \name Gather function to use from Vc::Common::subscript_operator
     *
     * \param args
     * \param mask
     */
    ///@{
    template <typename MT, typename IT>
    Vc_INTRINSIC void gather(const Common::GatherArguments<MT, IT> &args)
    {
        gather(args.address, adjustIndexParameter(args.indexes));
    }

    template <typename MT, typename IT>
    Vc_INTRINSIC void gather(const Common::GatherArguments<MT, IT> &args, MaskArgument mask)
    {
        gather(args.address, adjustIndexParameter(args.indexes), mask);
    }
    ///@}

#undef Vc_ASSERT_GATHER_PARAMETER_TYPES_
