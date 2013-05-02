/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

/**
 * The type of the vector used for indexes in gather and scatter operations.
 */
typedef INDEX_TYPE IndexType;

/**
 * The type of the entries in the vector.
 */
typedef ENTRY_TYPE EntryType;

/**
 * The type of the mask used for masked operations and returned from comparisons.
 */
typedef MASK_TYPE Mask;

enum {
    /**
     * The size of the vector. I.e. the number of scalar entries in the vector. Do not make any
     * assumptions about the size of vectors. If you need a vector of float vs. integer of the same
     * size make use of IndexType instead. Note that this still does not guarantee the same size
     * (e.g. double_v on SSE has two entries but there exists no 64 bit integer vector type in Vc -
     * which would have two entries; thus double_v::IndexType is uint_v).
     *
     * Also you can easily use if clauses that compare sizes. The
     * compiler can statically evaluate and fully optimize dead code away (very much like \#ifdef,
     * but with syntax checking).
     */
    Size
};

/**
 * Construct an uninitialized vector.
 */
VECTOR_TYPE();

/**
 * Construct a vector with the entries initialized to zero.
 *
 * \see Vc::Zero, Zero()
 */
VECTOR_TYPE(Vc::Zero);

/**
 * Returns a vector with the entries initialized to zero.
 */
static VECTOR_TYPE Zero();

/**
 * Construct a vector with the entries initialized to one.
 *
 * \see Vc::One
 */
VECTOR_TYPE(Vc::One);

/**
 * Returns a vector with the entries initialized to one.
 */
static VECTOR_TYPE One();

#ifdef INTEGER
/**
 * Construct a vector with the entries initialized to 0, 1, 2, 3, 4, 5, ...
 *
 * \see Vc::IndexesFromZero, IndexesFromZero()
 */
VECTOR_TYPE(Vc::IndexesFromZero);

/**
 * Returns a vector with the entries initialized to 0, 1, 2, 3, 4, 5, ...
 */
static VECTOR_TYPE IndexesFromZero();
#endif

/**
 * Returns a vector with pseudo-random entries.
 *
 * Currently the state of the random number generator cannot be modified and starts off with the
 * same state. Thus you will get the same sequence of numbers for the same sequence of calls.
 *
 * \return a new random vector. Floating-point values will be in the 0-1 range. Integers will use
 * the full range the integer representation allows.
 *
 * \note This function may use a very small amount of state and thus will be a weak random number generator.
 */
static VECTOR_TYPE Random();

/**
 * Construct a vector loading its entries from \p alignedMemory.
 *
 * \param alignedMemory A pointer to data. The pointer must be aligned on a
 *                      Vc::VectorAlignment boundary.
 */
VECTOR_TYPE(ENTRY_TYPE *alignedMemory);

/**
 * Convert from another vector type.
 */
template<typename OtherVector> explicit VECTOR_TYPE(const OtherVector &);

/**
 * Broadcast Constructor.
 *
 * Constructs a vector with all entries of the vector filled with the given value.
 *
 * \param x The scalar value to broadcast to all entries of the constructed vector.
 *
 * \note If you want to set it to 0 or 1 use the special initializer constructors above. Calling
 * this constructor with 0 will cause a compilation error because the compiler cannot know which
 * constructor you meant.
 */
VECTOR_TYPE(ENTRY_TYPE x);

/**
 * Construct a vector from an array of vectors with different Size.
 *
 * E.g. convert from two double_v to one float_v.
 *
 * \see expand
 */
//VECTOR_TYPE(const OtherVector *array);

/**
 * Expand the values into an array of vectors that have a different Size.
 *
 * E.g. convert from one float_v to two double_v.
 *
 * This is the reverse of the above constructor.
 */
//void expand(OtherVector *array) const;

/**
 * Load the vector entries from \p memory, overwriting the previous values.
 *
 * \param memory A pointer to data.
 * \param align  Determines whether \p memory is an aligned pointer or not.
 *
 * \see Memory
 */
void load(const ENTRY_TYPE *memory, LoadStoreFlags align = Aligned);

/**
 * Set all entries to zero.
 */
void setZero();

/**
 * Set all entries to zero where the mask is set. I.e. a 4-vector with a mask of 0111 would
 * set the last three entries to 0.
 *
 * \param mask Selects the entries to be set to zero.
 */
void setZero(const MASK_TYPE &mask);

/**
 * Store the vector data to \p memory.
 *
 * \param memory A pointer to memory, where to store.
 * \param align  Determines whether \p memory is an aligned pointer or not.
 *
 * \see Memory
 */
void store(EntryType *memory, LoadStoreFlags align = Aligned) const;

/**
 * This operator can be used to modify scalar entries of the vector.
 *
 * \param index A value between 0 and Size. This value is not checked internally so you must make/be
 *              sure it is in range.
 *
 * \return a reference to the vector entry at the given \p index.
 *
 * \warning This operator is known to miscompile with GCC 4.3.x.
 * \warning The use of this function may result in suboptimal performance. Please check whether you
 * can find a more vector-friendly way to do what you need.
 */
ENTRY_TYPE &operator[](int index);

/**
 * This operator can be used to read scalar entries of the vector.
 *
 * \param index A value between 0 and Size. This value is not checked internally so you must make/be
 *              sure it is in range.
 *
 * \return the vector entry at the given \p index.
 */
ENTRY_TYPE operator[](int index) const;

/**
 * Writemask the vector before an assignment.
 *
 * \param mask The writemask to be used.
 *
 * \return an object that can be used for any kind of masked assignment.
 *
 * The returned object is only to be used for assignments and should not be assigned to a variable.
 *
 * Examples:
 * \code
 * float_v v = float_v::Zero();         // v  = [0, 0, 0, 0]
 * int_v v2 = int_v::IndexesFromZero(); // v2 = [0, 1, 2, 3]
 * v(v2 < 2) = 1.f;                     // v  = [1, 1, 0, 0]
 * v(v2 < 3) += 1.f;                    // v  = [2, 2, 1, 0]
 * ++v2(v < 1.f);                       // v2 = [0, 1, 2, 4]
 * \endcode
 */
MaskedVector operator()(const MASK_TYPE &mask);

/**
 * \name Gather and Scatter Functions
 * The gather and scatter functions allow you to easily use vectors with structured data and random
 * accesses.
 *
 * There are several variants:
 * \li random access in arrays (a[i])
 * \li random access of members of structs in an array (a[i].member)
 * \li random access of members of members of structs in an array (a[i].member1.member2)
 *
 * All gather and scatter functions optionally take a mask as last argument. In that case only the
 * entries that are selected in the mask are read in memory and copied to the vector. This allows
 * you to have invalid indexes in the \p indexes vector if those are masked off in \p mask.
 *
 * \note If you use a constructor for a masked gather then the unmodified entries of the vector are
 * initilized to 0 before the gather. If you really want them uninitialized you can create a
 * uninitialized vector object first and then call the masked gather function on it.
 *
 * The index type (IndexT) can either be a pointer to integers (array) or a vector of integers.
 *
 * Accessing values of a struct works like this:
 * \code
 * struct MyData {
 *   float a;
 *   int b;
 * };
 *
 * void foo(MyData *data, uint_v indexes) {
 *   const float_v v1(data, &MyData::a, indexes);
 *   const int_v   v2(data, &MyData::b, indexes);
 *   v1.scatter(data, &MyData::a, indexes - float_v::Size);
 *   v2.scatter(data, &MyData::b, indexes - 1);
 * }
 * \endcode
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
//@{
/// gather constructor
template<typename IndexT> VECTOR_TYPE(const ENTRY_TYPE *array, const IndexT indexes);
/// masked gather constructor, initialized to zero
template<typename IndexT> VECTOR_TYPE(const ENTRY_TYPE *array, const IndexT indexes, const MASK_TYPE &mask);

/// gather
template<typename IndexT> void gather(const ENTRY_TYPE *array, const IndexT indexes);
/// masked gather
template<typename IndexT> void gather(const ENTRY_TYPE *array, const IndexT indexes, const MASK_TYPE &mask);

/// scatter
template<typename IndexT> void scatter(ENTRY_TYPE *array, const IndexT indexes) const;
/// masked scatter
template<typename IndexT> void scatter(ENTRY_TYPE *array, const IndexT indexes, const MASK_TYPE &mask) const;

/////////////////////////

/// struct member gather constructor
template<typename S1, typename IndexT> VECTOR_TYPE(const S1 *array, const ENTRY_TYPE S1::* member1, const IndexT indexes);
/// masked struct member gather constructor, initialized to zero
template<typename S1, typename IndexT> VECTOR_TYPE(const S1 *array, const ENTRY_TYPE S1::* member1, const IndexT indexes, const MASK_TYPE &mask);

/// struct member gather
template<typename S1, typename IndexT> void gather(const S1 *array, const ENTRY_TYPE S1::* member1, const IndexT indexes);
/// masked struct member gather
template<typename S1, typename IndexT> void gather(const S1 *array, const ENTRY_TYPE S1::* member1, const IndexT indexes, const MASK_TYPE &mask);

/// struct member scatter
template<typename S1, typename IndexT> void scatter(S1 *array, ENTRY_TYPE S1::* member1, const IndexT indexes) const ;
/// masked struct member scatter
template<typename S1, typename IndexT> void scatter(S1 *array, ENTRY_TYPE S1::* member1, const IndexT indexes, const MASK_TYPE &mask) const ;

/////////////////////////

/// struct member of struct member gather constructor
template<typename S1, typename S2, typename IndexT> VECTOR_TYPE(const S1 *array, const S2 S1::* member1, const ENTRY_TYPE S2::* member2, const IndexT indexes);
/// masked struct member of struct member gather constructor, initialized to zero
template<typename S1, typename S2, typename IndexT> VECTOR_TYPE(const S1 *array, const S2 S1::* member1, const ENTRY_TYPE S2::* member2, const IndexT indexes, const MASK_TYPE &mask);

/// struct member of struct member gather
template<typename S1, typename S2, typename IndexT> void gather(const S1 *array, const S2 S1::* member1, const ENTRY_TYPE S2::* member2, const IndexT indexes);
/// masked struct member of struct member gather
template<typename S1, typename S2, typename IndexT> void gather(const S1 *array, const S2 S1::* member1, const ENTRY_TYPE S2::* member2, const IndexT indexes, const MASK_TYPE &mask);

/// struct member of struct member scatter
template<typename S1, typename S2, typename IndexT> void scatter(S1 *array, S2 S1::* member1, ENTRY_TYPE S2::* member2, const IndexT indexes) const ;
/// maksed struct member of struct member scatter
template<typename S1, typename S2, typename IndexT> void scatter(S1 *array, S2 S1::* member1, ENTRY_TYPE S2::* member2, const IndexT indexes, const MASK_TYPE &mask) const ;
//@}

/**
 * \name Comparisons
 *
 * All comparison operators return a mask object.
 *
 * \code
 * void foo(const float_v &a, const float_v &b) {
 *   const float_m mask = a < b;
 *   ...
 * }
 * \endcode
 *
 * \param x The vector to compare with.
 */
//@{
/// Returns mask that is \c true where vector entries are equal and \c false otherwise.
MASK_TYPE operator==(const VECTOR_TYPE &x) const;
/// Returns mask that is \c true where vector entries are not equal and \c false otherwise.
MASK_TYPE operator!=(const VECTOR_TYPE &x) const;
/// Returns mask that is \c true where the left vector entries are greater than on the right and \c false otherwise.
MASK_TYPE operator> (const VECTOR_TYPE &x) const;
/// Returns mask that is \c true where the left vector entries are greater than on the right or equal and \c false otherwise.
MASK_TYPE operator>=(const VECTOR_TYPE &x) const;
/// Returns mask that is \c true where the left vector entries are less than on the right and \c false otherwise.
MASK_TYPE operator< (const VECTOR_TYPE &x) const;
/// Returns mask that is \c true where the left vector entries are less than on the right or equal and \c false otherwise.
MASK_TYPE operator<=(const VECTOR_TYPE &x) const;
//@}

/**
 * \name Arithmetic Operations
 *
 * The vector classes implement all the arithmetic and (bitwise) logical operations as you know from
 * builtin types.
 *
 * \code
 * void foo(const float_v &a, const float_v &b) {
 *   const float_v product    = a * b;
 *   const float_v difference = a - b;
 * }
 * \endcode
 */
//@{
/// Returns a new vector with the sum of the respective entries of the left and right vector.
VECTOR_TYPE operator+(VECTOR_TYPE x) const;
/// Adds the respective entries of \p x to this vector.
VECTOR_TYPE &operator+=(VECTOR_TYPE x);
/// Returns a new vector with the difference of the respective entries of the left and right vector.
VECTOR_TYPE operator-(VECTOR_TYPE x) const;
/// Subtracts the respective entries of \p x from this vector.
VECTOR_TYPE &operator-=(VECTOR_TYPE x);
/// Returns a new vector with the product of the respective entries of the left and right vector.
VECTOR_TYPE operator*(VECTOR_TYPE x) const;
/// Multiplies the respective entries of \p x from to vector.
VECTOR_TYPE &operator*=(VECTOR_TYPE x);
/// Returns a new vector with the quotient of the respective entries of the left and right vector.
VECTOR_TYPE operator/(VECTOR_TYPE x) const;
/// Divides the respective entries of this vector by \p x.
VECTOR_TYPE &operator/=(VECTOR_TYPE x);
/// Returns a new vector with all entries negated.
VECTOR_TYPE operator-() const;
/// Returns a new vector with the binary or of the respective entries of the left and right vector.
VECTOR_TYPE operator|(VECTOR_TYPE x) const;
/// Returns a new vector with the binary and of the respective entries of the left and right vector.
VECTOR_TYPE operator&(VECTOR_TYPE x) const;
/// Returns a new vector with the binary xor of the respective entries of the left and right vector.
VECTOR_TYPE operator^(VECTOR_TYPE x) const;
#ifdef VECTOR_TYPE_HAS_SHIFTS
/// Returns a new vector with each entry bitshifted to the left by \p x bits.
VECTOR_TYPE operator<<(int x) const;
/// Bitshift each entry to the left by \p x bits.
VECTOR_TYPE &operator<<=(int x);
/// Returns a new vector with each entry bitshifted to the right by \p x bits.
VECTOR_TYPE operator>>(int x) const;
/// Bitshift each entry to the right by \p x bits.
VECTOR_TYPE &operator>>=(int x);
/// Returns a new vector with each entry bitshifted to the left by \p x[i] bits.
VECTOR_TYPE operator<<(VECTOR_TYPE x) const;
/// Bitshift each entry to the left by \p x[i] bits.
VECTOR_TYPE &operator<<=(VECTOR_TYPE x);
/// Returns a new vector with each entry bitshifted to the right by \p x[i] bits.
VECTOR_TYPE operator>>(VECTOR_TYPE x) const;
/// Bitshift each entry to the right by \p x[i] bits.
VECTOR_TYPE &operator>>=(VECTOR_TYPE x);
#endif
/**
 * Multiplies this vector with \p factor and then adds \p summand, without rounding between the
 * multiplication and the addition.
 *
 * \param factor The multiplication factor.
 * \param summand The summand that will be added after multiplication.
 *
 * \note This operation may have explicit hardware support, in which case it is normally faster to
 * use the FMA instead of separate multiply and add instructions.
 * \note If the target hardware does not have FMA support this function will be considerably slower
 * than a normal a * b + c. This is due to the increased precision fusedMultiplyAdd provides.
 */
void fusedMultiplyAdd(VECTOR_TYPE factor, VECTOR_TYPE summand);
//@}

/**
 * \name Horizontal Reduction Operations
 *
 * There are four horizontal operations available to reduce the values of a vector to a scalar
 * value.
 *
 * \code
 * void foo(const float_v &v) {
 *   float min = v.min(); // smallest value in v
 *   float sum = v.sum(); // sum of all values in v
 * }
 * \endcode
 */
//@{
/// Returns the smallest entry in the vector.
ENTRY_TYPE min() const;
/// Returns the largest entry in the vector.
ENTRY_TYPE max() const;
/// Returns the product of all entries in the vector.
ENTRY_TYPE product() const;
/// Returns the sum of all entries in the vector.
ENTRY_TYPE sum() const;
//@}

/**
 * \name Apply/Call/Fill Functions
 *
 * There are still many situations where the code needs to switch from SIMD operations to scalar
 * execution. In this case you can, of course rely on operator[]. But there are also a number of
 * functions that can help with common patterns.
 *
 * The apply functions expect a function that returns a scalar value, i.e. a function of the form "T f(T)".
 * The call functions do not return a value and thus the function passed does not need a return
 * value. The fill functions are used to serially set the entries of the vector from the return
 * values of a function.
 *
 * Example:
 * \code
 * void foo(float_v v) {
 *   float_v logarithm = v.apply(std::log);
 *   float_v exponential = v.apply(std::exp);
 * }
 * \endcode
 *
 * Of course, with C++11, you can also use lambdas here:
 * \code
 *   float_v power = v.apply([](float f) { return std::pow(f, 0.6f); })
 * \endcode
 *
 * \param f A functor: this can either be a function or an object that implements operator().
 */
//@{
/// Return a new vector where each entry is the return value of \p f called on the current value.
template<typename Functor> VECTOR_TYPE apply(Functor &f) const;
/// Const overload of the above function.
template<typename Functor> VECTOR_TYPE apply(const Functor &f) const;
/// As above, but skip the entries where \p mask is not set.
template<typename Functor> VECTOR_TYPE apply(Functor &f, MASK_TYPE mask) const;
/// Const overload of the above function.
template<typename Functor> VECTOR_TYPE apply(const Functor &f, MASK_TYPE mask) const;
/// Call \p f with the scalar entries of the vector.
template<typename Functor> void call(Functor &f) const;
/// Const overload of the above function.
template<typename Functor> void call(const Functor &f) const;
/// As above, but skip the entries where \p mask is not set.
template<typename Functor> void call(Functor &f, MASK_TYPE mask) const;
/// Const overload of the above function.
template<typename Functor> void call(const Functor &f, MASK_TYPE mask) const;
/// Fill the vector with the values [f(), f(), f(), ...].
void fill(ENTRY_TYPE (&f)());
/// Fill the vector with the values [f(0), f(1), f(2), ...].
template<typename IndexT> void fill(ENTRY_TYPE (&f)(IndexT));
//@}

/**
 * \name Swizzles
 *
 * Swizzles are a special form of shuffles that, depending on the target hardware and swizzle type,
 * may be used without extra cost. The swizzles act on every successive four entries in the vector.
 * Thus the swizzle \verbatim [0, 1, 2, 3, 4, 5, 6, 7].dcba() \endverbatim results in
 * \verbatim [3, 2, 1, 0, 7, 6, 5, 4] \endverbatim.
 *
 * This implies a portability issue. The swizzles can only work on vectors where Size is a
 * multiple of four.
 * On Vc::Scalar all swizzles are implemented as no-ops. If a swizzle is used on a vector of Size ==
 * 2 compilation will fail.
 */
//@{
/// Identity.
const VECTOR_TYPE abcd() const;
/// Permute pairs.
const VECTOR_TYPE badc() const;
/// Permute pairs of two / Rotate twice.
const VECTOR_TYPE cdab() const;
/// Broadcast a.
const VECTOR_TYPE aaaa() const;
/// Broadcast b.
const VECTOR_TYPE bbbb() const;
/// Broadcast c.
const VECTOR_TYPE cccc() const;
/// Broadcast d.
const VECTOR_TYPE dddd() const;
/// Rotate three: cross-product swizzle.
const VECTOR_TYPE bcad() const;
/// Rotate left.
const VECTOR_TYPE bcda() const;
/// Rotate right.
const VECTOR_TYPE dabc() const;
/// Permute inner pair.
const VECTOR_TYPE acbd() const;
/// Permute outer pair.
const VECTOR_TYPE dbca() const;
/// Reverse.
const VECTOR_TYPE dcba() const;
//@}

/**
 * \name Shift and Rotate
 *
 * These functions allow to shift or rotate the entries in a vector by the given \p amount. Both
 * functions support positive and negative numbers for the shift/rotate value.
 *
 * Example:
 * \code
 * using namespace Vc;
 * int_v foo = int_v::IndexesFromZero() + 1; // e.g. [1, 2, 3, 4] with SSE
 * int_v x;
 * x = foo.shifted( 1); // [2, 3, 4, 0]
 * x = foo.shifted( 2); // [3, 4, 0, 0]
 * x = foo.shifted( 3); // [4, 0, 0, 0]
 * x = foo.shifted( 4); // [0, 0, 0, 0]
 * x = foo.shifted(-1); // [0, 1, 2, 3]
 * x = foo.shifted(-2); // [0, 0, 1, 2]
 * x = foo.shifted(-3); // [0, 0, 0, 1]
 * x = foo.shifted(-4); // [0, 0, 0, 0]
 *
 * x = foo.rotated( 1); // [2, 3, 4, 1]
 * x = foo.rotated( 2); // [3, 4, 1, 2]
 * x = foo.rotated( 3); // [4, 1, 2, 3]
 * x = foo.rotated( 4); // [1, 2, 3, 4]
 * x = foo.rotated(-1); // [4, 1, 2, 3]
 * x = foo.rotated(-2); // [3, 4, 1, 2]
 * x = foo.rotated(-3); // [2, 3, 4, 1]
 * x = foo.rotated(-4); // [1, 2, 3, 4]
 * \endcode
 *
 * These functions are slightly related to the above swizzles. In any case, they are often useful for
 * communication between SIMD lanes or binary decoding operations.
 */
//@{
/// Shift vector entries to the left by \p amount; shifting in zeros.
const VECTOR_TYPE shifted(int amount) const;
/// Rotate vector entries to the left by \p amount.
const VECTOR_TYPE rotated(int amount) const;
//@}

/**
 * Return a sorted copy of the vector.
 *
 * \return A sorted vector. The returned values are in ascending order:
   \verbatim
   v[0] <= v[1] <= v[2] <= v[3] ...
   \endverbatim
 *
 * Example:
 * \code
 * int_v v = int_v::Random();
 * int_v s = v.sorted();
 * std::cout << v << '\n' << s << '\n';
 * \endcode
 *
 * With SSE the output would be:
 *
   \verbatim
   [1513634383, -963914658, 1763536262, -1285037745]
   [-1285037745, -963914658, 1513634383, 1763536262]
   \endverbatim
 *
 * With the Scalar implementation:
   \verbatim
   [1513634383]
   [1513634383]
   \endverbatim
 */
VECTOR_TYPE sorted() const;
