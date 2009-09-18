/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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
 * default constructor
 *
 * Leaves the mask uninitialized.
 */
MASK_TYPE();

/**
 * Constructs a mask with the entries initialized to zero.
 */
explicit MASK_TYPE(Vc::Zero);

/**
 * Constructs a mask with the entries initialized to one.
 */
explicit MASK_TYPE(Vc::One);

/**
 * Constructs a mask with the entries initialized to
 * \li one if \p b is \p true
 * \li zero if \p b is \p false
 */
explicit MASK_TYPE(bool b);

/**
 * Standard copy constructor
 */
MASK_TYPE(const MASK_TYPE &);

/**
 * Constructs a mask object from a mask of different size/type.
 */
template<typename OtherMask> MASK_TYPE(const OtherMask &);

/**
 * Returns whether the two masks are equal in all entries.
 */
bool operator==(const MASK_TYPE &) const;

/**
 * Returns whether the two masks differ in at least one entry.
 */
bool operator!=(const MASK_TYPE &) const;

/**
 * Return the per-entry resulting mask of a logical (in this case same as bitwise) AND operation.
 */
MASK_TYPE operator&&(const MASK_TYPE &) const;
/**
 * Return the per-entry resulting mask of a logical (in this case same as bitwise) AND operation.
 */
MASK_TYPE operator& (const MASK_TYPE &) const;
/**
 * Return the per-entry resulting mask of a logical (in this case same as bitwise) OR operation.
 */
MASK_TYPE operator||(const MASK_TYPE &) const;
/**
 * Return the per-entry resulting mask of a logical (in this case same as bitwise) OR operation.
 */
MASK_TYPE operator| (const MASK_TYPE &) const;
/**
 * Return the per-entry resulting mask of a logical (in this case same as bitwise) XOR operation.
 */
MASK_TYPE operator^ (const MASK_TYPE &) const;
/**
 * Return the per-entry resulting mask of a logical (in this case same as bitwise) NOT operation.
 */
MASK_TYPE operator! () const;
/**
 * Modify the mask per-entry using a logical (in this case same as bitwise) AND operation.
 */
MASK_TYPE operator&=(const MASK_TYPE &);
/**
 * Modify the mask per-entry using a logical (in this case same as bitwise) OR operation.
 */
MASK_TYPE operator|=(const MASK_TYPE &);

/**
 * Return whether all entries of the mask are one.
 */
bool isFull() const;
/**
 * Return whether all entries of the mask are zero.
 */
bool isEmpty() const;
/**
 * Return whether the mask is neither full nor empty.
 */
bool isMix() const;

/**
 * Cast to bool operator. Returns the same as isFull().
 *
 * \warning Be careful with the cast to bool. Often it is better to write explicitly whether you
 * want isFull or !isEmpty or ...
 */
operator bool() const;

/**
 * Return the \p i th entry of the mask as a bool.
 */
bool operator[](int i) const;

/**
 * Return how many entries of the mask are set to one.
 */
int count() const;

/**
 * Returns the index of the first one in the mask.
 */
int firstOne() const;
