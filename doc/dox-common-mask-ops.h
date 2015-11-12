/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>
All rights reserved.

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

/**
 * Denotes the size of the mask (the number of entries).
 */
static constexpr size_t Size;

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
 *
 * \param b Determines the initial state of the mask.
 */
explicit MASK_TYPE(bool b);

/**
 * Standard copy constructor
 */
MASK_TYPE(const MASK_TYPE &);

/**
 * Implicit conversion from a compatible mask object.
 */
template<typename OtherMask> MASK_TYPE(const OtherMask &);

/**
 * Explicit conversion (static_cast) from a mask object that potentially has a different \ref Size.
 */
template<typename OtherMask> explicit MASK_TYPE(const OtherMask &);

/**
 * Explicit conversion from an array of bool to a mask object.
 * This corresponds to a vector load.
 *
 * \param mem A pointer to the start of the array of booleans.
 */
explicit MASK_TYPE(const bool *mem);

/**
 * Load the values of the mask from an array of bool.
 *
 * \param mem A pointer to the start of the array of booleans.
 */
void load(const bool *mem);

/**
 * Store the values of the mask to an array of bool.
 *
 * \param mem A pointer to the start of the array of booleans.
 */
void store(bool *mem) const;

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
 * Read-only access to mask entries.
 *
 * \param i Determines the boolean to be accessed.
 * \return the \p i th entry of the mask as a bool.
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

/**
 * Convert mask to an integer.
 *
 * \return An int where each bit corresponds to the boolean value in the mask.
 *
 * E.g. a mask like [true, false, false, true] would result in a 9 (in binary: 1001).
 */
int toInt() const;
