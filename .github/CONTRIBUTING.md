## Copyright and License

Vc is licensed with the [3-clause BSD license](http://opensource.org/licenses/BSD-3-Clause).
Your contributions to Vc must be released under the same license. You must add
your copyright information to the files you modified/added.

## Code Formatting & Style

The recommended way is to format the code according to `clang-format` using the
`.clang-format` file in the repository.

In addition to the `clang-format` style, `if`, `else`, `for`, `while`, and `do`
*must* use braces.

If, for some reason, you cannot use `clang-format`, here's a quick overview of
the style rules:
* Constrain the code to no more than 90 characters per line.
* Use four spaces for indent. No tabs.
* Opening braces attach to the preceding expression, except for functions,
  namespaces, and classes/structs/unions/enums.
* Namespaces introduce no additional indent
* `case` labels are aligned with the `switch` statement
* No more than one empty line.
* No spaces in parentheses, but spaces between keywords and opening paren, i.e.
  `if (foo) { bar(); }`

### Naming Rules

* Naming is very important. Take time to choose a name that clearly explains the
  intended functionality & usage of the entity.
* Type names typically use `CamelCase`. No underscores.
* Function and variable names use `camelCase`. No underscores.
* Acronyms that appear in camel case names must use lowercase letters for all
  characters after the first characters. (e.g. `SimdArray`, `simdFunction`)
* Traits use `lower_case_with_underscores`.
* Macros are prefixed with `Vc_` and use `Vc_ALL_CAPITALS_WITH_UNDERSCORES`.
  Macro arguments use a single underscore suffix.
  Include guards are prefixed with `VC_` instead.
* File names use `alllowercasewithoutunderscores`. Basically, it is the type name
  declared/defined in the file with all letters in lower case.
* There are exceptions and inconsistencies in the code. Don't bother.

### Design Guidelines

* *Avoid out parameters.* Use the return value insted. Use `std::tuple` if you
  need to return multiple values.
* *Look for alternatives to in-out parameters.* An obvious exception (and thus
  design alternative) is the implicit `this` parameter to non-static member
  functions.
* Consequently, *pass function parameters by const-ref or by value.*
  Use const-ref for types that (potentially) require more than two CPU
  registers. (Consider fundamental types and the fundamental `Vector<T>` types
  to require one register, each.)
  By value otherwise.
* *Ensure const-correctness.* Member functions use the `const` qualifier if they
  do not modify observable state. Use `mutable` members for unobservable state.
* *Avoid macros.* Possible alternatives are constexpr variables and template
  code.

## Git History

Git history should be flat, if feasible. Feel free to use merges on your private
branch. However, once you submit a pull request, the history should apply
cleanly on top of master. Use `git rebase [-i]` to straighten the history.

Use different branches for different issues.

## Git Commit Logs

1. Write meaningful summaries and strive to use no more than 50 characters
1. Use imperative mood in the subject line (and possibly in bullet points in the
   summary)
1. Wrap the body at 72 characters
1. Use the body to explain *what* and *why* (normally it is irrelevant *how* you
   did it)

See also [Chris Beams article](http://chris.beams.io/posts/git-commit/).
