#!/bin/bash

cd "`dirname "$0"`/.."

# Read version number
version=`grep 'SIMD_VERSION_STRING "[0-9.]\+"' experimental/bits/simd_detail.h|cut -d'"' -f2`

if git tag|grep -q "^simd-$version$"; then
  echo "The tag 'simd-$version' already exists. Aborting." >&2
  exit 1
fi

echo "new release version: $version"
echo -n "correct? [y/n]"
read -n 1 ok
if [[ "$ok" != "y" ]]; then
  echo -e "\nPlease modify experimental/bits/simd_detail.h accordingly."
  exit 1
fi

git tag -m "std::experimental::simd $version release" -s "simd-$version" || exit

# Make sure git-archive-all (needed for archiving with virtest) is available
which git-archive-all >/dev/null 2>&1 || pip install --user git-archive-all || exit

# Create tarball
git-archive-all -v --prefix="simd-$version/" "../simd-$version.tar.gz" && \
  echo "Release tarball at ../simd-$version.tar.gz"
