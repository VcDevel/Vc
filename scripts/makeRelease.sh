#!/bin/bash

cd "`dirname "$0"`/.."

# Read version number
version=`grep 'SIMD_VERSION_STRING "[0-9.]\+"' experimental/bits/simd_detail.h|cut -d'"' -f2`
echo "new release version: $version"
echo -n "correct? [y/n]"
read -n 1 ok
if [[ "$ok" != "y" ]]; then
  echo -e "\nPlease modify experimental/bits/simd_detail.h accordingly."
  exit 1
fi

git tag -m "std::experimental::simd $version release" -s "simd-$version" || exit

# Create tarball
git archive --format=tar --prefix="simd-$version/" "$version" | gzip > ../"simd-$version.tar.gz"
