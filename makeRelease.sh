#!/bin/sh

cd "`dirname "$0"`"

# Read version number
version=`grep PROJECT_NUMBER Doxyfile|cut -d= -f2|cut -c2-`
read -p "Last version: $version. New version: " version

# Update the version number
sed -i "s/^PROJECT_NUMBER         = .*\$/PROJECT_NUMBER         = $version/" Doxyfile
git commit Doxyfile -m"change version to $version"
git tag "$version"

# Create tarball
git archive --format=tar --prefix="Vc-$version/" "$version" | gzip > ../"Vc-$version.tar.gz"

# Update the version number of the after-release code
version="$version-dev"
sed -i "s/^PROJECT_NUMBER         = .*\$/PROJECT_NUMBER         = $version/" Doxyfile
git commit Doxyfile -m"change version to $version"
