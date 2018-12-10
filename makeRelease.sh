#!/bin/bash

cd "`dirname "$0"`"

# Read version number
eval `awk '/Vc_VERSION_NUMBER 0x[0-9]+/ { h=$3 }
END {
major=strtonum(substr(h, 1, 4))
minor=strtonum("0x" substr(h, 5, 2))
patch=strtonum("0x" substr(h, 7, 2)) / 2
printf "oldVersion=\"%d.%d.%d\"\n", major, minor, patch
printf "newVersion=\"%d.%d.%d\"\n", major, minor, patch + 1
}' Vc/version.h`
echo    "current version: $oldVersion"
echo -n "    new release: "
read -e -i "$newVersion" newVersion

versionString=$newVersion
versionNumber=`echo $newVersion | awk '{ split($0, v, "."); printf "0x%02x%02x%02x", v[1], v[2], v[3] * 2 }'`

# Update the version number
sed -i \
	-e "s/^PROJECT_NUMBER         = .*\$/PROJECT_NUMBER         = $versionString/" \
	-e "s/^HTML_TIMESTAMP         = YES/HTML_TIMESTAMP         = NO/" \
	doc/Doxyfile
sed -i \
	-e "s/Vc_VERSION_STRING \".*\"\$/Vc_VERSION_STRING \"$versionString\"/" \
	-e "s/Vc_VERSION_NUMBER 0x.*\$/Vc_VERSION_NUMBER $versionNumber/" \
	Vc/version.h
cat Vc/version.h

# Modify README.md to link to release docs
ed README.md <<EOF
P
/github.io/a
* [$versionString release](https://vcdevel.github.io/Vc-$versionString/)
.
w
EOF

# Don't build tests with make all
sed -i -e 's/#Release# //' CMakeLists.txt
git commit README.md CMakeLists.txt doc/Doxyfile Vc/version.h -s -F- <<EOF
release: version $versionString

* change version strings/numbers to $versionString
* disable HTML_TIMESTAMP for doxygen
* don't build tests with make all
* Add documentation link to Vc-$versionString
EOF

git tag -m "Vc $versionString release" -s "$versionString" || exit

# Create tarball
git archive --format=tar --prefix="Vc-$versionString/" "$versionString" | gzip > ../"Vc-$versionString.tar.gz"

# Create API docs tarball
./makeApidox.sh

# Copy API docs to vcdevel.github.io
git clone --depth 2 git@github.com:VcDevel/vcdevel.github.io && \
cp -a doc/html vcdevel.github.io/Vc-$versionString && \
cd vcdevel.github.io && \
git add Vc-$versionString && \
git commit -m "Add Vc $versionString release docs" && \
git push && \
cd .. && \
rm -r vcdevel.github.io

# Create API docs tarball
mv doc/html/*.qch "../Vc-${versionString}.qch"
mv doc/html "Vc-docs-$versionString" && tar -czf "../Vc-docs-$versionString".tar.gz "Vc-docs-$versionString"
rm -rf "Vc-docs-$versionString"

# Get back to the state before the tag and fix up the version numbers afterwards
git revert -n HEAD
git reset HEAD README.md && git checkout README.md

# Update the version number of the after-release code
versionString="$versionString-dev"
versionNumber=`echo $versionNumber | awk '{ printf "0x%06x", (strtonum($0) + 1) }'`

sed -i \
	-e "s/^PROJECT_NUMBER         = .*\$/PROJECT_NUMBER         = $versionString/" \
	-e "s/^HTML_TIMESTAMP         = YES/HTML_TIMESTAMP         = NO/" \
	doc/Doxyfile
sed -i \
	-e "s/Vc_VERSION_STRING \".*\"\$/Vc_VERSION_STRING \"$versionString\"/" \
	-e "s/Vc_VERSION_NUMBER 0x.*\$/Vc_VERSION_NUMBER $versionNumber/" \
	Vc/version.h
git commit CMakeLists.txt doc/Doxyfile Vc/version.h -s -F- <<EOF
after release: version $versionString

* change version strings/numbers to $versionString
* enable HTML_TIMESTAMP for doxygen
* build tests with make all
EOF
