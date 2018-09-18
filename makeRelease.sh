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
}' include/Vc/version.h`
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
	include/Vc/version.h
cat include/Vc/version.h

# Modify README.md to link to release docs
ed README.md <<EOF
P
/web-docs/a
* [$versionString release](https://web-docs.gsi.de/~mkretz/Vc-$versionString/)
.
w
EOF

# Don't build tests with make all
sed -i -e 's/#Release# //' CMakeLists.txt
git commit CMakeLists.txt doc/Doxyfile include/Vc/version.h -s -F- <<EOF
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

# Copy API docs to web-docs
rsync -a --del doc/html/ lxpool.gsi.de:web-docs/Vc-$versionString/

# Create API docs tarball
mv doc/html/*.qch "../Vc-${versionString}.qch"
mv doc/html "Vc-docs-$versionString" && tar -czf "../Vc-docs-$versionString".tar.gz "Vc-docs-$versionString"
rm -rf "Vc-docs-$versionString"

# Get back to the state before the tag and fix up the version numbers afterwards
git revert -n HEAD

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
	include/Vc/version.h
git commit CMakeLists.txt doc/Doxyfile include/Vc/version.h -s -F- <<EOF
after release: version $versionString

* change version strings/numbers to $versionString
* enable HTML_TIMESTAMP for doxygen
* build tests with make all
EOF
