#!/bin/bash

cd "`dirname "$0"`/.."

# Read version number
ver=(`egrep 'define +Vc_VERSION_NUMBER \(Vc_VERSION_CHECK\(' Vc/version.h|tr -dc '[0-9 ]'`)
oldVersion=${ver[0]}.${ver[1]}.${ver[2]}
newVersion=${ver[0]}.${ver[1]}.$((${ver[2]}+1))
echo    "current version: $oldVersion"
echo -n "    new version: "
read -e -i "$newVersion" newVersion

versionString="$newVersion-dev"
versionNumber=`echo $newVersion | sed 's/\./, /g'`

sed -i "s/^PROJECT_NUMBER         = .*\$/PROJECT_NUMBER         = $versionString/" doc/Doxyfile
sed -i \
    -e "s/Vc_VERSION_STRING \".*\"\$/Vc_VERSION_STRING \"$versionString\"/" \
    -e "s/Vc_VERSION_NUMBER (Vc_VERSION_CHECK(.*\$/Vc_VERSION_NUMBER (Vc_VERSION_CHECK($versionNumber) + 1)/" \
    Vc/version.h
