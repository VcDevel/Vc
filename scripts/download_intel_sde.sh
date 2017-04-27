#!/bin/sh
login_url=https://software.intel.com/protected-download/267266/144917
download_url=https://software.intel.com/system/files/managed/d7/1b/sde-external-7.58.0-2017-01-23-lin.tar.bz2
cookies=`mktemp` || exit
wget --save-cookies $cookies --keep-session-cookies -q -O - $login_url | \
  grep -A9999 'form-item-accept-license' | \
  grep 'name="form_[ib]' | \
  sed 's/^.* name="\([^"]*\)".* value="\([^"]*\)".*$/\1=\2/' | {
read a
read b
wget --load-cookies $cookies --save-cookies $cookies --keep-session-cookies --post-data "accept_license=1&${a}&${b}&op=Continue" -q -O - $login_url > /dev/null && \
wget --load-cookies $cookies -nv --progress=dot:giga -O sde.tar.bz2 $download_url && \
tar -xf sde.tar.bz2 && \
rm sde.tar.bz2 && \
rm $cookies && \
for dir in sde-*; do
  mv ./$dir/* .
  rmdir ./$dir
done
}
