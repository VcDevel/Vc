file(READ "${metafile}" data)
string(REPLACE "R Graphics Output" "${name} benchmark" data "${data}")
file(WRITE "${metafile}" "${data}")
