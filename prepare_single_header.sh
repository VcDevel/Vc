#!/bin/zsh -e

start=$1
if [[ ! -f "$start" ]]; then
  echo "Usage: $0 <header-file>"
  exit 1
fi

setopt extendedglob
IFS="\n\0"
seen=()

parse_file() {
  dir=${1%/*}
  file=${1##*/}
  [[ "$file" =~ "deprecated" ]] && return
  [[ "$1" =~ "/" ]] && pushd "$dir"
  #echo "${seen[@]}"
  if [[ "$PWD/$file" = ${(~j.|.)seen} ]]; then
    [[ "$1" =~ "/" ]] && popd || true
    return
  fi
  case "$file" in
     *generalinterface.h);;
     *loadinterface.h);;
     *storeinterface.h);;
     *gatherinterface.h);;
     *scatterinterface.h);;
     *)
        seen=(${seen[@]} "$PWD/$file")
        ;;
  esac
  while read -r line; do
    #match='*include*'
    case "$line" in
      \ #\#\ #include\ #\"*\"*)
        echo $line|cut -d\" -f2|read include
        parse_file "$include"
        ;;
      *)
        printf "%s\n" "$line"
        ;;
    esac
  done < "$file"
  [[ "$1" =~ "/" ]] && popd || true
}

parse_file "$start" | cpp -dD -E -fpreprocessed -w -P | sed 's/^ *//'
