#!/bin/zsh -e

setopt extendedglob
IFS="\n\0"
seen=()

parse_file() {
  dir=${1%/*}
  file=${1##*/}
  case "$file" in
    memory.h) return;;
    memorybase.h) return;;
    *deprecated*) return;;
    *debug.h) return;;
  esac
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
  do_skip=false
  while read -r line; do
    #match='*include*'
    case "$line" in
      \ #\#\ #include\ #\"*\"*)
        echo $line|cut -d\" -f2|read include
        parse_file "$include"
        ;;
      *"@BEGIN SKIP GODBOLT@"*) do_skip=true ;;
      *"@END SKIP GODBOLT@"*) do_skip=false ;;
      *)
        $do_skip || printf "%s\n" "$line"
        ;;
    esac
  done < "$file"
  [[ "$1" =~ "/" ]] && popd || true
}

cd `dirname $0`

parse_file "Vc/vector.h" > >(cpp -dD -E -fpreprocessed -w -P | sed 's/^ *//' > godbolt/Vc)
parse_file "Vc/algorithm" > >(cpp -dD -E -fpreprocessed -w -P | sed 's/^ *//' > godbolt/algorithm)
parse_file "Vc/Vc" > >(cpp -dD -E -fpreprocessed -w -P | sed 's/^ *//' > godbolt/containers)
