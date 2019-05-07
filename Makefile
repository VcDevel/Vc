CXX ?= c++
build_dir := $(shell which $(CXX))
tmp := "case $$(readlink -f $(build_dir)) in *icecc) which $${ICECC_CXX:-g++};; *) echo $(build_dir);; esac"
build_dir := $(shell sh -c $(tmp))
build_dir := $(realpath $(build_dir))
build_dir := build-$(subst /,-,$(build_dir:/%=%))
cols := $(shell sh -c 'stty size|cut -d" " -f2')

test: $(build_dir)/CMakeCache.txt
	$(MAKE) --no-print-directory -C "$(build_dir)" all test 2>&1|sed -u 's/std::\(experimental::\([a-z_0-9]\+::\)\?\)\?/⠶/g'|stdbuf -oL fold -s -w $(cols)

%:: $(build_dir)/CMakeCache.txt
	$(MAKE) --no-print-directory -C "$(build_dir)" $(MAKECMDGOALS) 2>&1|sed -u 's/std::\(experimental::\([a-z_0-9]\+::\)\?\)\?/⠶/g'|stdbuf -oL fold -s -w $(cols)

$(build_dir)/CMakeCache.txt:
	@test -n "$(build_dir)"
	@mkdir -p "$(build_dir)"
	@test -e "$(build_dir)/CMakeCache.txt" || cmake -Htests -B"$(build_dir)"

print_build_dir:
	@echo "$(PWD)/$(build_dir)"

clean_builddir:
	rm -rf "$(build_dir)"

# the following rule works around %:: grabbing the Makefile rule and thus stops it from running every time
Makefile:
	@true

install:
	./install.sh

benchmarks:
	cd benchmarks && for i in *.cpp; do ./run.sh $$i; done

benchmark-%:
	cd benchmarks && ./run.sh $*

.PHONY: print_build_dir clean_builddir install benchmarks
