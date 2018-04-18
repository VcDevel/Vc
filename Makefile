build_dir := $(shell $(PWD)/scripts/build_dir.sh)

$(build_dir)/Makefile:
	@test -n "$(build_dir)"
	@mkdir -p "$(build_dir)"
	@test -e "$(build_dir)/Makefile" || cmake -H. -B"$(build_dir)"

%:: $(build_dir)/Makefile
	$(MAKE) --no-print-directory -C "$(build_dir)" $(MAKECMDGOALS)
