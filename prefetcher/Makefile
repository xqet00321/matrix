CC ?= gcc
CFLAGS = -msse2 --std gnu99 -O0 -Wall -Wextra -mavx2 -g -I.
LDFLAGS = -lpthread

GIT_HOOKS := .git/hooks/applied
OUT ?= .build

ifdef ARRAY_SIZE_I
CFLAGS += -D ARRAY_SIZE_I=${ARRAY_SIZE_I}
endif

EXEC = \
	   tests/test_matrix \
	   naive_transpose \
	   sse_transpose \
	   sse_prefetch_transpose \
	   avx_transpose \
	   avx_prefetch_transpose \
	   avx_set_transpose \

OBJS := \
		stopwatch.o \
		naive_transpose.o \
		sse_transpose.o \
		sse_prefetch_transpose.o \
                avx_transpose.o \
                avx_set_transpose.o \
		avx_prefetch_transpose.o \

OBJS := $(addprefix $(OUT)/,$(OBJS))
deps := $(OBJS:%.o=%.o.d)

.PHONY: all
all: $(GIT_HOOKS) $(OUT) $(EXEC)

tests/test_matrix: $(OBJS) tests/test_matrix.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%_transpose: $(OBJS) tests/test_matrix.c
	$(CC) $(CFLAGS) -D$(subst _transpose,,$@) -o $@ $^ $(LDFLAGS)

$(OUT)/%.o: impl/%.c $(OUT)
	$(CC) $(CFLAGS) -c -o $@ -MMD -MF $@.d $<

$(OUT):
	@mkdir -p $@

check: $(EXEC)
	@for test in $^ ; \
	do \
		echo "Execute $$test..." ; $$test && echo "OK!\n" ; \
	done
data: $(EXEC)
	for i in `seq 1 100`; do  \
		printf "$$i " >> time.txt ; \
		./naive_transpose >> time.txt ; \
		printf " " >> time.txt; \
		./sse_transpose >> time.txt; \
		printf " " >> time.txt; \
		./sse_prefetch_transpose >> time.txt; \
		printf " " >> time.txt; \
                ./avx_transpose >> time.txt; \
		printf " " >> time.txt; \
		./avx_prefetch_transpose >> time.txt; \
		printf "\n" >> time.txt; \
		printf " " >> time.txt; \
		./avx_set_transpose >> time.txt; \
		printf "\n" >> time.txt; \
	done

plot: data
	gnuplot scripts/plot/plot.gp

pfd_plot: clean
	for i in `seq 1 20`; do \
		$(MAKE) PFDIST=$$i --silent; \
		printf "$$i " >> time.txt ; \
		./sse_prefetch_transpose >> time.txt; \
		printf "\n" >> time.txt ; \
	done
	gnuplot scripts/plot/pfdist_plot.gp

array_size:
	for i in `seq 0 12`; do \
		$(MAKE) ARRAY_SIZE_I=$$i --silent; \
		echo; echo i = $$i ; \
		echo 1 | sudo tee /proc/sys/vm/drop_caches; \
		sudo perf stat --repeat 5 -e cache-misses,cache-references,L1-dcache-load-misses \
		-e L1-dcache-loads,L1-dcache-stores,L1-icache-load-misses \
		-e r02D1,r10D1 \
		./naive_transpose; \
	done

$(GIT_HOOKS):
	@scripts/install-git-hooks
	@echo

clean:
	$(RM) $(EXEC) $(OBJS) $(deps)
	@rm -rf $(OUT)
	rm -f time.png time.txt
-include $(deps)
