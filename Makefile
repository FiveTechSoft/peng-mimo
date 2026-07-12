.PHONY: all glm mimo portable test check check-core cuda-test clean

all glm mimo portable test check check-core cuda-test clean:
	$(MAKE) -C c $@
