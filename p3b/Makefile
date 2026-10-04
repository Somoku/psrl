# sandboxd: build, install, test.
#
# Two artifacts ship from this directory and they install separately, because
# they are deployed to different places: the Go service runs on every sandbox
# node, and the Python SDK is installed wherever a caller runs. `make all`
# builds both; `make install sdk-install` places both.
#
# PREFIX, DESTDIR, GO, and PIP are overridable because a packager needs them to
# be. DESTDIR in particular is what lets a distribution build into a staging
# root without writing to the live filesystem.

GO      ?= go
PIP     ?= python3 -m pip
PREFIX  ?= /usr/local
DESTDIR ?=
BINDIR   = $(DESTDIR)$(PREFIX)/bin

BINARY  = bin/sandboxd
CONFIG  ?= example.json

# Trim the symbol table and DWARF. A sandbox node runs this, it does not debug
# it, and the stripped binary is roughly a third smaller to ship.
LDFLAGS ?= -s -w

.PHONY: all
all: build

.PHONY: help
help:
	@echo "Service:"
	@echo "  build         compile $(BINARY)"
	@echo "  install       install the binary to $(PREFIX)/bin (override PREFIX, DESTDIR)"
	@echo "  uninstall     remove it again"
	@echo "  preflight     validate CONFIG=$(CONFIG) and exit, without serving"
	@echo "  run           build, then serve CONFIG=$(CONFIG) in the foreground"
	@echo ""
	@echo "Python SDK:"
	@echo "  sdk-install   pip install -e sdk  (the surface a caller imports)"
	@echo "  sdk-check     confirm the installed SDK imports"
	@echo ""
	@echo "Tests:"
	@echo "  test          unit tests, no daemon needed"
	@echo "  test-race     the same under the race detector"
	@echo "  test-rollout  end-to-end against a real Docker daemon (build tag)"
	@echo "  bench         the create-path and allocator benchmarks"
	@echo "  vet fmt       static checks"
	@echo ""
	@echo "  clean         remove bin/ and stray test binaries"

.PHONY: build
build:
	$(GO) build -ldflags '$(LDFLAGS)' -o $(BINARY) ./cmd/sandboxd

# install -D creates the directory, so this works into an empty DESTDIR.
.PHONY: install
install: build
	install -D -m 0755 $(BINARY) $(BINDIR)/sandboxd
	@echo "installed $(BINDIR)/sandboxd"

.PHONY: uninstall
uninstall:
	rm -f $(BINDIR)/sandboxd

# Validates the configuration and every backend it declares, then exits. The
# exit code is the whole answer, which is what a deployment pipeline reads.
.PHONY: preflight
preflight: build
	./$(BINARY) -preflight -config $(CONFIG)

.PHONY: run
run: build
	./$(BINARY) -config $(CONFIG)

# Editable, because the SDK is developed alongside the service it speaks to and
# a copied install would hide a wire-format change until the next reinstall.
.PHONY: sdk-install
sdk-install:
	$(PIP) install -e sdk

.PHONY: sdk-check
sdk-check:
	@python3 -c "from sandboxd import SandboxClient, SandboxSpec, Source, Resources; \
import sandboxd; print('sandboxd SDK:', sandboxd.__file__)"

.PHONY: test
test:
	$(GO) test ./...

.PHONY: test-race
test-race:
	$(GO) test -race ./...

# Behind a build tag because it starts real processes and needs a Docker daemon.
# See tests/rollout/doc.go.
.PHONY: test-rollout
test-rollout:
	$(GO) test -tags rollout ./tests/rollout/ -v -timeout 10m

.PHONY: bench
bench:
	$(GO) test ./... -run XXX -bench . -benchtime 20000x

.PHONY: vet
vet:
	$(GO) vet ./...

.PHONY: fmt
fmt:
	$(GO) fmt ./...

.PHONY: clean
clean:
	rm -rf bin
	find . -name '*.test' -delete
