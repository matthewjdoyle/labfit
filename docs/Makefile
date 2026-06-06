SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build

# Use a local .venv if present instead of the system sphinx-build.
VENV ?= ../.venv
ifneq ("$(wildcard $(VENV))","")
SPHINXBUILD   = $(VENV)/bin/sphinx-build
endif
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Convenience shortcut
.PHONY: html
html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
