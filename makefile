PYLINT = pylint
JUPYTEXT = jupytext
BLACK = black
NB_DIR = notebooks

# List all Python source files in the notebooks subfolder
NB_PY_FILES = $(wildcard $(NB_DIR)/*.py)

.PHONY: all

all: pylint

# Run pylint on all source files
pylint-notebooks:
	$(PYLINT) $(NB_PY_FILES)

black-notebooks:
	jupytext --sync $(NB_PY_FILES)
	$(BLACK) $(NB_PY_FILES)
	jupytext --sync $(NB_PY_FILES)