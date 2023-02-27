KERNEL_NAME = Python3
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

run-notebook:
	cd $(NB_DIR) && \
	papermill "$(IN_NOTEBOOK)" "$(call add_suffix,$(basename $(IN_NOTEBOOK)),'_view.ipynb')" -k $(KERNEL_NAME)

test-notebook:
	cd $(NB_DIR) && \
	papermill "$(IN_NOTEBOOK)" "$(call add_suffix,$(basename $(IN_NOTEBOOK)),_test.ipynb)" -p FAST_EXECUTION 1 -k $(KERNEL_NAME)

# Run Papermill on all .ipynb files in the notebooks directory
view-all-notebooks: clean_views
	cd $(NB_DIR) && \
	for nb in *.ipynb; do \
		papermill "$$nb" "$$(basename "$$nb" .ipynb)_view.ipynb" -k $(KERNEL_NAME); \
	done

clean_views:
	rm -f $(NB_DIR)/*_view.ipynb

# Define a function to add a suffix to a string
define add_suffix
$(1)$(2)
endef