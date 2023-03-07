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

black-notebooks: sync-notebooks
	$(BLACK) $(NB_PY_FILES)
	jupytext --sync $(NB_PY_FILES)

sync-notebooks:
	jupytext --sync $(NB_PY_FILES)

test-notebook:
	cd $(NB_DIR) && \
	papermill "$(IN)" "TEST_$(IN)" \
	-f "parameters/TEST_params_$(basename $(IN)).yaml" \
	--language python \
	--kernel $(KERNEL_NAME) 

test-notebooks: sync-notebooks
	cd $(NB_DIR) && \
	for nb in nb_*.ipynb; do \
		papermill "$$nb" "TEST_""$$nb" \
		-f "parameters/TEST_params_$$(basename $$nb .ipynb).yaml" \
		--language python \
		--kernel $(KERNEL_NAME); \
	done

# Run Papermill on all .ipynb files in the notebooks directory
view-notebooks: clean_views sync-notebooks
	cd $(NB_DIR) && \
	for nb in nb_*.ipynb; do \
		papermill "$$nb" "VIEW_""$$nb" \
		-f "parameters/VIEW_params_$$(basename $$nb .ipynb).yaml" \
		--language python \
		--kernel $(KERNEL_NAME); \
	done

clean_views:
	rm -f $(NB_DIR)/*_view.ipynb

list-notebooks:
	cd $(NB_DIR) && \
	for nb in nb_*.ipynb; do \
		echo "$$nb"; \
	done

# Define a function to add a suffix to a string
define add_suffix
$(1)$(2)
endef