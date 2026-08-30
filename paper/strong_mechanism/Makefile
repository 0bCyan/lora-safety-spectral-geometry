PYTHON ?= python
ROOT := $(abspath ../..)
DERIVED := $(ROOT)/paper/strong_mechanism/data/derived

.PHONY: validate test commands analysis figures tables paper smoke all clean

validate:
	$(PYTHON) $(ROOT)/paper/strong_mechanism/scripts/validate_study.py

test:
	cd $(ROOT) && $(PYTHON) -m unittest discover -s paper/strong_mechanism/tests -p 'test_*.py'

commands:
	mkdir -p $(DERIVED)
	$(PYTHON) $(ROOT)/paper/strong_mechanism/scripts/render_run_commands.py \
		--output $(DERIVED)/run_commands.txt \
		--manifest $(DERIVED)/run_commands.json

analysis:
	$(PYTHON) $(ROOT)/paper/strong_mechanism/scripts/aggregate_results.py

figures:
	$(PYTHON) $(ROOT)/paper/strong_mechanism/scripts/plot_main_figures.py

tables:
	$(PYTHON) $(ROOT)/paper/strong_mechanism/scripts/build_tables.py

paper:
	$(MAKE) -C $(ROOT)/paper/strong_mechanism/latex all

smoke: validate test commands

all: validate test commands analysis figures tables paper

clean:
	rm -rf $(ROOT)/paper/strong_mechanism/latex/build
	find $(ROOT)/paper/strong_mechanism -type d -name __pycache__ -prune -exec rm -rf {} +
