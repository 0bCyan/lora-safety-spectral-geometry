PYTHON ?= python
ROOT := $(abspath ../..)
DERIVED := $(ROOT)/paper/strong_mechanism/data/derived
RAW := $(ROOT)/paper/strong_mechanism/data/raw
AVAILABLE_FIGURES := $(ROOT)/paper/strong_mechanism/figures/available
GENERATED_FIGURES := $(ROOT)/paper/strong_mechanism/figures/generated

.PHONY: validate test commands eval-commands collect-targets analysis figures figures-available figures-export figures-final tables paper smoke preflight all clean

validate:
	$(PYTHON) $(ROOT)/paper/strong_mechanism/scripts/validate_study.py

test:
	cd $(ROOT) && $(PYTHON) -m unittest discover -s paper/strong_mechanism/tests -p 'test_*.py'

commands:
	mkdir -p $(DERIVED)
	$(PYTHON) $(ROOT)/paper/strong_mechanism/scripts/render_run_commands.py \
		--output $(DERIVED)/run_commands.txt \
		--manifest $(DERIVED)/run_commands.json

eval-commands:
	mkdir -p $(DERIVED)
	$(PYTHON) $(ROOT)/paper/strong_mechanism/scripts/render_evaluation_commands.py \
		--output $(DERIVED)/evaluation_commands.txt \
		--manifest $(DERIVED)/evaluation_commands.json

collect-targets:
	mkdir -p $(RAW)/review_targets
	$(PYTHON) $(ROOT)/paper/strong_mechanism/scripts/collect_generation_targets.py \
		--manifest $(DERIVED)/evaluation_commands.json \
		--output-dir $(RAW)/review_targets

analysis:
	$(PYTHON) $(ROOT)/paper/strong_mechanism/scripts/aggregate_results.py

# Always legal before formal training: design figures plus frozen historical evidence.
figures-available:
	$(PYTHON) $(ROOT)/paper/strong_mechanism/scripts/plot_available_figures.py \
		--output-dir $(AVAILABLE_FIGURES) \
		--formats svg

# Local high-resolution export of the currently available assets.
figures-export:
	$(PYTHON) $(ROOT)/paper/strong_mechanism/scripts/plot_available_figures.py \
		--output-dir $(GENERATED_FIGURES)/available \
		--formats svg,pdf,png \
		--dpi 300

# Requires normalized formal result inputs; intentionally fails before they exist.
figures-final:
	$(PYTHON) $(ROOT)/paper/strong_mechanism/scripts/plot_main_figures.py \
		--output-dir $(GENERATED_FIGURES)/results

figures: figures-available

tables:
	$(PYTHON) $(ROOT)/paper/strong_mechanism/scripts/build_tables.py

paper:
	$(MAKE) -C $(ROOT)/paper/strong_mechanism/latex all

# Static pre-GPU checks. Available figures are generated in CI to a temporary directory.
smoke: validate test commands eval-commands

# Pre-result local workflow.
preflight: smoke figures-available paper

# Release workflow. It is expected to fail until formal evidence exists.
all: validate test commands eval-commands analysis figures-available figures-final tables paper

clean:
	rm -rf $(ROOT)/paper/strong_mechanism/latex/build
	rm -rf $(GENERATED_FIGURES)
	find $(ROOT)/paper/strong_mechanism -type d -name __pycache__ -prune -exec rm -rf {} +
