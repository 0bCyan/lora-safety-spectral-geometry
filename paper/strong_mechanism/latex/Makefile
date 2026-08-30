LATEXMK ?= latexmk
BUILD := build
MAIN := main.tex

.PHONY: all clean check

all:
	mkdir -p $(BUILD)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD) $(MAIN)

check: all
	! grep -E 'undefined references|Citation.*undefined|There were undefined' $(BUILD)/main.log

clean:
	$(LATEXMK) -C -outdir=$(BUILD) $(MAIN) || true
	rm -rf $(BUILD)
