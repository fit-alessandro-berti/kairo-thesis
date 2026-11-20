MDPI_URL := https://mdpi-res.com/data/MDPI_template_ACS.zip?v=20261112
MDPI_SHA256 := f6d5ae546a3d246bd73ee1dbaefc60a8b7398496ef7cfc8e10b7cc83d26067b7

.PHONY: all
all: Definitions/mdpi.cls
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=.build main.tex
	cp .build/main.pdf main.pdf

Definitions/mdpi.cls:
	mkdir -p .build
	curl --fail --location --silent --show-error '$(MDPI_URL)' -o .build/mdpi.zip
	printf '%s  %s\n' '$(MDPI_SHA256)' '.build/mdpi.zip' | sha256sum --check
	unzip -oq .build/mdpi.zip 'Definitions/*'
