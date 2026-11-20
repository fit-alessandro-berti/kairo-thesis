# Kairo journal manuscript

This is a LaTeX research article draft in MDPI Applied Sciences format. The manuscript develops the state-based method from Bora Deren's thesis and reports the repository's subsequent comparative assessment.

Run `make -C journal_paper` from the repository root. The build needs `curl`, `unzip`, `sha256sum`, `latexmk`, pdfLaTeX, BibTeX, Ghostscript, and the MDPI class's LaTeX packages. A full TeX Live installation supplies the LaTeX packages. The first build downloads the pinned [official MDPI template](https://www.mdpi.com/authors/latex). The compiled `journal_paper/main.pdf`, downloaded template, and build intermediates are local files.

The [comparative assessment](../assessments/concept-drift/README.md) links the logs, run records, and reproduction instructions. Editable TikZ drawings live in `figures/`. The assessment plot uses the archived window scores and drift annotations. Square placeholders mark the planned dashboard and Copilot screenshots.

The author order, correspondence, contributor roles, funding, conflicts of interest, and disclosure text are draft metadata for author review before submission. The manuscript uses the Applied Sciences format. The [example process mining special issue](https://www.mdpi.com/journal/applsci/special_issues/87G1T64B1M) lists a submission deadline that has passed.
