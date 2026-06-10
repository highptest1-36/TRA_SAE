#!/usr/bin/env bash
# setup_texlive.sh — Cài TeX Live + poppler để compile bản Expert Systems trong Colab.
# CHỈ cần chạy 1 lần sau khi Colab (re)connect (TeX Live KHÔNG nằm trên Drive, mất khi restart).
#   bash paper/bin/setup_texlive.sh
set -e
echo "[setup] apt update..."
apt-get update -qq
echo "[setup] installing TeX Live (vài GB, ~5-15 phút)..."
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  texlive-latex-base texlive-latex-recommended texlive-latex-extra \
  texlive-fonts-recommended texlive-fonts-extra texlive-science \
  texlive-pictures texlive-bibtex-extra texlive-plain-generic latexmk \
  poppler-utils
echo "[setup] DONE. pdflatex: $(which pdflatex) | bibtex: $(which bibtex)"
