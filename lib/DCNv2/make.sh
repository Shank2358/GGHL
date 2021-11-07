#!/usr/bin/env bash
rm *.so
rm -r build/
python setup.py build develop
