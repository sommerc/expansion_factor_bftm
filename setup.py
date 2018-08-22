# -*- coding: utf-8 -*-
 
"""setup.py: setuptools control."""
 
import re
from setuptools import setup
 
 
version = re.search(
    '^__version__\s*=\s*"(.*)"',
    open('bf_template_match/bf_template_match.py').read(),
    re.M
    ).group(1)
 
 
with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")

description = \
"""
Brute force image template matching for expansion  microscopy
--------------------------------------------------------------------------------

Find the similarity transfrom (rotation, scaling and translation) of a post-
expansion image to optimally match a sub-region in the pre-expansion image,
using normalized cross-correlation. A search range for the linear expansion- 
factor (scaling) and angles (rotation) is searched in parallel to compute the
best matching transformation and outputs the best matching image-regions as tif.

Output: best_match_<found_parameters>.tif containing the matched image pair:
image_pre and image_post at working and post-image resolution.

"""
 
setup(
    name = "bf_template_match",
    packages = ["bf_template_match"],
    entry_points = {
        "console_scripts": ['bf_template_match = bf_template_match.bf_template_match:main']
        },
    version = version,
    description = description,
    long_description = long_descr,
    author = "Christoph Sommer",
    author_email = "christoph.sommer@gmail.com",
    )