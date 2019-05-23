# bf_template_match
Brute force image template matching for expansion  microscopy

Find the similarity transform (rotation, scaling and translation) of a post-
expansion image to optimally match a sub-region in the pre-expansion image,
using normalized cross-correlation. A search range for the linear expansion- 
factor (scaling) and angles (rotation) is searched in parallel to compute the
best matching transformation and outputs the best matching image-regions as tif.

## Installation
---

1. Recommended: download and install Python 3.6 using the Anaconda python distribution 
   from https://www.anaconda.com/download/. Anaconda already comes with 
   most of the required packages.
2. Download bf_template_match and/or extract the bf_template_match package to path/to/dir
3. Open (Anaconda) prompt and type 
```
cd path/to/dir
```
4. Install additional requirements
```
conda install -c conda-forge tifffile
pip install -r requirements.txt
```
Note for windows users: If you experience errors installing the tiffffile library, 
download and install a pre-built .whl file from [Christoph Gohlke's repository](https://www.lfd.uci.edu/~gohlke/pythonlibs/#tifffile) 

5. Install the bf_template_match package
```
python setup.py install
```
6. Quick test
```
bf_template_match --help
```
If all went fine, the command line help is displayed (see below)


## Run on demo data
---
1. Open (Anaconda) prompt
```
cd path/to/dir
bf_template_match -a 1-360:2 -ef 7.9-8.1:0.1 demo/img_post.tif demo/img_pre.tif
```
2. This will search for the optimal alignment of `img_post.tif` having a pixel size of `1 um` and
`img_pre.tif` having a pixel size of `1 um` (default). The applied post-expansion image smoothing 
has sigma 1 (default). Search for three expansion factors of `7.9, 8, 8.1` and 
180 rotation angles of `1, 3, 5, ..., 357, 359` using all available CPUs.

3. After roughly 15 min. you should see the following output:

```
--------------------------------------------------------------------
--- bf_template_match 0.2.0 ----------------------------------------
--------------------------------------------------------------------
 - Arguments:
  -- img_post                   demo/img_post.tif
  -- img_pre                    demo/img_pre.tif
  -- ef                         7.9-8.1:0.1
  -- a                          1-360:2
  -- ds                         1
  -- post_smooth                1
  -- post_um                    1
  -- pre_um                     1
  -- n_cpus                     12
--------------------------------------------------------------------
 - Loading pre-expansion image
 - Loading post-expansion image
 - Using initial post-expansion down scaling 0.12658227848101264
 - Downsample pre-expansion image with 1.000
 - Downsample and smooth post-expansion image with sigma 1.000
 = BF-template match on 12 CPUS: 100%|███████████████████████████████████████████████| 540/540 [10:56<00:00,  1.22s/it]
 - took 656.4738886356354 sec.
 - best match with correlation with corr 0.561 and final expansion-factor 8.000

 - Found post-expansion image at upper-left corner = (1375, 24) with height, width = (341, 341) and rotation angle 155.0

 - Exporting image to 'demo\best_match_c0.561_ef8.000_a155.0_low-res.tif'
  -- Use working scale from pre-expansion
 - Exporting image to 'demo\best_match_c0.561_ef8.000_a155.0_high-post-res.tif'
  -- Upscaling pre-expansion image to post-expansion resolution
```

Inspect the output image `best_match_c0.560_ef8.000_a155.0_high-post-res.tif` in the demo folder. The tiffile contains the pre-expansion image (first channel) and the optimally 
transformed post-expansion image. The pixel-size is the pixel-size of the post-expansion 
image.

## Usage: more command line arguments
---
```
usage: bf_template_match [-h] -ef EF [-a A] [-ds DS]
                         [-post_smooth POST_SMOOTH] [-post_um POST_UM]
                         [-pre_um PRE_UM] [-n_cpus N_CPUS]
                         img_post img_pre

Brute force image template matching for expansion microscopy
--------------------------------------------------------------------------------

Find the similarity transform (rotation, scaling and translation) of a post-
expansion image to optimally match a sub-region in the pre-expansion image,
using normalized cross-correlation. A search range for the linear expansion-
factor (scaling) and angles (rotation) is searched in parallel to compute the
best matching transformation and outputs the best matching image-regions as tif.

Output: best_match_<found_parameters>.tif containing the matched image pair:
image_pre and image_post at working and post-image resolution.

positional arguments:
  img_post              Post-expansion image as tif file
  img_pre               Pre-expansion image as tif file

required arguments:
  -ef EF                The search range for the linear expansion-factor in the format

                          start-stop:step (e. g.: 8-10:0.1)

                        The expansion-factor range '8-10:0.1' computes optimal transformations for
                        factors of 8, 8.1, 8.2, ..., 10.


optional arguments:
  -a A                  The search range for the rotations in the format

                          start-stop:step (e. g.: 1-360:2)

                        The angles range '1-360:2' computes optimal transformations for
                        rotation angles 1, 3, 5, ..., 359.

  -ds DS                Down-scaling factor applied to the pre- and post expansion image prior to
                        finding the optimal translation (default 1). For big pre-expansion overview images
                        0.25 is a good starting value.
  -post_smooth POST_SMOOTH
                        The post expansion usually depicts a higher resolution than the pre-expansion.
                        It can be useful to apply a Gaussian smoothing to the post-expansion image,
                        prior to find the optimal translation. Sigma of Gaussian in pixel values.
                        (default: 1)
  -post_um POST_UM      The pixel-size in micro-meter of the post-expansion image
  -pre_um PRE_UM        The pixel-size in micro-meter of the pre-expansion image
  -n_cpus N_CPUS        Number of CPUS used for parallel execution (default: number of available CPUS)
```

## Distortion analysis
---
For the local distortion analysis we provide an IPython notebook 'distortion_analysis.ipynb' located in the demo folder.

Start Jupyter-Notebook from your Anaconda installation and open the 'distortion_analysis.ipynb'.

This notebook uses the result of 'bf_template_match' to quantify the extent of local distortions.

The processing steps are:
1. Import required Python modules
2. Define helper functions 
3. Specify input parameters
4. Compute local distortion field (from pre- to post-expansion image)
5. Calculates the magnitude and angular histograms of the distortion field

Each code block can be executed by "SHIFT+Enter".


