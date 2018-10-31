# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import tqdm
import numpy
import argparse
import tifffile
from scipy.ndimage.interpolation import rotate
from multiprocessing import Pool, RawArray, cpu_count
from skimage import feature, transform, exposure, filters

import warnings
warnings.filterwarnings("ignore")

__author__ = "christoph.sommer@ist.ac.at"
__licence__ = "GPLv3"
__version__ = "0.2.0"

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

img_post_description = "Post-expansion image as tif file"
img_pre_description  = "Pre-expansion image as tif file"

ef_description = \
"""The search range for the linear expansion-factor in the format 

  start-stop:step (e. g.: 8-10:0.1)

The expansion-factor range '8-10:0.1' computes optimal transformations for
factors of 8, 8.1, 8.2, ..., 10.

"""

angle_description = \
"""The search range for the rotations in the format 

  start-stop:step (e. g.: 1-360:2)

The angles range '1-360:2' computes optimal transformations for
rotation angles 1, 3, 5, ..., 359.

"""

ds_description = \
"""Down-scaling factor applied to the pre- and post expansion image prior to 
finding the optimal transformation (default 1.). For big pre-expansion overview 
images 0.25 is a good starting value.
"""

post_smooth_description = \
"""The post expansion usually depicts a higher resolution than the pre-expansion.
It can be usefull to apply a Gaussian smoothing to the post-expansion image, 
prior to find the optimal transformation. Sigma of Gaussian in pixel values.
(default: 1.)
"""

post_um_description = "The pixel-size in micro-meter of the post-expansion image"
pre_um_description = "The pixel-size in micro-meter of the pre-expansion image"

n_cpus_description = "Number of CPUS used for parallel execution (default: number"\
"of available CPUS)"


var_dict = {}

def init_worker(img_pre, img_pre_shape, img_post, img_post_shape, srange, arange):
    """
    Function to initialize the workers, passing the sizes of the input arrays
    """
    var_dict['IMG_PRE'] = img_pre
    var_dict['IMG_PRE_SHAPE'] = img_pre_shape

    var_dict['IMG_POST'] = img_post
    var_dict['IMG_POST_SHAPE'] = img_post_shape

def transform_scale_rot(img, s, a):
    """
    Function to scale and rotate the array 'img' by scale 's' and angle 'a'
    """
    if s != 1:
        img_s = transform.rescale(img, s, preserve_range=True)
    else:
        img_s = img
    
    img_s_a = rotate(img_s, -a, reshape=False)
    return img_s_a

def worker_func(arg):
    """
    Main function for processing. Inputs are scale 's' and angle 'a'. The post-expansion image
    is transformed and the best match agains the pre-expansion image is computed and returned.
    """
    s, a = arg
    img_pre  = numpy.frombuffer(var_dict['IMG_PRE'],  dtype=numpy.uint8).reshape(var_dict['IMG_PRE_SHAPE'])
    img_post = numpy.frombuffer(var_dict['IMG_POST'], dtype=numpy.uint8).reshape(var_dict['IMG_POST_SHAPE'])

    img_s_a = transform_scale_rot(img_post, s, a)
    match = feature.match_template(img_pre, img_s_a)
    
    best_correlation = match.max()
    # at
    y0, x0 = numpy.unravel_index(numpy.argmax(match, axis=None), match.shape)
    h, w = img_s_a.shape

    return best_correlation, (y0, x0), (h, w), (s, a)

def load_pre_expansion(pre_fn):
    """
    Helper function to read the pre-expansion image
    """
    print(" - Loading pre-expansion image")
    return tifffile.imread(pre_fn)
     

def prepare_pre_expansion(img_pre_expansion, down_sample_factor):
    """
    Helper function to normalize the image intensities and scale of the pre-expansion image
    """
    print(" - Downsample pre-expansion image with {:4.3f}".format(down_sample_factor))
    pre_overview_xd = transform.rescale(img_pre_expansion, down_sample_factor, preserve_range=True)

    pre_xd = exposure.rescale_intensity(pre_overview_xd, "image", "uint8").astype(numpy.uint8)
    return pre_xd

def load_post_expansion(post_fn):
    """
    Helper function to read the post-expansion image
    """
    print(" - Loading post-expansion image")
    return tifffile.imread(post_fn)

def prepare_post_expansion(img_post_expansion, down_sample_factor, smooth_sigma=1):
    """
    Helper function to normalize the image intensities and scale of the post-expansion image.
    Additionally, the post expansion image is slightly smoothed by an Gaussian filter
    """
    print(" - Downsample and smooth post-expansion image with sigma {:4.3f}".format(smooth_sigma))
    post_16xd = transform.rescale(img_post_expansion, down_sample_factor, preserve_range=True)
    post_16xd_smooth = filters.gaussian(post_16xd, smooth_sigma, preserve_range=True)
    post_16xd = exposure.rescale_intensity(post_16xd_smooth, "image", "uint8").astype(numpy.uint8)
    return post_16xd


def export_matching_images(img_pre, img_post, ul, hw, sa, pixel_size, out_fn):
    """
    Helper function for exporting results.
    """
    print(" - Exporting image to '{}'".format(out_fn))
    (y0, x0) = ul
    (h, w) = hw
    (s,a) = sa

    img_pre_cut = img_pre[y0:y0+h, x0:x0+w]
    img_post_sa = transform_scale_rot(img_post, s, a)

    if img_pre_cut.shape == img_post_sa.shape:
        print("  -- Use working scale from pre-expansion")
    else:
        ## only needed for export high-res output
        ## there might be slight shape mis-match for non-square images
        print("  -- Upscaling pre-expansion image to post-expansion resolution")
        img_pre_cut = transform.resize(img_pre_cut, img_post_sa.shape, clip=True, preserve_range=True)


    combi = numpy.stack([img_pre_cut, img_post_sa])
    tifffile.imsave(out_fn, combi[:, None, ...].astype(numpy.float32), imagej=True, 
                                                                       resolution=(1/pixel_size, 1/pixel_size),
                                                                       metadata={'spacing': 1, 
                                                                                 'unit': 'um', 
                                                                                 'Composite mode': 'composite'})

def get_args():
    """
    Helper function for the argument parser.
    """
    parser = argparse.ArgumentParser(
        description=description,
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        formatter_class=argparse.RawTextHelpFormatter
        )

    parser._action_groups.pop()

    # Add arguments
    parser.add_argument('img_post', help=img_post_description)
    parser.add_argument('img_pre' , help=img_pre_description)

    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument('-ef', type=str, action='store', required=True, help=ef_description)

    optionalNamed = parser.add_argument_group('optional arguments')
    optionalNamed.add_argument('-a', type=str, action='store', default="1-360:2", help=angle_description)
    optionalNamed.add_argument('-ds', type=float, action='store', default=1, help=ds_description)
    optionalNamed.add_argument('-post_smooth', type=float, action='store', default=1, help=post_smooth_description)

    optionalNamed.add_argument('-post_um', type=float, action='store', default=1, help=post_um_description)
    optionalNamed.add_argument('-pre_um', type=float, action='store', default=1, help=pre_um_description)

    optionalNamed.add_argument('-n_cpus', type=int, action='store', help=n_cpus_description)
    
    args = parser.parse_args()
    check_args(args)
    return args

def match_str_range(str_range):
    """
    Helper function to parse specified scale and angle ranges
    """
    str_range = str_range.strip()
    match = re.match("(?P<start>{fl})-(?P<stop>{fl}):(?P<step>{fl})".format(fl=r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'), str_range)
    assert match is not None, "Range '{}' incorrect".format(str_range)

    mdict = match.groupdict()
    assert float(mdict["start"]) <= float(mdict["stop"]), "Start value has to be <= than stop value"
    assert float(mdict["step"]) >= 0, "Step value has to be positive"
    
    return numpy.arange(float(mdict["start"]), float(mdict["stop"]) + float(mdict["step"])/2, float(mdict["step"]))


def check_args(args):
    """
    Helper function for basic sanity checks of input parameters
    """
    assert os.path.exists(args.img_post), "specified post-expansion image does not exist '{}'".format(args.img_post)
    assert os.path.exists(args.img_pre),  "specified pre-expansion image does not exist '{}'".format(args.img_pre)
    assert args.post_smooth > 0, "post-expansion smoothing sigma has to be greater than 0"
    assert 0 < args.ds <= 1, "down-scaling factor has to be: 0 < ds <= 1"

    if args.n_cpus is None:
        args.n_cpus = cpu_count()

    print()
    print("--------------------------------------------------------------------")
    print("--- bf_template_match {:5s} ----------------------------------------".format(__version__))
    print("--------------------------------------------------------------------")
    print(" - Arguments:")
    for k, v in vars(args).items():
        print( "  -- {:25s}\t{}".format(k,v))
    print("--------------------------------------------------------------------")

    args.ef = match_str_range(args.ef)
    args.a = match_str_range(args.a)

def main():

    args = get_args()

    img_pre_orig = load_pre_expansion(args.img_pre)
    img_post_orig = load_post_expansion(args.img_post)  

    img_folder = os.path.dirname(args.img_pre)

    img_pre_ds = args.ds

    img_pre_reso_um = args.pre_um
    img_post_reso_um = args.post_um

    img_post_ds = (img_post_reso_um/img_pre_reso_um * img_pre_ds) / args.ef[0]

    print(" - Using initial post-expansion down scaling", img_post_ds)

    def get_s_from_ef(ef):
        """
        Computes the scaling from provided expansion factors incorporating the actual pixel-sizes
        """
        return (img_post_reso_um / img_pre_reso_um) * (img_pre_ds / img_post_ds) / ef
    
    s_range = [get_s_from_ef(ef) for ef in args.ef]
    a_range = args.a

    ### Preparing for parallel execution
    img_pre = prepare_pre_expansion(img_pre_orig, img_pre_ds)
    img_post = prepare_post_expansion(img_post_orig, img_post_ds, smooth_sigma=args.post_smooth)
    
    img_pre_shape = img_pre.shape
    img_post_shape = img_post.shape
    
    # Create raw arrays for sharing with pool
    img_pre_raw  = RawArray('B', img_pre_shape[0] * img_pre_shape[1])
    img_post_raw = RawArray('B', img_post_shape[0] * img_post_shape[1])
    
    # Wrap iamges as an numpy array 
    img_pre_shared = numpy.frombuffer(img_pre_raw, dtype=numpy.uint8).reshape(img_pre_shape)
    img_post_shared = numpy.frombuffer(img_post_raw, dtype=numpy.uint8).reshape(img_post_shape)
    
    # Copy data to our shared array.
    numpy.copyto(img_pre_shared, img_pre)
    numpy.copyto(img_post_shared, img_post)

    # Start the process pool with init_args 
    with Pool(processes=args.n_cpus, initializer=init_worker, initargs=(img_pre_raw, img_pre_shape, img_post_raw, img_post_shape, s_range, a_range)) as pool:
        tic = time.time()   
        sa_range =  [(s,a) for s in s_range for a in a_range]

        result = []
        for res in tqdm.tqdm(pool.imap_unordered(worker_func, sa_range), total=len(sa_range), desc=" = BF-template match on {:d} CPUS".format(args.n_cpus)):
            result.append(res)
        print(" - took", time.time() - tic, "sec.")

        best_correlation, (y0, x0), (h, w), (s, a) = sorted(result, key=lambda xxx: xxx[0], reverse=True)[0]
        exp_factor = (img_post_reso_um / (img_post_ds*s) ) / (img_pre_reso_um / img_pre_ds)
        
        print(" - best match with correlation with corr {:4.3f} and final expansion-factor {:4.3f}".format(best_correlation, exp_factor))
        print("")
        print(" - Found post-expansion image at upper-left corner =", (y0, x0), "with height, width =", (h, w), "and rotation angle", a)
        print("")

        out_fn = os.path.join(img_folder, "best_match_c{:4.3f}_ef{:4.3f}_a{:4.1f}_low-res.tif".format(best_correlation, exp_factor, a))
        pixel_size = img_pre_reso_um / img_pre_ds
        export_matching_images(img_pre, img_post, (y0, x0), (h, w), (s, a), pixel_size, out_fn)

        pixel_size = img_post_reso_um
        out_fn = os.path.join(img_folder, "best_match_c{:4.3f}_ef{:4.3f}_a{:4.1f}_high-post-res.tif".format(best_correlation, exp_factor, a))
        export_matching_images(img_pre_orig, img_post_orig, (int(y0/img_pre_ds), int(x0/img_pre_ds)), (int(h/img_pre_ds), int(w/img_pre_ds)), (s, a), pixel_size, out_fn)
        

       





