#pragma once

#include "sift_cuda/types/HostImage.hh"

namespace sift_cuda {
    /*
     * Get a Gaussian Kernel in the 2D style
     */
    Imagef getGaussianKernel(float sigma_x);

    /*
     * Get a Gaussian Kernel in the 1D format, of which would produce the same outcome when
     *   applied over an image twice (horizontal + vertical) as the above.
     */
    Imagef get1DGaussian(float sigma);
}
