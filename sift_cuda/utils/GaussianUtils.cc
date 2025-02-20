#include "GaussianUtils.hh"

#include <algorithm>

namespace sift_cuda {
    Image<float> getGaussianKernel(float sigma_x) {
        
        const int size = int(round(sigma_x * 6 + 1))|1; // Ensure it's odd
        Image<float> kernel;
        kernel.m_image_size = {size, size};
        Vectorf data; data.reserve(size * size);

        float mean = size / 2;
        float cov_double = sigma_x * sigma_x * 2;
        float sum = 0;
        for (float row = 0; row < size; row++) {
            for (float col = 0; col < size; col++) {
                float gaussianValue = expf(
                    -(
                            (powf(row - mean, 2) + powf(col - mean, 2)) / cov_double
                        )
                    );
                sum += gaussianValue;
                data.push_back(gaussianValue);
            }
        }

        std::for_each(
            data.begin(),
            data.end(),
            [&sum](auto& v){
                v = v / sum;
            });

        kernel.m_data = std::make_shared<Vectorf>(data);
        return kernel;
    }

    Imagef get1DGaussian(float sigma_x) {
        const int size = int(round(sigma_x * 6 + 1))|1; // Ensure it's odd

        Image<float> kernel;
        kernel.m_image_size = {1, size};
        Vectorf data; data.reserve(1 * size);

        float mean = size / 2;
        float cov_double = sigma_x * sigma_x * 2;
        float sum = 0;
        for (float row = 0; row < size; row++) {
            float gaussianValue = expf(
                -(
                        (powf(row - mean, 2)) / cov_double
                    )
                );
            sum += gaussianValue;
            data.push_back(gaussianValue);
        }

        std::for_each(
            data.begin(),
            data.end(),
            [&sum](auto& v){
                v = v / sum;
            });

        kernel.m_data = std::make_shared<Vectorf>(data);
        return kernel;
    }
}