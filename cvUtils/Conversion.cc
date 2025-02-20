#include "Conversion.hh"

#include "opencv2/core/mat.hpp"

#include <cstring>

namespace OpencvUtils {

    bool areEqual(const cv::Mat& a, const cv::Mat& b)
    {
        bool size = a.channels() == b.channels() && a.rows == b.rows && a.cols == b.cols;
        if (!size) {
            return false;
        }
        
        cv::UMat temp;
        cv::bitwise_xor(a,b,temp);
        return !(cv::countNonZero(temp.reshape(1)));
    }

    std::vector<cv::KeyPoint> localKptToCvKpt(
        const thrust::host_vector<float3>& kpts,
        const thrust::host_vector<float4>& features,
        int size
    ) {
        std::vector<cv::KeyPoint> results{};
        assert(kpts.size() == features.size());

        size_t convert_size = size == -1 ? kpts.size() : size;
        results.reserve(convert_size);

        for (size_t idx = 0; idx < convert_size; idx++) {
            cv::KeyPoint cv_kpt;
            cv_kpt.pt.x = kpts[idx].x;
            cv_kpt.pt.y = kpts[idx].y;
            cv_kpt.octave = features[idx].x;
            cv_kpt.response = features[idx].z;
            cv_kpt.size = features[idx].y;
            cv_kpt.angle = features[idx].w;
            results.push_back(cv_kpt);
        }
        return results;
    }

    std::vector<cv::DMatch> cvtMatchToDMatch(
        const std::vector<int>& match
    ) {
        std::vector<cv::DMatch> result;
        result.reserve(match.size());

        for (int idx = 0; idx < match.size(); idx++) {
            if (match[idx] != -1) {
                result.push_back(cv::DMatch(idx, match[idx], 0.f));
            }
        }

        return result;
    }

}
