#ifndef IMAGE_UTILS_CONVERSIONIMPL_HPP
#define IMAGE_UTILS_CONVERSIONIMPL_HPP
#include "Conversion.hh"

namespace OpencvUtils {

    template <typename DataType_T>
    Image<DataType_T> cvMatToImage(const cv::Mat& mat) {
        if (mat.channels() > 1) {
            throw std::runtime_error("More channels than expected");
        }

        Image<DataType_T> img{};
        img.m_image_size = Size{
            static_cast<float>(mat.rows),
            static_cast<float>(mat.cols)
        };

        std::vector<DataType_T> data;

        if (mat.type() == 6) {
            data = std::vector<DataType_T>(mat.begin<double>(), mat.end<double>());
        } else if (mat.type() == 5) {
            data = std::vector<DataType_T>(mat.begin<float>(), mat.end<float>());
        } else if (mat.type() == 0) {
            data = std::vector<DataType_T>(mat.begin<uchar>(), mat.end<uchar>());
        }

        img.m_data = std::make_shared<std::vector<DataType_T>>(data);
        return img;
    }

    template <typename DataType_T>
    cv::Mat imageToCvMat(Image<DataType_T> image) {

        int cv_data_type;
        if constexpr (std::is_same_v<DataType_T, uint8_t>) {
            cv_data_type = CV_8UC1;
        } else if constexpr (std::is_same_v<DataType_T, float>) {
            cv_data_type = CV_32FC1;
        }
        cv::Mat newMat( int(image.m_image_size.row), int(image.m_image_size.col), cv_data_type);

        std::memcpy(newMat.data, image.m_data->data(), image.m_data->size() * sizeof(DataType_T));
        return newMat;
    }


    template <typename DataType_T>
    Image8U normalize(const Image<DataType_T>& image) {
        DataType_T min = *std::min_element(image.m_data->begin(), image.m_data->end());
        DataType_T max = *std::max_element(image.m_data->begin(), image.m_data->end());

        Image8U retImg{};
        retImg.m_image_size = image.m_image_size;
        retImg.m_data = std::make_shared<std::vector<uint8_t>>();

        DataType_T scale = 255 / (max - min);
        std::for_each(image.m_data->begin(), image.m_data->end(), [&retImg, &scale, &min](const auto v) {
            retImg.m_data->push_back(static_cast<uint8_t>((v - min) * scale));
        });
        return retImg;
    }

    template <typename Data_T>
    cv::Mat descriptorToCvMat(
        const thrust::host_vector<Data_T>& descriptors,
        int num_pts
    ) {
        num_pts = std::min(num_pts, int(descriptors.size() / 128));
        cv::Mat desc(num_pts, 128, CV_32FC1);

        std::transform(
            descriptors.begin(), descriptors.begin() + num_pts * 128, desc.ptr<float>(), [](Data_T d) {
                if constexpr (std::is_same_v<Data_T, half>) {
                    return __half2float(d);
                } else {
                    return static_cast<float>(d);
                }
            }
        );
        return desc;
    }
};

#endif