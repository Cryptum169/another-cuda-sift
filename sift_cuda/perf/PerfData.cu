#include "PerfData.cuh"
#include "sift_cuda/types/CudaMemRAII.cuh"

SerializationImage::SerializationImage(const Imagef& image) {
    data = *image.m_data;
    r = image.m_image_size.row;
    c = image.m_image_size.col;
    z = 1;
    pitch = -1;
}

void SerializationImage::operator=(const sift_cuda::CudaImage& image) {
    Imagef temp;
    image.copyTo(temp);
    data = *temp.m_data;
    r = temp.m_image_size.row;
    c = temp.m_image_size.col;
    z = 1;
    pitch = -1;
}

void SerializationImage::operator=(const sift_cuda::ImageBlock& block) {
    auto dim = block.getDim();
    r = dim.y;
    c = dim.x;
    z = dim.z;

    pitch = block.getPitch();
    int size = r * c * z;
    data.resize(size);

    CUDA_CHECK(
        cudaMemcpy2D(
            data.data(),
            c * sizeof(float),
            block.getData(),
            block.getPitch(),
            c * sizeof(float),
            r * z,
            cudaMemcpyDeviceToHost
        )
    );
}

void SerializationImage::operator=(const sift_cuda::ImageBlock::ImageView& image_slice) {
    auto dim = image_slice.getDim();
    r = dim.y;
    c = dim.x;
    z = -1;

    pitch = image_slice.getPitch();
    int size = r * c;
    data.resize(size);

    CUDA_CHECK(
        cudaMemcpy2D(
            data.data(),
            c * sizeof(float),
            image_slice.getData(),
            pitch,
            c * sizeof(float),
            r,
            cudaMemcpyDeviceToHost
        )
    );
}

bool SerializationImage::operator==(const SerializationImage& other) const {
    bool result{true};
    result &= r == other.r;
    result &= c == other.c;
    result &= z == other.z;
    result &= pitch == other.pitch;

    result &= data.size() == other.data.size();
    if (result) {
        for (size_t idx = 0; idx < data.size(); idx++) {
            if (data[idx] != other.data[idx]) {
                return false;
            }
        }
    }

    return result;
}
