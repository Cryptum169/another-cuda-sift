#include "CudaImage.cuh"
#include "sift_cuda/types/CudaMemRAII.cuh"

namespace sift_cuda {
    void CudaImage::allocate(int x, int y) {
        if (x == 0 || y == 0) {
            return;
        }
        dim = make_float2(x, y);

        float* ptr;
        CUDA_CHECK(
            cudaMallocPitch(
                &ptr,
                &pitch,
                dim.x * sizeof(float),
                dim.y
            )
        );
        data.reset(ptr);

        allocated = true;
    }

    CudaImage::CudaImage(CudaImage&& other) noexcept {
        if (!other.allocated) {
            return;
        }

        allocated = other.allocated;

        dim = other.dim;
        data = std::move(other.data);
        pitch = other.pitch;
    }

    CudaImage::CudaImage(const Imagef& host) {
        allocate(host.m_image_size.col, host.m_image_size.row);

        CUDA_CHECK(
            cudaMemcpy2D(
                data.get(),
                pitch,
                host.m_data->data(),
                host.m_image_size.col * sizeof(float),
                host.m_image_size.col * sizeof(float),
                host.m_image_size.row,
                cudaMemcpyHostToDevice
            )
        );
    }

    void CudaImage::copyFrom(const CudaImage& other) {
        // if (allocated) {
        // No need to worry, as reset will automatically release currently held data
        // }

        dim = other.dim;
        pitch = other.pitch;
        allocated = other.allocated;

        if (allocated) {
            float *ptr;
            CUDA_CHECK(
                cudaMallocPitch(
                    &ptr,
                    &pitch,
                    dim.x * sizeof(float),
                    dim.y
                )
            );

            CUDA_CHECK(
                cudaMemcpy2D(
                    ptr,
                    pitch,
                    other.getData(), 
                    other.pitch,
                    dim.x * sizeof(float),
                    dim.y,
                    cudaMemcpyDeviceToDevice
                )
            );
            data.reset(ptr);
        }
    }

    CudaImage& CudaImage::operator=(const Imagef& image) {
        // We don't need to check for `allocated` again because unique_ptr::reset will automatically
        // take care of deallocation

        // However overhead is pretty bad
        if (!allocated) {
            allocate(image.m_image_size.col, image.m_image_size.row);
        }

        CUDA_CHECK(cudaMemcpy2D(
            data.get(),
            pitch,
            image.m_data->data(),
            image.m_image_size.col * sizeof(float),
            image.m_image_size.col * sizeof(float),
            image.m_image_size.row,
            cudaMemcpyHostToDevice
        ));

        return *this;
    }

    void CudaImage::copyTo(Imagef& image) const {
        image.m_image_size = {dim.y, dim.x};
        int size = image.m_image_size.col * image.m_image_size.row;
        image.m_data = std::make_shared<std::vector<float>>(size);

        CUDA_CHECK(
            cudaMemcpy2D(
                image.m_data->data(),
                image.m_image_size.col * sizeof(float),
                data.get(),
                pitch,
                dim.x * sizeof(float),
                dim.y, 
                cudaMemcpyDeviceToHost
            );
        );
    }

    void CudaImage::fromHost(const std::vector<float>& host_data, int x, int y) {
        dim = make_float2(x, y);

        float* ptr;
        CUDA_CHECK(
            cudaMallocPitch(
                &ptr,
                &pitch,
                dim.x * sizeof(float),
                dim.y
            )
        );
        data.reset(ptr);

        CUDA_CHECK(
            cudaMemcpy2D(
                ptr,
                pitch, 
                host_data.data(),
                x * sizeof(float),
                x * sizeof(float),
                y, 
                cudaMemcpyHostToDevice
            )
        );

        allocated = true;
    }

    ImageBlock::ImageBlock(ImageBlock&& other) noexcept {
        dim = other.dim;
        pitch = other.pitch;
        allocated = other.allocated;
        data = std::move(other.data);
    }

    void ImageBlock::fromHost(const std::vector<float>& host_data, int r, int c, int z) {
        dim = make_float3(c, r, z);
        float* ptr;
        CUDA_CHECK(
            cudaMallocPitch(
                &ptr,
                &pitch,
                dim.x * sizeof(float),
                dim.y * dim.z
            )
        );

        CUDA_CHECK(
            cudaMemcpy2D(
                ptr,
                pitch,
                host_data.data(),
                c * sizeof(float),
                c * sizeof(float),
                r * z,
                cudaMemcpyHostToDevice
            )
        );

        data.reset(ptr);
        allocated = true;
    }

    void ImageBlock::allocate(int x, int y, int z) {
        dim = make_float3(x, y, z);

        float* ptr;
        CUDA_CHECK(
            cudaMallocPitch(
                &ptr,
                &pitch,
                dim.x * sizeof(float),
                dim.y * dim.z
            )
        );

        data.reset(ptr);
        allocated = true;
    }
}
