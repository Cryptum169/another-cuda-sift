#pragma once

#include "CudaMemRAII.cuh"
#include "HostImage.hh"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <memory>
#include <cstring>
#include <concepts>


namespace sift_cuda {
    template<typename Image_T>
    concept ImageContainerType = requires(Image_T t) {
        { t.getDim() } -> std::convertible_to<float2>;
        { t.getData() } -> std::convertible_to<float*>;
        { t.getPitch() } -> std::convertible_to<size_t>;
    };

    /*
     * Container for Image on Device
     */
    struct CudaImage {
        public:
            CudaImage() = default;
            ~CudaImage() = default;
            CudaImage(CudaImage&& other) noexcept;
            CudaImage& operator=(const CudaImage&) = delete;

            explicit CudaImage(const Imagef&);
            void copyFrom(const CudaImage&);
            CudaImage& operator=(const Imagef&);
            void copyTo(Imagef&) const;

            // Functions
            void allocate(int x, int y);
            void fromHost(const std::vector<float>& data, int x, int y);

            float2 getDim() { return dim; }
            float* getData() { return data.get(); }
            size_t getPitch() { return pitch; }

            float2 getDim() const { return dim; }
            float* getData() const { return data.get(); }
            size_t getPitch() const { return pitch; }

            bool isAllocated() const { return allocated; }

            class RowView {
                private:
                    float* row_ptr;
                    int row_dim;
                public:
                    RowView(float* ptr, int dim) : row_ptr(ptr), row_dim(dim) {}

                    float* getData() const { return nullptr;}
                    int getDim() const { return row_dim; }
            };

            RowView getRowPtrAt(size_t idx) {
                float* ptr = data.get() + idx * pitch / sizeof(float);
                return RowView(ptr, dim.x);
            }

        private:
            // Pitched memory
            float2 dim;
            std::unique_ptr<float[], cuda_raii::CudaMemDeleter> data;
            size_t pitch = 0;
            bool allocated{false};
    };
    
    /*
     * Container for Block of Image on Device
     * Used as GaussianBlock and DoG blocks.
     */
    struct ImageBlock {
        public:
            ImageBlock() = default;
            ~ImageBlock() = default;
            __host__ ImageBlock(ImageBlock&& other) noexcept;

            ImageBlock(const ImageBlock&) = delete;
            ImageBlock& operator=(const ImageBlock&) = delete;

            class ImageView {
                private:
                    float* img_ptr;
                    size_t img_pitch;
                    float2 img_dim;
                public:
                    ImageView(float* ptr, size_t pitch, float3 dim): img_ptr(ptr), img_pitch(pitch) {
                        img_dim = make_float2(dim.x, dim.y);
                    }

                    float2 getDim() const { return img_dim; }
                    float* getData() { return img_ptr; }
                    float* getData() const { return img_ptr; }
                    size_t getPitch() const { return img_pitch; }

                    void copyTo(Imagef& image) const {
                        auto dim = getDim();
                        image.m_image_size = {dim.y, dim.x};
                        int size = image.m_image_size.col * image.m_image_size.row;
                        image.m_data = std::make_shared<std::vector<float>>(size);

                        CUDA_CHECK(
                            cudaMemcpy2D(
                                image.m_data->data(),
                                image.m_image_size.col * sizeof(float),
                                img_ptr,
                                img_pitch,
                                dim.x * sizeof(float),
                                dim.y, 
                                cudaMemcpyDeviceToHost
                            );
                        );
                    }
            };

            void allocate(int x, int y, int num_img);
            float3 getDim() { return dim; }
            float* getData() { return data.get(); }
            size_t getPitch() { return pitch; }

            float3 getDim() const { return dim; }
            float* getData() const { return data.get(); }
            size_t getPitch() const { return pitch; }

            ImageView getImagePtrAt(size_t idx) { 
                float* ptr = data.get() + pitch / sizeof(float) * size_t(dim.y) * idx;
                return ImageView(ptr, pitch, dim);
            }

            void fromHost(const std::vector<float>& host_data, int r, int c, int z);
        private:
            float3 dim;
            std::unique_ptr<float[], cuda_raii::CudaMemDeleter> data;
            size_t pitch = 0;

            bool allocated{false};
    };

    /*
     * This is for having functions to be compiled to use non-reference for views, and reference for non-views.
     *    This is necessary because otherwise we cannot pass a view to the `output` argument of a function, since Views are created as rvalues.
     *
     * Since everything is really just pointer and values, it doesn't really matter if it's passed as value or reference.
     *    However just for conveying the intention that this output will be modified, I thought it would be useful to mark it as non-const reference.
     *
     * All in all, the followings aren't quite working, as compiler can't infer the argument types, and we'd need to explicitly specify types.
     * TODO: Somehow fix this.
     */
    template <typename T>
    struct ConstParamType {
        using type = const T&;
    };

    template <typename T>
    struct MutableParamType {
        using type = T&;
    };

    template <>
    struct ConstParamType<ImageBlock::ImageView> {
        using type = ImageBlock::ImageView;
    };

    template <>
    struct MutableParamType<ImageBlock::ImageView> {
        using type = ImageBlock::ImageView;
    };

    template <typename T>
    using CParamType_t = typename ConstParamType<T>::type;

    template <typename T>
    using MParamType_t = typename MutableParamType<T>::type;
};

