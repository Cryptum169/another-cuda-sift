#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <iostream>
#include <cstdint>

struct PixelCoord {
    float px;
    float py;

    PixelCoord() {}
    PixelCoord(int x, int y) : px(x), py(y) {}

    PixelCoord(float x, float y) : px(x), py(y) {}
};

struct Size {
    float row;
    float col;

    Size(){};
    Size(float r, float c) : row(r), col(c) {};
    Size(int r, int c): row(r), col(c) {};
};

typedef std::vector<uint8_t> Vector8;
typedef std::vector<float> Vectorf;
typedef std::vector<double> Vectord;


template <typename Data_T>
class Image {

    public:
        Image() {};

        Data_T& at(int row, int col) {

            if (!m_data) {
                throw std::runtime_error("at: Matrix accessed before initialization");
            }

            checkGuards(col, row);

            return m_data->at(row * m_image_size.col + col);
        };

        const Data_T& at(int row, int col) const {

            if (!m_data) {
                throw std::runtime_error("at: Matrix accessed before initialization");
            }

            checkGuards(col, row);

            return m_data->at(row * m_image_size.col + col);
        };

        Data_T& atCoord(const PixelCoord& coord) {
            return atXY(coord.px, coord.py);
        }

        Data_T& atXY(int x, int y) {
            checkGuards(x, y);
            return m_data->at(y * m_image_size.col + x);
        }

        const Data_T& atXY(int x, int y) const {
            checkGuards(x, y);
            return m_data->at(y * m_image_size.col + x);
        }

        void checkGuards(const int x, const int y) const {
            if (!m_data) {
                throw std::runtime_error("atXY: Matrix accessed before initialization");
            }

            if (x >= m_image_size.col || x < 0) {
                throw std::runtime_error("atXY x out of range");
            }

            if (y >= m_image_size.row || y < 0) {
                throw std::runtime_error("atXY y out of range");
            }
        }
        
        template<typename OtherData_T>
        bool operator==(const Image<OtherData_T> & other) const {
            if constexpr (!std::is_same_v<Data_T, OtherData_T>) {
                std::cout << "Data type difference" << std::endl;
                return false;
            }

            bool dim = m_image_size.col == other.m_image_size.col && m_image_size.row == other.m_image_size.row;
            if (!dim) {
                std::cout << "dim difference" << std::endl;
                return false;
            }

            for (size_t idx = 0; idx < m_data->size(); idx++) {
                if (m_data->at(idx) != other.m_data->at(idx)) {
                    std::cout << "Data difference" << std::endl;
                    return false;
                }
            }

            return true;
        }

        template <typename DataType_T = Data_T>
        std::string toString() const {
            std::string str;
            for (int py = 0; py < m_image_size.row; py++) {
                for (int px = 0; px < m_image_size.col; px++) {
                    str += std::to_string(DataType_T(atXY(px, py))) + " ";
                }
                str += "\n";
            }

            return str;
        }

        bool equals(const Image<Data_T> & other, Data_T tolerance = 1, bool test_output = false) const {
            bool dim = m_image_size.col == other.m_image_size.col && m_image_size.row == other.m_image_size.row;
            if (!dim) {
                return false;
            }

            for (size_t idx = 0; idx < m_data->size(); idx++) {
                if (abs(m_data->at(idx) - other.m_data->at(idx)) > tolerance) {
                    if (test_output) {
                        std::cout << "Diff idx: " << idx << ", " << m_data->at(idx) << " vs " << other.m_data->at(idx) << std::endl;
                    }
                    return false;
                }
            }

            return true;
        }

        template <typename NewDataType_T>
        Image<NewDataType_T> minus(const Image<Data_T>& other) const {
            bool dim = m_image_size.col == other.m_image_size.col && m_image_size.row == other.m_image_size.row;
            if (!dim) {
                throw std::runtime_error("Image minus dimensions not the same");
            }

            Image<NewDataType_T> new_data;
            new_data.m_image_size = m_image_size;
            std::vector<NewDataType_T> data; data.reserve(m_data->size());
            for (size_t idx = 0; idx < m_data->size(); idx++) {

                if constexpr (std::is_same_v<NewDataType_T, Data_T>) {
                    Data_T diff = m_data->at(idx) - other.m_data->at(idx);
                    data.push_back(diff);
                } else {
                    NewDataType_T diff = static_cast<NewDataType_T>(m_data->at(idx)) - static_cast<NewDataType_T>(other.m_data->at(idx));
                    data.push_back(diff);
                }
            }

            new_data.m_data = std::make_shared<std::vector<NewDataType_T>>(data);
            return new_data;
        }

        template <typename NewDataType_T>
        Image<NewDataType_T> toType() const {
            Image<NewDataType_T> new_data;
            new_data.m_image_size = m_image_size;
            new_data.m_data = std::make_shared<std::vector<NewDataType_T>>();
            new_data.m_data->reserve(m_data->size());
            for (const auto & v : *m_data) {
                new_data.m_data->push_back(static_cast<NewDataType_T>(v));
            }

            return new_data;
        }

    std::shared_ptr<std::vector<Data_T>> m_data{};
    Size m_image_size{};
};

typedef Image<uint8_t> Image8U;
typedef Image<float> Imagef;
