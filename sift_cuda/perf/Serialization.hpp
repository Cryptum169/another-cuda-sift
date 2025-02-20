#ifndef SERIALIZATION_HH
#define SERIALIZATION_HH

#include <string>
#include <zlib.h>
#include <msgpack.hpp>
#include <fstream>

template <typename SerialType>
bool saveCompressed(
    const SerialType& data, 
    const std::string& filename
) {
    try {
        msgpack::sbuffer buffer;
        msgpack::pack(buffer, data);

        uLong compressedSize = compressBound(buffer.size());
        std::vector<Bytef> compressed(compressedSize);

        int result = compress2(
            compressed.data(),
            &compressedSize,
            reinterpret_cast<const Bytef*>(buffer.data()),
            buffer.size(),
            Z_BEST_COMPRESSION
        );

        if (result != Z_OK) {
            return false;
        }

        std::ofstream file(filename, std::ios::binary);
        if (!file) return false;
        
        uint64_t size = compressedSize;
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(compressed.data()), compressedSize);

        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

template <typename SerialType>
bool loadCompressed(
    SerialType& data,
    const std::string& filename
) {
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;

        uint64_t compressedSize;
        file.read(reinterpret_cast<char*>(&compressedSize), sizeof(compressedSize));

        std::vector<Bytef> compressed(compressedSize);
        file.read(reinterpret_cast<char*>(compressed.data()), compressedSize);

        uLong uncompressedSize = compressedSize * 2;
        std::vector<Bytef> uncompressed;
        int result;

        do {
            uncompressedSize *= 2;
            uncompressed.resize(uncompressedSize);
            
            result = uncompress(
                uncompressed.data(),
                &uncompressedSize,
                compressed.data(),
                compressedSize
            );
        } while (result == Z_BUF_ERROR);
        
        if (result != Z_OK) {
            return false;
        }

        msgpack::object_handle oh = msgpack::unpack(
            reinterpret_cast<char*>(uncompressed.data()), 
            uncompressedSize
        );
        
        msgpack::object obj = oh.get();
        obj.convert(data);
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

#endif

