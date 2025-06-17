#include "NpyLoader.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <sstream>
#include <algorithm>
#include <cstring>

NpyArray NpyLoader::load(const std::string& filename) {
    NpyArray array;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return array;
    }

    // Read magic string
    char magic[6];
    file.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
        std::cerr << "Error: Invalid NPY file format" << std::endl;
        return array;
    }

    // Read version
    uint8_t major_version, minor_version;
    file.read(reinterpret_cast<char*>(&major_version), 1);
    file.read(reinterpret_cast<char*>(&minor_version), 1);

    // Read header length
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    // Read header
    std::string header(header_len, ' ');
    file.read(&header[0], header_len);

    // Parse header
    parseHeader(header, array);

    // Calculate total size
    size_t total_size = array.size();

    // Read data based on dtype
    if (array.dtype.find("float32") != std::string::npos || array.dtype.find("f4") != std::string::npos) {
        std::vector<float> data(total_size);
        file.read(reinterpret_cast<char*>(data.data()), total_size * sizeof(float));
        array.data = data;
    } else if (array.dtype.find("float64") != std::string::npos || array.dtype.find("f8") != std::string::npos) {
        std::vector<double> temp_data(total_size);
        file.read(reinterpret_cast<char*>(temp_data.data()), total_size * sizeof(double));
        array.data.resize(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            array.data[i] = static_cast<float>(temp_data[i]);
        }
    } else if (array.dtype.find("int32") != std::string::npos || array.dtype.find("i4") != std::string::npos) {
        std::vector<int32_t> temp_data(total_size);
        file.read(reinterpret_cast<char*>(temp_data.data()), total_size * sizeof(int32_t));
        array.data.resize(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            array.data[i] = static_cast<float>(temp_data[i]);
        }
    } else if (array.dtype.find("int64") != std::string::npos || array.dtype.find("i8") != std::string::npos) {
        std::vector<int64_t> temp_data(total_size);
        file.read(reinterpret_cast<char*>(temp_data.data()), total_size * sizeof(int64_t));
        array.data.resize(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            array.data[i] = static_cast<float>(temp_data[i]);
        }
    } else if (array.dtype.find("uint8") != std::string::npos || array.dtype.find("u1") != std::string::npos) {
        std::vector<uint8_t> temp_data(total_size);
        file.read(reinterpret_cast<char*>(temp_data.data()), total_size * sizeof(uint8_t));
        array.data.resize(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            array.data[i] = static_cast<float>(temp_data[i]);
        }
    } else {
        std::cerr << "Error: Unsupported dtype: " << array.dtype << std::endl;
        return array;
    }

    file.close();

    std::cout << "Loaded NPY file: " << filename << std::endl;
    std::cout << "Shape: (";
    for (size_t i = 0; i < array.shape.size(); ++i) {
        std::cout << array.shape[i];
        if (i < array.shape.size() - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
    std::cout << "Dtype: " << array.dtype << std::endl;

    return array;
}

void NpyLoader::parseHeader(const std::string& header, NpyArray& array) {
    // Find the shape
    size_t shape_pos = header.find("'shape':");
    if (shape_pos == std::string::npos) {
        std::cerr << "Error: Could not find shape in header" << std::endl;
        return;
    }

    size_t shape_start = header.find('(', shape_pos);
    size_t shape_end = header.find(')', shape_start);
    std::string shape_str = header.substr(shape_start + 1, shape_end - shape_start - 1);
    array.shape = parseShape(shape_str);

    // Find the dtype
    size_t dtype_pos = header.find("'descr':");
    if (dtype_pos == std::string::npos) {
        std::cerr << "Error: Could not find dtype in header" << std::endl;
        return;
    }

    size_t dtype_start = header.find('\'', dtype_pos + 8);
    size_t dtype_end = header.find('\'', dtype_start + 1);
    array.dtype = header.substr(dtype_start + 1, dtype_end - dtype_start - 1);

    // Find fortran_order
    size_t fortran_pos = header.find("'fortran_order':");
    if (fortran_pos != std::string::npos) {
        size_t value_start = header.find_first_not_of(" \t", fortran_pos + 16);
        array.fortran_order = (header.substr(value_start, 4) == "True");
    } else {
        array.fortran_order = false;
    }
}

std::string NpyLoader::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

std::vector<size_t> NpyLoader::parseShape(const std::string& shapeStr) {
    std::vector<size_t> shape;
    std::stringstream ss(shapeStr);
    std::string item;

    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty() && item != "") {
            try {
                size_t dim = std::stoull(item);
                shape.push_back(dim);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing shape dimension: " << item << std::endl;
            }
        }
    }

    return shape;
}
