#ifndef NPYLOADER_H
#define NPYLOADER_H

#include <vector>
#include <string>

struct NpyArray {
    std::vector<float> data;
    std::vector<size_t> shape;
    std::string dtype;
    bool fortran_order;

    size_t size() const {
        size_t total = 1;
        for (size_t dim : shape) {
            total *= dim;
        }
        return total;
    }
};

class NpyLoader {
public:
    static NpyArray load(const std::string& filename);

private:
    static void parseHeader(const std::string& header, NpyArray& array);
    static std::string trim(const std::string& str);
    static std::vector<size_t> parseShape(const std::string& shapeStr);
};

#endif
