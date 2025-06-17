#ifndef MNISTLOADER_H
#define MNISTLOADER_H

#include <vector>
#include <string>

enum class DatasetType {
    MNIST,
    FASHION_MNIST
};

struct MNISTImage {
    std::vector<float> pixels;  // Normalized to [0,1]
    int label;
    DatasetType type;
};

class MNISTLoader {
public:
    static std::vector<MNISTImage> loadTrainingData(const std::string& imagesPath,
                                                    const std::string& labelsPath,
                                                    DatasetType type = DatasetType::MNIST,
                                                    int maxSamples = -1);

    static std::vector<MNISTImage> loadTestData(const std::string& imagesPath,
                                                const std::string& labelsPath,
                                                DatasetType type = DatasetType::MNIST,
                                                int maxSamples = -1);

    // Get label names for different datasets
    static std::string getLabelName(int label, DatasetType type);
    static std::vector<std::string> getAllLabelNames(DatasetType type);

private:
    static int reverseInt(int i);
    static std::vector<std::vector<unsigned char>> readImages(const std::string& path, int maxSamples);
    static std::vector<int> readLabels(const std::string& path, int maxSamples);
};

#endif
