#include "MNISTLoader.h"
#include <fstream>
#include <iostream>
#include <algorithm>

std::string MNISTLoader::getLabelName(int label, DatasetType type)
{
    if (type == DatasetType::MNIST)
    {
        return std::to_string(label);
    }
    else if (type == DatasetType::FASHION_MNIST)
    {
        const std::vector<std::string> fashionLabels = {
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};
        if (label >= 0 && label < 10)
        {
            return fashionLabels[label];
        }
    }
    return "Unknown";
}

std::vector<std::string> MNISTLoader::getAllLabelNames(DatasetType type)
{
    if (type == DatasetType::MNIST)
    {
        return {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    }
    else if (type == DatasetType::FASHION_MNIST)
    {
        return {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};
    }
    return {};
}

std::vector<MNISTImage> MNISTLoader::loadTrainingData(const std::string &imagesPath,
                                                      const std::string &labelsPath,
                                                      DatasetType type,
                                                      int maxSamples)
{
    auto images = readImages(imagesPath, maxSamples);
    auto labels = readLabels(labelsPath, maxSamples);

    std::vector<MNISTImage> dataset;
    dataset.reserve(images.size());

    for (size_t i = 0; i < images.size() && i < labels.size(); ++i)
    {
        MNISTImage img;
        img.label = labels[i];
        img.type = type;
        img.pixels.reserve(784); // 28x28

        // Normalize pixels to [0,1]
        for (unsigned char pixel : images[i])
        {
            img.pixels.push_back(pixel / 255.0f);
        }

        dataset.push_back(std::move(img));
    }

    return dataset;
}

std::vector<MNISTImage> MNISTLoader::loadTestData(const std::string &imagesPath,
                                                  const std::string &labelsPath,
                                                  DatasetType type,
                                                  int maxSamples)
{
    return loadTrainingData(imagesPath, labelsPath, type, maxSamples);
}

int MNISTLoader::reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<std::vector<unsigned char>> MNISTLoader::readImages(const std::string &path, int maxSamples)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Cannot open file: " << path << std::endl;
        return {};
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;

    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    file.read((char *)&num_images, sizeof(num_images));
    num_images = reverseInt(num_images);

    file.read((char *)&num_rows, sizeof(num_rows));
    num_rows = reverseInt(num_rows);

    file.read((char *)&num_cols, sizeof(num_cols));
    num_cols = reverseInt(num_cols);

    if (maxSamples > 0 && maxSamples < num_images)
    {
        num_images = maxSamples;
    }

    std::vector<std::vector<unsigned char>> images(num_images, std::vector<unsigned char>(num_rows * num_cols));

    for (int i = 0; i < num_images; ++i)
    {
        file.read((char *)images[i].data(), num_rows * num_cols);
    }

    file.close();
    return images;
}

std::vector<int> MNISTLoader::readLabels(const std::string &path, int maxSamples)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Cannot open file: " << path << std::endl;
        return {};
    }

    int magic_number = 0, num_labels = 0;

    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    file.read((char *)&num_labels, sizeof(num_labels));
    num_labels = reverseInt(num_labels);

    if (maxSamples > 0 && maxSamples < num_labels)
    {
        num_labels = maxSamples;
    }

    std::vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; ++i)
    {
        unsigned char temp = 0;
        file.read((char *)&temp, sizeof(temp));
        labels[i] = (int)temp;
    }

    file.close();
    return labels;
}
