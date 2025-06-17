#ifndef KOHONENNETWORK_H
#define KOHONENNETWORK_H

#include <vector>
#include <random>
#include "MNISTLoader.h"
#include "Metrics.h"

struct Neuron {
    std::vector<float> weights;
    float x, y, z;
    int activationCount;
    int dominantClass;
    std::vector<float> color;
    std::vector<float> prototypeImage;

    Neuron(int inputSize, float posX, float posY, float posZ)
    : weights(inputSize), x(posX), y(posY), z(posZ),
    activationCount(0), dominantClass(-1), color(3, 0.5f),
    prototypeImage(784, 0.0f) {}
};

class KohonenNetwork {
public:
    KohonenNetwork(int width, int height, int depth, int inputSize);

    void initialize();
    void train(const std::vector<MNISTImage>& dataset, int epochs);
    void trainStep(const MNISTImage& input, float learningRate, float neighborhoodRadius);

    int findBestMatchingUnit(const std::vector<float>& input);
    float calculateDistance(const std::vector<float>& input, const Neuron& neuron);
    float calculateSpatialDistance(int neuron1, int neuron2);
    std::vector<int> getNeighbors(int neuronIndex, float radius);

    ClassificationResult classifySample(const MNISTImage& sample);
    MetricsReport evaluateOnDataset(const std::vector<MNISTImage>& testDataset);
    void setDatasetType(DatasetType type) { currentDatasetType = type; }
    DatasetType getDatasetType() const { return currentDatasetType; }

    const std::vector<Neuron>& getNeurons() const { return neurons; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getDepth() const { return depth; }
    int getCurrentEpoch() const { return currentEpoch; }

private:
    int width, height, depth;
    int inputSize;
    std::vector<Neuron> neurons;
    std::mt19937 rng;
    int currentEpoch;
    DatasetType currentDatasetType;

    int get3DIndex(int x, int y, int z) const;
    void getXYZ(int index, int& x, int& y, int& z) const;
    float neighborhoodFunction(float distance, float radius);
};

#endif
