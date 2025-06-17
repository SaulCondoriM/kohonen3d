#include "KohonenNetwork.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <map>
#include <limits>

KohonenNetwork::KohonenNetwork(int w, int h, int d, int inputSize)
: width(w), height(h), depth(d), inputSize(inputSize),
rng(std::random_device{}()), currentEpoch(0),
currentDatasetType(DatasetType::MNIST) {

    neurons.reserve(width * height * depth);

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float posX = (x - width/2.0f) * 2.0f / width;
                float posY = (y - height/2.0f) * 2.0f / height;
                float posZ = (z - depth/2.0f) * 2.0f / depth;

                neurons.emplace_back(inputSize, posX, posY, posZ);
            }
        }
    }
}

void KohonenNetwork::initialize() {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& neuron : neurons) {
        for (auto& weight : neuron.weights) {
            weight = dist(rng);
        }
    }

    std::cout << "Network initialized with " << neurons.size() << " neurons" << std::endl;
}

void KohonenNetwork::train(const std::vector<MNISTImage>& dataset, int epochs) {
    std::cout << "Starting training for " << epochs << " epochs..." << std::endl;

    if (!dataset.empty()) {
        currentDatasetType = dataset[0].type;
        std::string datasetName = (currentDatasetType == DatasetType::MNIST) ? "MNIST" : "Fashion-MNIST";
        std::cout << "Training on " << datasetName << " dataset" << std::endl;
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        currentEpoch = epoch;

        float learningRate = 0.5f * std::exp(-epoch / (epochs / 3.0f));

        float neighborhoodRadius = std::max(width, std::max(height, depth)) / 2.0f *
        std::exp(-epoch / (epochs / 3.0f));

        auto shuffledDataset = dataset;
        std::shuffle(shuffledDataset.begin(), shuffledDataset.end(), rng);

        for (const auto& sample : shuffledDataset) {
            trainStep(sample, learningRate, neighborhoodRadius);
        }

        if (epoch % 10 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << "/" << epochs
            << " - LR: " << learningRate
            << " - Radius: " << neighborhoodRadius << std::endl;
        }
    }

    classifyNeurons(dataset);
    findPrototypeImages(dataset);
    updateColors();

    std::cout << "Training completed!" << std::endl;
}

void KohonenNetwork::trainStep(const MNISTImage& input, float learningRate, float neighborhoodRadius) {
    int bmuIndex = findBestMatchingUnit(input.pixels);
    auto neighbors = getNeighbors(bmuIndex, neighborhoodRadius);

    for (int neighborIndex : neighbors) {
        float spatialDistance = calculateSpatialDistance(bmuIndex, neighborIndex);
        float influence = neighborhoodFunction(spatialDistance, neighborhoodRadius);

        for (size_t i = 0; i < neurons[neighborIndex].weights.size(); ++i) {
            neurons[neighborIndex].weights[i] += learningRate * influence *
            (input.pixels[i] - neurons[neighborIndex].weights[i]);
        }
    }

    neurons[bmuIndex].activationCount++;
}

int KohonenNetwork::findBestMatchingUnit(const std::vector<float>& input) {
    int bestIndex = 0;
    float minDistance = calculateDistance(input, neurons[0]);

    for (size_t i = 1; i < neurons.size(); ++i) {
        float distance = calculateDistance(input, neurons[i]);
        if (distance < minDistance) {
            minDistance = distance;
            bestIndex = i;
        }
    }

    return bestIndex;
}

float KohonenNetwork::calculateDistance(const std::vector<float>& input, const Neuron& neuron) {
    float distance = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        float diff = input[i] - neuron.weights[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}

float KohonenNetwork::calculateSpatialDistance(int neuron1, int neuron2) {
    int x1, y1, z1, x2, y2, z2;
    getXYZ(neuron1, x1, y1, z1);
    getXYZ(neuron2, x2, y2, z2);

    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;

    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

std::vector<int> KohonenNetwork::getNeighbors(int neuronIndex, float radius) {
    std::vector<int> neighbors;

    for (size_t i = 0; i < neurons.size(); ++i) {
        if (calculateSpatialDistance(neuronIndex, i) <= radius) {
            neighbors.push_back(i);
        }
    }

    return neighbors;
}

float KohonenNetwork::neighborhoodFunction(float distance, float radius) {
    if (radius <= 0) return distance == 0 ? 1.0f : 0.0f;
    return std::exp(-(distance * distance) / (2 * radius * radius));
}

ClassificationResult KohonenNetwork::classifySample(const MNISTImage& sample) {
    ClassificationResult result;
    result.trueLabel = sample.label;

    int bmuIndex = findBestMatchingUnit(sample.pixels);
    result.predictedLabel = neurons[bmuIndex].dominantClass;
    result.confidence = calculateDistance(sample.pixels, neurons[bmuIndex]);

    return result;
}

MetricsReport KohonenNetwork::evaluateOnDataset(const std::vector<MNISTImage>& testDataset) {
    std::cout << "Evaluating network on test dataset..." << std::endl;

    std::vector<ClassificationResult> results;
    results.reserve(testDataset.size());

    for (const auto& sample : testDataset) {
        results.push_back(classifySample(sample));
    }

    DatasetType evalType = testDataset.empty() ? currentDatasetType : testDataset[0].type;
    MetricsReport report = Metrics::evaluateClassification(results, evalType);

    std::cout << "Evaluation completed!" << std::endl;
    return report;
}


int KohonenNetwork::get3DIndex(int x, int y, int z) const {
    return z * width * height + y * width + x;
}

void KohonenNetwork::getXYZ(int index, int& x, int& y, int& z) const {
    z = index / (width * height);
    int remainder = index % (width * height);
    y = remainder / width;
    x = remainder % width;
}
