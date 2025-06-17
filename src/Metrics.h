#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <string>
#include <map>
#include "MNISTLoader.h"

struct ClassificationResult {
    int predictedLabel;
    int trueLabel;
    float confidence;  // Distance to BMU (lower = more confident)
};

struct MetricsReport {
    float accuracy;
    std::vector<std::vector<int>> confusionMatrix;
    std::vector<float> precisionPerClass;
    std::vector<float> recallPerClass;
    std::vector<float> f1ScorePerClass;
    float averagePrecision;
    float averageRecall;
    float averageF1;
    DatasetType datasetType;

    void print(const std::vector<std::string>& classNames) const;
};

class Metrics {
public:
    static MetricsReport evaluateClassification(
        const std::vector<ClassificationResult>& results,
        DatasetType datasetType,
        int numClasses = 10
    );

    static void printConfusionMatrix(
        const std::vector<std::vector<int>>& matrix,
        const std::vector<std::string>& classNames
    );

    static float calculateAccuracy(const std::vector<ClassificationResult>& results);

    static std::vector<std::vector<int>> calculateConfusionMatrix(
        const std::vector<ClassificationResult>& results,
        int numClasses
    );

    static void calculatePrecisionRecallF1(
        const std::vector<std::vector<int>>& confusionMatrix,
        std::vector<float>& precision,
        std::vector<float>& recall,
        std::vector<float>& f1
    );

    static void saveReportToFile(const MetricsReport& report,
                                 const std::vector<std::string>& classNames,
                                 const std::string& filename);
};

#endif
