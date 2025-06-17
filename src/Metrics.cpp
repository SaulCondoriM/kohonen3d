#include "Metrics.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>

MetricsReport Metrics::evaluateClassification(
    const std::vector<ClassificationResult> &results,
    DatasetType datasetType,
    int numClasses)
{

    MetricsReport report;
    report.datasetType = datasetType;

    // Calculate accuracy
    report.accuracy = calculateAccuracy(results);

    // Calculate confusion matrix
    report.confusionMatrix = calculateConfusionMatrix(results, numClasses);

    // Calculate precision, recall, and F1 score
    calculatePrecisionRecallF1(report.confusionMatrix,
                               report.precisionPerClass,
                               report.recallPerClass,
                               report.f1ScorePerClass);

    // Calculate averages
    report.averagePrecision = 0.0f;
    report.averageRecall = 0.0f;
    report.averageF1 = 0.0f;

    for (int i = 0; i < numClasses; ++i)
    {
        report.averagePrecision += report.precisionPerClass[i];
        report.averageRecall += report.recallPerClass[i];
        report.averageF1 += report.f1ScorePerClass[i];
    }

    report.averagePrecision /= numClasses;
    report.averageRecall /= numClasses;
    report.averageF1 /= numClasses;

    return report;
}

float Metrics::calculateAccuracy(const std::vector<ClassificationResult> &results)
{
    if (results.empty())
        return 0.0f;

    int correct = 0;
    for (const auto &result : results)
    {
        if (result.predictedLabel == result.trueLabel)
        {
            correct++;
        }
    }

    return static_cast<float>(correct) / results.size();
}

std::vector<std::vector<int>> Metrics::calculateConfusionMatrix(
    const std::vector<ClassificationResult> &results,
    int numClasses)
{

    std::vector<std::vector<int>> matrix(numClasses, std::vector<int>(numClasses, 0));

    for (const auto &result : results)
    {
        if (result.trueLabel >= 0 && result.trueLabel < numClasses &&
            result.predictedLabel >= 0 && result.predictedLabel < numClasses)
        {
            matrix[result.trueLabel][result.predictedLabel]++;
        }
    }

    return matrix;
}

void Metrics::calculatePrecisionRecallF1(
    const std::vector<std::vector<int>> &confusionMatrix,
    std::vector<float> &precision,
    std::vector<float> &recall,
    std::vector<float> &f1)
{

    int numClasses = confusionMatrix.size();
    precision.resize(numClasses);
    recall.resize(numClasses);
    f1.resize(numClasses);

    for (int i = 0; i < numClasses; ++i)
    {
        // Calculate precision for class i
        int truePositives = confusionMatrix[i][i];
        int predictedPositives = 0;
        for (int j = 0; j < numClasses; ++j)
        {
            predictedPositives += confusionMatrix[j][i];
        }

        precision[i] = (predictedPositives > 0) ? static_cast<float>(truePositives) / predictedPositives : 0.0f;

        // Calculate recall for class i
        int actualPositives = 0;
        for (int j = 0; j < numClasses; ++j)
        {
            actualPositives += confusionMatrix[i][j];
        }

        recall[i] = (actualPositives > 0) ? static_cast<float>(truePositives) / actualPositives : 0.0f;

        // Calculate F1 score
        f1[i] = (precision[i] + recall[i] > 0) ? 2 * precision[i] * recall[i] / (precision[i] + recall[i]) : 0.0f;
    }
}

void Metrics::printConfusionMatrix(
    const std::vector<std::vector<int>> &matrix,
    const std::vector<std::string> &classNames)
{

    int numClasses = matrix.size();

    std::cout << "\n=== CONFUSION MATRIX ===" << std::endl;
    std::cout << std::setw(12) << "True\\Pred";

    for (int i = 0; i < numClasses; ++i)
    {
        std::cout << std::setw(8) << classNames[i].substr(0, 7);
    }
    std::cout << std::endl;

    for (int i = 0; i < numClasses; ++i)
    {
        std::cout << std::setw(12) << classNames[i].substr(0, 11);
        for (int j = 0; j < numClasses; ++j)
        {
            std::cout << std::setw(8) << matrix[i][j];
        }
        std::cout << std::endl;
    }
}

void MetricsReport::print(const std::vector<std::string> &classNames) const
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "       CLASSIFICATION METRICS REPORT" << std::endl;
    std::cout << "========================================" << std::endl;

    std::string datasetName = (datasetType == DatasetType::MNIST) ? "MNIST" : "Fashion-MNIST";
    std::cout << "Dataset: " << datasetName << std::endl;
    std::cout << "Overall Accuracy: " << std::fixed << std::setprecision(4)
              << accuracy * 100 << "%" << std::endl;

    // Print confusion matrix
    Metrics::printConfusionMatrix(confusionMatrix, classNames);

    // Print per-class metrics
    std::cout << "\n=== PER-CLASS METRICS ===" << std::endl;
    std::cout << std::setw(15) << "Class" << std::setw(12) << "Precision"
              << std::setw(12) << "Recall" << std::setw(12) << "F1-Score" << std::endl;
    std::cout << std::string(51, '-') << std::endl;

    for (size_t i = 0; i < classNames.size() && i < precisionPerClass.size(); ++i)
    {
        std::cout << std::setw(15) << classNames[i].substr(0, 14)
                  << std::setw(12) << std::fixed << std::setprecision(4) << precisionPerClass[i]
                  << std::setw(12) << std::fixed << std::setprecision(4) << recallPerClass[i]
                  << std::setw(12) << std::fixed << std::setprecision(4) << f1ScorePerClass[i]
                  << std::endl;
    }

    std::cout << std::string(51, '-') << std::endl;
    std::cout << std::setw(15) << "AVERAGE"
              << std::setw(12) << std::fixed << std::setprecision(4) << averagePrecision
              << std::setw(12) << std::fixed << std::setprecision(4) << averageRecall
              << std::setw(12) << std::fixed << std::setprecision(4) << averageF1
              << std::endl;

    std::cout << "========================================\n"
              << std::endl;
}

void Metrics::saveReportToFile(const MetricsReport &report,
                               const std::vector<std::string> &classNames,
                               const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }

    std::string datasetName = (report.datasetType == DatasetType::MNIST) ? "MNIST" : "Fashion-MNIST";
    file << "Classification Report - " << datasetName << std::endl;
    file << "Overall Accuracy: " << report.accuracy * 100 << "%" << std::endl;
    file << std::endl;

    // Save confusion matrix
    file << "Confusion Matrix:" << std::endl;
    for (size_t i = 0; i < report.confusionMatrix.size(); ++i)
    {
        for (size_t j = 0; j < report.confusionMatrix[i].size(); ++j)
        {
            file << report.confusionMatrix[i][j] << "\t";
        }
        file << std::endl;
    }

    file << std::endl
         << "Per-class metrics:" << std::endl;
    file << "Class\tPrecision\tRecall\tF1-Score" << std::endl;
    for (size_t i = 0; i < classNames.size() && i < report.precisionPerClass.size(); ++i)
    {
        file << classNames[i] << "\t"
             << report.precisionPerClass[i] << "\t"
             << report.recallPerClass[i] << "\t"
             << report.f1ScorePerClass[i] << std::endl;
    }

    file.close();
    std::cout << "Report saved to: " << filename << std::endl;
}
