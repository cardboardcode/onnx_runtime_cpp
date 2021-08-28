/**
 * @file    MaskRCNNApp.cpp
 *
 * @author  btran
 *
 * @date    2020-05-18
 *
 * Copyright (c) organization
 *
 */
#include <chrono>
#include <iostream>
#include <memory>

#include <opencv2/opencv.hpp>

#include <ort_utility/ort_utility.hpp>

#include "MaskRCNN.hpp"
#include "Utility.hpp"

static constexpr const float CONFIDENCE_THRESHOLD = 0.5;
static const std::vector<cv::Scalar> COLORS = toCvScalarColors(Ort::MSCOCO_COLOR_CHART);

namespace
{
cv::Mat processOneFrame(Ort::MaskRCNN& osh, const cv::Mat& inputImg, int newW, int newH, int paddedW, int paddedH,
                        float ratio, float* dst, const float confThresh = 0.5,
                        const cv::Scalar& meanVal = cv::Scalar(102.9801, 115.9465, 122.7717),
                        bool visualizeMask = true);
}  // namespace

/*! \brief

The following steps outlines verbosely what the code in this .cpp file does.

1. Checks if the number of commandline arguments is not exactly 3. If true, output verbose error and exit with failure.
2. Store the first commandline argument as the file path to the referenced onnx model and the second as the file path to input image.
3. Read in input image using Opencv.
4. Instantiate a MaskRCNN class object and initialize it with the total number of prefined MSCOCO_NUM_CLASSES for an input onnx model with the file path to referenced onnx model.
5. Initialize the classNames in the class object with MSCOCO_CLASSES as defined under Constants.hpp.

6. Initializes a float-type vector variable called dst that takes into account 3 channels for expected input RGB images and a padded image.
7. Calls processOneFrame function which is defined in the same script here and gets the output detection result in the form of an image.

  a. Pads the the input RGB image proportionally to a minimum 800 pixels by 800 pixels input format for MaskRCNN.

  b. Calls preprocess function to convert the resized input image matrix to 1-dimensional float array.

  c. Run the inference with the 1-dimensional float array.

  d. Extract the anchors and attributes value from the inference output, storing them in numAnchors and numAttrs variables.

  e. Convert the inference output to appropriately segregated vector outputs that capture bounding boxes information, corresponding scores and
  class indices. Filters out any bounding box detection that falls below the defalt 0.15 confidence threshold which is pre-defined in the auxillary function call for processOneFrame.

  f. If the number of bounding boxes in the inference output is zero, just return the original input image.

  i. Calls the visualizeOneImageWithMask function which is defined in examples/Utilty.hpp and returns an output image with all bounding boxes with segmentation masks, class labels and confidence scores printed on image. This function call is done by default by the auxillary processOneFrame function with the boolean visualizeMask variable.

8. Write the output detection result into an image file named result.jpg.
*/
int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [apps] [path/to/onnx/maskrcnn.onnx] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::string IMAGE_PATH = argv[2];

    cv::Mat img = cv::imread(IMAGE_PATH);

    if (img.empty()) {
        std::cerr << "Failed to read input image" << std::endl;
        return EXIT_FAILURE;
    }

    float ratio = 800.0 / std::min(img.cols, img.rows);
    int newW = ratio * img.cols;
    int newH = ratio * img.rows;
    int paddedH = static_cast<int>(((newH + 31) / 32) * 32);
    int paddedW = static_cast<int>(((newW + 31) / 32) * 32);

    Ort::MaskRCNN osh(Ort::MSCOCO_NUM_CLASSES, ONNX_MODEL_PATH, 0,
                      std::vector<std::vector<int64_t>>{{Ort::MaskRCNN::IMG_CHANNEL, paddedH, paddedW}});

    osh.initClassNames(Ort::MSCOCO_CLASSES);

    std::vector<float> dst(Ort::MaskRCNN::IMG_CHANNEL * paddedH * paddedW);

    auto resultImg = ::processOneFrame(osh, img, newW, newH, paddedW, paddedH, ratio, dst.data(), CONFIDENCE_THRESHOLD);
    cv::imwrite("result.jpg", resultImg);

    return EXIT_SUCCESS;
}

namespace
{
cv::Mat processOneFrame(Ort::MaskRCNN& osh, const cv::Mat& inputImg, int newW, int newH, int paddedW, int paddedH,
                        float ratio, float* dst, float confThresh, const cv::Scalar& meanVal, bool visualizeMask)
{
    cv::Mat tmpImg;
    cv::resize(inputImg, tmpImg, cv::Size(newW, newH));

    tmpImg.convertTo(tmpImg, CV_32FC3);
    tmpImg -= meanVal;

    cv::Mat paddedImg(paddedH, paddedW, CV_32FC3, cv::Scalar(0, 0, 0));
    tmpImg.copyTo(paddedImg(cv::Rect(0, 0, newW, newH)));

    osh.preprocess(dst, paddedImg.ptr<float>(), paddedW, paddedH, 3);
    // or
    // osh.preprocess(dst, paddedImg, paddedW, paddedH, 3);

    // boxes, labels, scores, masks
    auto inferenceOutput = osh({dst});

    assert(inferenceOutput[1].second.size() == 1);
    size_t nBoxes = inferenceOutput[1].second[0];

    std::vector<std::array<float, 4>> bboxes;
    std::vector<uint64_t> classIndices;
    std::vector<cv::Mat> masks;

    bboxes.reserve(nBoxes);
    classIndices.reserve(nBoxes);
    masks.reserve(nBoxes);

    for (size_t i = 0; i < nBoxes; ++i) {
        if (inferenceOutput[2].first[i] > confThresh) {
            DEBUG_LOG("%f", inferenceOutput[2].first[i]);

            float xmin = inferenceOutput[0].first[i * 4 + 0] / ratio;
            float ymin = inferenceOutput[0].first[i * 4 + 1] / ratio;
            float xmax = inferenceOutput[0].first[i * 4 + 2] / ratio;
            float ymax = inferenceOutput[0].first[i * 4 + 3] / ratio;

            xmin = std::max<float>(xmin, 0);
            ymin = std::max<float>(ymin, 0);
            xmax = std::min<float>(xmax, inputImg.cols);
            ymax = std::min<float>(ymax, inputImg.rows);

            bboxes.emplace_back(std::array<float, 4>{xmin, ymin, xmax, ymax});
            classIndices.emplace_back(reinterpret_cast<int64_t*>(inferenceOutput[1].first)[i]);

            cv::Mat curMask(28, 28, CV_32FC1);
            memcpy(curMask.data, inferenceOutput[3].first + i * 28 * 28, 28 * 28 * sizeof(float));
            masks.emplace_back(curMask);
        }
    }

    if (bboxes.size() == 0) {
        return inputImg;
    }

    return visualizeMask ? ::visualizeOneImageWithMask(inputImg, bboxes, classIndices, masks, COLORS, osh.classNames())
                         : ::visualizeOneImage(inputImg, bboxes, classIndices, COLORS, osh.classNames());
}
}  // namespace
