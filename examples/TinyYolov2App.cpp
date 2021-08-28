/**
 * @file    TinyYolov2App.cpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */

#include <chrono>
#include <memory>

#include <opencv2/opencv.hpp>

#include <ort_utility/ort_utility.hpp>

#include "TinyYolov2.hpp"
#include "Utility.hpp"

static constexpr const float CONFIDENCE_THRESHOLD = 0.5;
static constexpr const float NMS_THRESHOLD = 0.6;
static const std::vector<cv::Scalar> COLORS = toCvScalarColors(Ort::VOC_COLOR_CHART);

namespace
{
cv::Mat processOneFrame(Ort::TinyYolov2& osh, const cv::Mat& inputImg, float* dst);
}  // namespace

/*! \brief

The following steps outlines verbosely what the code in this .cpp file does.

1. Checks if the number of commandline arguments is not exactly 3. If true, output verbose error and exit with failure.
2. Store the first commandline argument as the file path to the referenced onnx model and the second as the file path to input image.
3. Read in input image using Opencv.
4. Instantiate an TinyYolov2 class object and initialize it with the total number of a custom FACE_CLASSES for an unknown onnx model with the file path to referenced onnx model.
5. Initialize the classNames in the class object with FACE_CLASSES as defined under Constants.hpp.

6. Initializes a float-type vector variable called dst that takes into account 3 channels for expected input RGB images and a fixed height of 800 pixels and a fixed width of 800 pixels.
7. Calls processOneFrame function which is defined in the same script here and gets the output detection result in the form of an image.

  a. Resizes the the input RGB image proportionally to the fixed 416 pixels by 416 pixels input format for TinyYolov2.

  b. Calls preprocess function to convert the resized input image matrix to 1-dimensional float array.

  c. Run the inference with the 1-dimensional float array.

  d. Extract the anchors and attributes value from the inference output, storing them in numAnchors and numAttrs variables.

  e. Convert the inference output to appropriately segregated vector outputs that capture bounding boxes information, corresponding scores and
  class indices. Filters out any bounding box detection that falls below the defalt 0.5 confidence threshold which is pre-defined in the auxillary function call for processOneFrame.

  f. If the number of bounding boxes in the inference output is zero, just return the original input image.

  g. Perform Non-Maximum Suppression on the segregated vector outputs and filter out bounding boxes with their corresponding confidence score and class indices, based on  0.6 nms threshold value. This value is defined in the auxillary function call for processOneFrame.

  h. Store the filtered results from afterNmsBboxes and afterNmsIndices variables.

  i. Calls the visualizeOneImage function which is defined in examples/Utilty.hpp and returns an output image with all bounding boxes with class label and confidence score printed on image.

8. Write the output detection result into an image file named result.jpg.
*/
int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [apps] [path/to/onnx/yolov3-tiny.onnx] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::string IMAGE_PATH = argv[2];

    Ort::TinyYolov2 osh(Ort::VOC_NUM_CLASSES, ONNX_MODEL_PATH, 0,
                        std::vector<std::vector<int64_t>>{{1, Ort::TinyYolov2::IMG_CHANNEL, Ort::TinyYolov2::IMG_WIDTH,
                                                           Ort::TinyYolov2::IMG_HEIGHT}});

    osh.initClassNames(Ort::VOC_CLASSES);
    std::array<float, Ort::TinyYolov2::IMG_WIDTH * Ort::TinyYolov2::IMG_HEIGHT * Ort::TinyYolov2::IMG_CHANNEL> dst;

    cv::Mat img = cv::imread(IMAGE_PATH);

    if (img.empty()) {
        std::cerr << "Failed to read input image" << std::endl;
        return EXIT_FAILURE;
    }

    auto resultImg = ::processOneFrame(osh, img, dst.data());

    cv::imwrite("result.jpg", resultImg);

    return EXIT_SUCCESS;
}

namespace
{
cv::Mat processOneFrame(Ort::TinyYolov2& osh, const cv::Mat& inputImg, float* dst)
{
    cv::Mat result;
    cv::resize(inputImg, result, cv::Size(Ort::TinyYolov2::IMG_WIDTH, Ort::TinyYolov2::IMG_HEIGHT));

    osh.preprocess(dst, result.data, Ort::TinyYolov2::IMG_WIDTH, Ort::TinyYolov2::IMG_HEIGHT,
                   Ort::TinyYolov2::IMG_CHANNEL);
    auto inferenceOutput = osh({dst});
    assert(inferenceOutput.size() == 1);

    auto processedResult = osh.postProcess(inferenceOutput, CONFIDENCE_THRESHOLD);
    std::vector<std::array<float, 4>> bboxes;
    std::vector<float> scores;
    std::vector<uint64_t> classIndices;
    std::tie(bboxes, scores, classIndices) = processedResult;

    if (bboxes.size() == 0) {
        return result;
    }

    auto afterNmsIndices = Ort::nms(bboxes, scores, NMS_THRESHOLD);

    std::vector<std::array<float, 4>> afterNmsBboxes;
    std::vector<uint64_t> afterNmsClassIndices;

    afterNmsBboxes.reserve(afterNmsIndices.size());
    afterNmsClassIndices.reserve(afterNmsIndices.size());

    for (const auto idx : afterNmsIndices) {
        afterNmsBboxes.emplace_back(bboxes[idx]);
        afterNmsClassIndices.emplace_back(classIndices[idx]);
    }

    result = ::visualizeOneImage(result, afterNmsBboxes, afterNmsClassIndices, COLORS, osh.classNames());

    return result;
}
}  // namespace
