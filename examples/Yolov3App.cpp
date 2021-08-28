/**
 * @file    Yolov3App.cpp
 *
 * @author  btran
 *
 * @date    2020-05-31
 *
 * Copyright (c) organization
 *
 */

#include <ort_utility/ort_utility.hpp>

#include "Utility.hpp"
#include "Yolov3.hpp"

static const std::vector<std::string> BIRD_CLASSES = {"bird_small", "bird_medium", "bird_large"};
static constexpr int64_t BIRD_NUM_CLASSES = 3;
static const std::vector<std::array<int, 3>> BIRD_COLOR_CHART = Ort::generateColorCharts(BIRD_NUM_CLASSES);

static constexpr const float CONFIDENCE_THRESHOLD = 0.2;
static constexpr const float NMS_THRESHOLD = 0.6;
static const std::vector<cv::Scalar> COLORS = toCvScalarColors(BIRD_COLOR_CHART);

namespace
{
/*! \brief
    This is an auxillary function that calls the real processOneFrame function with default values of confThresh and nmsThresh.

    You can adjust the confThresh and nmsThresh value here to tweak the inference results in terms of what detection outputs can filtered out.
*/
cv::Mat processOneFrame(Ort::Yolov3& osh, const cv::Mat& inputImg, float* dst, const float confThresh = 0.15,
                        const float nmsThresh = 0.5);
}  // namespace

/*! \brief

The following steps outlines verbosely what the code in this .cpp file does.

1. Checks if the number of commandline arguments is not exactly 3. If true, output verbose error and exit with failure.
2. Store the first commandline argument as the file path to the referenced onnx model and the second as the file path to input image.
3. Read in input image using Opencv.
4. Instantiate a Yolov3 class object and initialize it with the total number of a custom Bird Classes for an unknown onnx model with the file path to referenced onnx model.
5. Initialize the classNames in the class object with BIRD_CLASSES as defined under Constants.hpp.

6. Initializes a float-type vector variable called dst that takes into account 3 channels for expected input RGB images and a fixed height of 800 pixels and a fixed width of 800 pixels.
7. Calls processOneFrame function which is defined in the same script here and gets the output detection result in the form of an image.

  a. Resizes the the input RGB image proportionally to the fixed 800 pixels by 800 pixels input format for Yolov3.

  b. Calls preprocess function to convert the resized input image matrix to 1-dimensional float array.

  c. Run the inference with the 1-dimensional float array.

  d. Extract the anchors and attributes value from the inference output, storing them in numAnchors and numAttrs variables.

  e. Convert the inference output to appropriately segregated vector outputs that capture bounding boxes information, corresponding scores and
  class indices. Filters out any bounding box detection that falls below the defalt 0.15 confidence threshold which is pre-defined in the auxillary function call for processOneFrame.

  f. If the number of bounding boxes in the inference output is zero, just return the original input image.

  g. Perform Non-Maximum Suppression on the segregated vector outputs and filter out bounding boxes with their corresponding confidence score and class indices, based on  0.5 nms threshold value. This value is defined in the auxillary function call for processOneFrame.

  h. Store the filtered results from afterNmsBboxes and afterNmsIndices variables.

  i. Calls the visualizeOneImage function which is defined in examples/Utilty.hpp and returns an output image with all bounding boxes with class label and confidence score printed on image.

8. Write the output detection result into an image file named result.jpg.
*/
int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [apps] [path/to/onnx/yolov3.onnx] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::string IMAGE_PATH = argv[2];

    cv::Mat img = cv::imread(IMAGE_PATH);

    if (img.empty()) {
        std::cerr << "Failed to read input image" << std::endl;
        return EXIT_FAILURE;
    }

    Ort::Yolov3 osh(
        BIRD_NUM_CLASSES, ONNX_MODEL_PATH, 0,
        std::vector<std::vector<int64_t>>{{1, Ort::Yolov3::IMG_CHANNEL, Ort::Yolov3::IMG_H, Ort::Yolov3::IMG_W}});

    osh.initClassNames(BIRD_CLASSES);

    std::vector<float> dst(Ort::Yolov3::IMG_CHANNEL * Ort::Yolov3::IMG_H * Ort::Yolov3::IMG_W);
    auto result = processOneFrame(osh, img, dst.data());
    cv::imwrite("result.jpg", result);

    return 0;
}


namespace
{

cv::Mat processOneFrame(Ort::Yolov3& osh, const cv::Mat& inputImg, float* dst, const float confThresh,
                        const float nmsThresh)
{
    int origH = inputImg.rows;
    int origW = inputImg.cols;
    float ratioH = origH * 1.0 / osh.IMG_H;
    float ratioW = origW * 1.0 / osh.IMG_W;

    cv::Mat processedImg;
    cv::resize(inputImg, processedImg, cv::Size(osh.IMG_W, osh.IMG_H));

    osh.preprocess(dst, processedImg.data, Ort::Yolov3::IMG_W, Ort::Yolov3::IMG_H, 3);
    auto inferenceOutput = osh({dst});
    int numAnchors = inferenceOutput[0].second[1];
    int numAttrs = inferenceOutput[0].second[2];

    std::vector<std::array<float, 4>> bboxes;
    std::vector<float> scores;
    std::vector<uint64_t> classIndices;

    for (int i = 0; i < numAnchors * numAttrs; i += numAttrs) {
        float conf = inferenceOutput[0].first[i + 4];

        if (conf >= confThresh) {
            float xcenter = inferenceOutput[0].first[i + 0];
            float ycenter = inferenceOutput[0].first[i + 1];
            float width = inferenceOutput[0].first[i + 2];
            float height = inferenceOutput[0].first[i + 3];

            float xmin = xcenter - width / 2;
            float ymin = ycenter - height / 2;
            float xmax = xcenter + width / 2;
            float ymax = ycenter + height / 2;
            xmin = std::max<float>(xmin, 0);
            ymin = std::max<float>(ymin, 0);
            xmax = std::min<float>(xmax, osh.IMG_W - 1);
            ymax = std::min<float>(ymax, osh.IMG_H - 1);

            bboxes.emplace_back(std::array<float, 4>{xmin * ratioW, ymin * ratioH, xmax * ratioW, ymax * ratioH});

            scores.emplace_back(conf);
            int maxIdx = std::max_element(inferenceOutput[0].first + i + 5,
                                          inferenceOutput[0].first + i + 5 + osh.numClasses()) -
                         (inferenceOutput[0].first + i + 5);
            classIndices.emplace_back(maxIdx);
        }
    }

    if (bboxes.size() == 0) {
        return inputImg;
    }

    auto afterNmsIndices = Ort::nms(bboxes, scores, nmsThresh);

    std::vector<std::array<float, 4>> afterNmsBboxes;
    std::vector<uint64_t> afterNmsClassIndices;

    afterNmsBboxes.reserve(afterNmsIndices.size());
    afterNmsClassIndices.reserve(afterNmsIndices.size());

    for (const auto idx : afterNmsIndices) {
        afterNmsBboxes.emplace_back(bboxes[idx]);
        afterNmsClassIndices.emplace_back(classIndices[idx]);
    }

    return visualizeOneImage(inputImg, afterNmsBboxes, afterNmsClassIndices, COLORS, osh.classNames());
}
}  // namespace
