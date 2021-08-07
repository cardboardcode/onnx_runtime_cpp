/**
 * @file    TestImageClassification.cpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <ort_utility/ort_utility.hpp>

static constexpr int64_t IMG_WIDTH = 224;
static constexpr int64_t IMG_HEIGHT = 224;
static constexpr int64_t IMG_CHANNEL = 3;
static constexpr int64_t TEST_TIMES = 1000;

/*! \brief
The following steps outlines verbosely what the code in this .cpp file does.

1. Checks if the number of commandline arguments is not exactly 3. If true, output verbose error and exit with failure.
2. Store the first commandline argument as the file path to the referenced onnx model and the second as the file path to input image.
3. Instantiate an ImageClassificationOrtSessionHandler class object and initialize it with the total number of pretrained ImageNet Classes for squeezenet1.1.onnx with the file path to referenced onnx model.
4. Initialize the classNames in the class object with IMAGENET_CLASSES as defined under Constants.hpp.
5. Read in input image using Opencv.
6. Check if input image is empty. If so, output error and exit with failure.
7. Resize input image down to 244 by 244, as defined in this .cpp file.
8. Convert input image to 1-dimensional float array.
9. Pass 1-dimensional float array to ImageClassificationOrtSessionHandler preprocess function to account for ImageNet images Mean and Standard Deviation. This helps normalize the input image based on how squeezenet1.1.onnx was trained on ImageNet.
10. Start debug timer.
11. Pass normalized 1-dimensional float array to ImageClassificationOrtSessionHandler inference step and store in inferenceOutput variable.
12. Pass inferenceOutput to ImageClassificationOrtSessionHandler mutator function to output the first 5 pairs of object class name strings with their corresponding output confidence score.
13. Stop debug timer. Calculate and output to terminal the time taken to run 1000 rounds of inference. 
*/
int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [apps] [path/to/onnx/model] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::string IMAGE_PATH = argv[2];

    Ort::ImageClassificationOrtSessionHandler osh(Ort::IMAGENET_NUM_CLASSES, ONNX_MODEL_PATH, 0);
    osh.initClassNames(Ort::IMAGENET_CLASSES);

    cv::Mat img = cv::imread(IMAGE_PATH);

    if (img.empty()) {
        std::cerr << "Failed to read input image" << std::endl;
        return EXIT_FAILURE;
    }

    cv::resize(img, img, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    float* dst = new float[IMG_WIDTH * IMG_HEIGHT * IMG_CHANNEL];
    osh.preprocess(dst, img.data, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, Ort::IMAGENET_MEAN, Ort::IMAGENET_STD);

    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_TIMES; ++i) {
        auto inferenceOutput = osh({reinterpret_cast<float*>(dst)});

        const int TOP_K = 5;
        // osh.topK({inferenceOutput[0].first}, TOP_K);
        std::cout << osh.topKToString({inferenceOutput[0].first}, TOP_K) << std::endl;
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << elapsedTime.count() / 1000. << "[sec]" << std::endl;

    delete[] dst;

    return 0;
}
