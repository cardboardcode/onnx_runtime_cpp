/**
 * @file    ImageClassificationOrtSessionHandler.hpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "ImageRecognitionOrtSessionHandlerBase.hpp"

namespace Ort
{
  /*! \class ImageClassificationOrtSessionHandler
      \brief An ImageClassificationOrtSessionHandler class object.
      This class inherits ImageRecognitionOrtSessionHandlerBase and
      is only used in TestImageClassication.cpp where squeezenet1.1.onnx is utilized
      according to the preface instructions on README.md.
  */
class ImageClassificationOrtSessionHandler : public ImageRecognitionOrtSessionHandlerBase
{
  /*! \brief This calls ImageClassificationOrtSessionHandler's constructor.
  It also:
  1. Initializes m_numClasses with numClasses.
  2. Initializes an internal OrtSessionHandler with modelPath, gpuIdx and inputShapes.*/
 public:
    ImageClassificationOrtSessionHandler(
        const uint16_t numClasses,                           //
        const std::string& modelPath,                        //
        const std::optional<size_t>& gpuIdx = std::nullopt,  //
        const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);

    /*! \brief Calls standard destructor. Is empty destructor.*/
    ~ImageClassificationOrtSessionHandler();

    /*! \brief A mutator function
    1. Utilizes external softmax function to parse resulting inference from model.
    2. Map corresponding object class name index to resulting confidence score.
    3. Output the top k pairs of object class name index and corresponding confidence score. K is arbitruarily set by user in TestImageClassication.cpp.
    */
    std::vector<std::pair<int, float>> topK(const std::vector<float*>& inferenceOutput,  //
                                            const uint16_t k = 1,                        //
                                            const bool useSoftmax = true) const;

    /*! \brief A mutator function
    1. Utilizes external softmax function to parse resulting inference from model.
    2. Map corresponding object class name string to resulting confidence score.
    3. Print to terminal the top k pairs of object class name string and corresponding confidence score. K is arbitruarily set by user in TestImageClassication.cpp.
    */
    std::string topKToString(const std::vector<float*>& inferenceOutput,  //
                             const uint16_t k = 1,                        //
                             const bool useSoftmax = true) const;
};
}  // namespace Ort
