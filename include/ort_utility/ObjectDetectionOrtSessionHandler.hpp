/**
 * @file    ObjectDetectionOrtSessionHandler.hpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */

#pragma once

#include <string>
#include <vector>

#include "ImageRecognitionOrtSessionHandlerBase.hpp"

namespace Ort
{
  /*! \class ObjectDetectionOrtSessionHandler
      \brief An ObjectDetectionOrtSessionHandler class object.
      This class inherits ImageRecognitionOrtSessionHandlerBase and is used by
      TestObjectDetection.cpp under examples.
  */
class ObjectDetectionOrtSessionHandler : public ImageRecognitionOrtSessionHandlerBase
{
 public:
   /*! \brief This calls ObjectDetectionOrtSessionHandler's constructor.
   It also:
   1. Initializes m_numClasses with numClasses.
   2. Initializes an internal OrtSessionHandler with modelPath, gpuIdx and inputShapes.*/
    ObjectDetectionOrtSessionHandler(
        const uint16_t numClasses,                           //
        const std::string& modelPath,                        //
        const std::optional<size_t>& gpuIdx = std::nullopt,  //
        const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);

    /*! \brief Calls standard destructor. Is empty destructor.*/
    ~ObjectDetectionOrtSessionHandler();
};
}  // namespace Ort
