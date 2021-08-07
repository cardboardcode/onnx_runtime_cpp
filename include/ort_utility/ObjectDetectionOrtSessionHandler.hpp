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
      This class is not used anywhere in the codebase. So I don't know why this is here.
      Maybe I am dumb. I don't know.

      Will update this if the original author still see a need for this. Okay. Bye.
  */
class ObjectDetectionOrtSessionHandler : public ImageRecognitionOrtSessionHandlerBase
{
 public:
    ObjectDetectionOrtSessionHandler(
        const uint16_t numClasses,                           //
        const std::string& modelPath,                        //
        const std::optional<size_t>& gpuIdx = std::nullopt,  //
        const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);

    ~ObjectDetectionOrtSessionHandler();
};
}  // namespace Ort
