/**
 * @file    ImageRecognitionOrtSessionHandlerBase.hpp
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

#include "OrtSessionHandler.hpp"

namespace Ort
{

  /*! \class ImageRecognitionOrtSessionHandlerBase
      \brief An ImageRecognitionOrtSessionHandlerBase class object.
      This class inherits from the base class, OrtSessionHandler and serves as the base class for
      MaskRCNN, TinyYolov2, Yolov3 and UltraLightFastGenericFaceDetector.

  */
class ImageRecognitionOrtSessionHandlerBase : public OrtSessionHandler
{
 public:
   /*! \brief This calls OrtSessionHandler’s constructor.
   It also:
   1. Initializes m_numClasses with numClasses.
   2. Initializes empty m_classNames.
   3. Populates m_classNames with numeric text strings corresponding to
   numClasses.*/
    ImageRecognitionOrtSessionHandlerBase(
        const uint16_t numClasses,                           //
        const std::string& modelPath,                        //
        const std::optional<size_t>& gpuIdx = std::nullopt,  //
        const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);

    /*! \brief Calls standard destructor. Is empty destructor.*/
    ~ImageRecognitionOrtSessionHandlerBase();

    /*! \brief
    A mutator function.
    1. Runs check on if input array of class names is equal to previously assigned
    number of classes in m_numClasses.
    2. If true, assigns m_classNames with classNames.
    3. Otherwise, report an error.*/
    void initClassNames(const std::vector<std::string>& classNames);

    /*! \brief
    A mutator function.

    Given a certain format of an image like cv::Mat, populate dst to used during
    inferencing.
    The only difference it has as compared to the overriding counterparts is that it considers
    input arguments, meanVal and stdVal when populating dst.*/
    virtual void preprocess(float* dst,                              //
                            const unsigned char* src,                //
                            const int64_t targetImgWidth,            //
                            const int64_t targetImgHeight,           //
                            const int numChanels,                    //
                            const std::vector<float>& meanVal = {},  //
                            const std::vector<float>& stdVal = {}) const;

    /*! \brief
    A getter function. Returns the number of classes defined before.
    */
    uint16_t numClasses() const
    {
        return m_numClasses;
    }

    /*! \brief
    A getter function. Returns the string of class names.
    */
    const std::vector<std::string>& classNames() const
    {
        return m_classNames;
    }

 protected:
    /*! \brief The number of classes defined beforehand.*/
    const uint16_t m_numClasses;
    /*! \brief The the string of class names defined beforehand.*/
    std::vector<std::string> m_classNames;
};
}  // namespace Ort
