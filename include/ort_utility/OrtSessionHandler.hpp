/**
 * @file    OrtSessionHandler.hpp
 *
 * @author  btran
 *
 * @date    2020-04-19
 *
 * Copyright (c) organization
 *
 */

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Ort
{

/*! \class OrtSessionHandler
    \brief An OrtSessionHandler class object.
    This class serves as the base parent class which is inherited by class object
    , ImageRecognitionOrtSessionHandlerBase. This class is implemented, following
    Pointer to Implementation (pimpl) C++ programming methodology.

*/
class OrtSessionHandler
{
 public:
    // DataOutputType->(pointer to output data, shape of output data)
    /*! \brief An alias that represents a data structure which pairs a float
    pointer to a std::vector of 64 bit long integer.*/
    using DataOutputType = std::pair<float*, std::vector<int64_t>>;

    /*! \brief A Constructor function*/
    OrtSessionHandler(const std::string& modelPath,  //
                      const std::optional<size_t>& gpuIdx = std::nullopt,
                      const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);
    /*! \brief A Deconstructor function*/
    ~OrtSessionHandler();

    /*! \brief A custom operator which serves as the main function call that
    processes an input image and outputs the resulting tensor.*/
    std::vector<DataOutputType> operator()(const std::vector<float*>& inputImgData);

 private:
    /*! \brief The internal OrtSessionHandlerIml that contains the main
    OrtSessionHandlerIml implementation..*/
    class OrtSessionHandlerIml;
    /*! \brief An opaque pointer to OrtSessionHandlerIml which is called within
    OrtSessionHandler constructor function.*/
    std::unique_ptr<OrtSessionHandlerIml> m_piml;
};
}  // namespace Ort
