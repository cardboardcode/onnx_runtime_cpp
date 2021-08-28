.. _api_Utility:

Utilty
======

.. doxygenfile:: Utility.hpp
   :project: onnx_runtime_cpp

This file is not documented as per normal like other files due to duplicate **Utility.hpp** in ``include`` folder as well. This pertains to a documentation bug that is still unresolved in **Doxygen**. Please look at `this GitHub issue <https://github.com/doxygen/doxygen/issues/897>`_ for more details.

The following is an approximation of what the various functions do in **Utility.hpp** under ``examples`` folder.

toCvScalarColors
****************
This function is called under ``MaskRCNNApp.cpp``, ``TinyYolov2App.cpp``, ``UltraLightFastGenericFaceDetectorApp.cpp`` and ``Yolov3App.cpp``.

Takes in a vector of integer array and outputs a vector of cv::Scalar.

visualizeOneImage
*****************
This function is called under ``MaskRCNNApp.cpp``, ``TinyYolov2App.cpp``, ``UltraLightFastGenericFaceDetectorApp.cpp`` and ``Yolov3App.cpp``.

Takes in the segregated vector outputs that contain the detection results, parses them and draws bounding boxes with corresponding class labels and confidence scores on the output RGB image.

visualizeOneImageWithMask
*************************
This function is called under ``MaskRCNNApp.cpp``.

Takes in the segregated vector outputs that contain the detection results, parses them and draws segmentation masks and bounding boxes with corresponding class labels and confidence scores on the output RGB image.
