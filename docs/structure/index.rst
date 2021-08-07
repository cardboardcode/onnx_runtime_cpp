.. _api:

Codebase Architecture
=====================

The illustration below shows the inheritance hierachy within the codebase. In other words, from this diagram, you can get a clear sense which modules depend on what.

.. image:: ../img/20210807_onnx_runtime_cpp_inheritance_hierachy.png
   :width: 800
   :alt: Alternative text

Other Information
+++++++++++++++++


Note that **OrtSessionHandlerIml** private class does the real work here.

    a. **Defines** DataOutputType to be std::pair<float*, std::vector<std::int64_t>>
    b. **Defines** a private class called OrtSessionHandlerIml.
    c. **Defines** a toString policy on how to printing the input shapes and data types in DEBUG_LOG function calls.
