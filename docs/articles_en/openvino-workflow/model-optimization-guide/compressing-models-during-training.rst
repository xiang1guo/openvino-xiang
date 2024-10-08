Compressing Models During Training
==================================


.. toctree::
   :maxdepth: 1
   :hidden:

   compressing-models-during-training/quantization-aware-training
   compressing-models-during-training/filter-pruning


Introduction
####################

Training-time model compression improves model performance by applying optimizations (such as quantization) during the training. The training process minimizes the loss associated with the lower-precision optimizations, so it is able to maintain the model’s accuracy while reducing its latency and memory footprint. Generally, training-time model optimization results in better model performance and accuracy than :doc:`post-training optimization <quantizing-models-post-training>`, but it can require more effort to set up.

OpenVINO provides the Neural Network Compression Framework (NNCF) tool for implementing compression algorithms on models to improve their performance. NNCF is a Python library that integrates into PyTorch and TensorFlow training pipelines to add training-time compression methods to the pipeline. To apply training-time compression methods with NNCF, you need:

- A floating-point model from the PyTorch or TensorFlow framework.
- A training pipeline set up in the PyTorch or TensorFlow framework.
- Training and validation datasets.

Adding compression to a training pipeline only requires a few lines of code. The compression techniques are defined through a single configuration file that specifies which algorithms to use during fine-tuning.

NNCF Quick Start Examples
+++++++++++++++++++++++++

See the following Jupyter Notebooks for step-by-step examples showing how to add model compression to a PyTorch or Tensorflow training pipeline with NNCF:

- `Quantization Aware Training with NNCF and PyTorch <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pytorch-quantization-aware-training>`__.
- `Quantization Aware Training with NNCF and TensorFlow <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/tensorflow-quantization-aware-training>`__.

More detailed information about NNCF (e.g. installation guide) is present in the dedicated
:doc:`NNCF article <../model-optimization>`.

Working with NNCF
####################

The figure below shows a common workflow of applying training-time compressions with NNCF. The NNCF optimizations are added to the TensorFlow or PyTorch training script, and then the model undergoes fine-tuning. The optimized model can then be exported to OpenVINO IR format for accelerated performance with OpenVINO Runtime.

.. image:: ../../assets/images/nncf_workflow.svg
   :align: center

Training-Time Compression Methods
+++++++++++++++++++++++++++++++++

NNCF provides several methods for improving model performance with training-time compression.

Quantization
--------------------
Quantization is the process of converting the weights and activation values in a neural network from a high-precision format (such as 32-bit floating point) to a lower-precision format (such as 8-bit integer). It helps to reduce the model’s memory footprint and latency. NNCF uses quantization-aware training to quantize models.

Quantization-aware training inserts nodes into the neural network during training that simulate the effect of lower precision. This allows the training algorithm to consider quantization errors as part of the overall training loss that gets minimized during training. The network is then able to achieve enhanced accuracy when quantized.

The officially supported method of quantization in NNCF is uniform 8-bit quantization. This means all the weights and activation functions in the neural network are converted to 8-bit values. See the :doc:`Quantization-aware Training guide <compressing-models-during-training/quantization-aware-training>` to learn more.

Filter pruning
--------------------

Filter pruning algorithms compress models by zeroing out the output filters of convolutional layers based on a certain filter importance criterion. During fine-tuning, an importance criteria is used to search for redundant filters that don’t significantly contribute to the network’s output and zero them out. After fine-tuning, the zeroed-out filters are removed from the network. For more information, see the :doc:`Filter Pruning <compressing-models-during-training/filter-pruning>` page.

Experimental methods
--------------------

NNCF also provides state-of-the-art compression techniques that are still in the experimental stages of development and are only recommended for expert developers. These include:

- Mixed-precision quantization
- Sparsity
- Binarization

To learn more about these methods, visit the `NNCF repository on GitHub <https://github.com/openvinotoolkit/nncf>`__.

Recommended Workflow
++++++++++++++++++++

Using compression-aware training requires a training pipeline, an annotated dataset, and compute resources (such as CPUs or GPUs). If you don't already have these set up and available, it can be easier to start post-training quantization to quickly see quantized results. Then you can use compression-aware training if the model isn't accurate enough. We recommend the following workflow for compressing models with NNCF:

1. :doc:`Perform post-training quantization <quantizing-models-post-training>` on your model and then compare performance to the original model.
2. If the accuracy is too degraded, use :doc:`Quantization-aware Training <compressing-models-during-training/quantization-aware-training>` to increase accuracy while still achieving faster inference time.
3. If the quantized model is still too slow, use :doc:`Filter Pruning <compressing-models-during-training/filter-pruning>` to further improve the model’s inference speed.

Additional Resources
####################

- :doc:`Quantizing Models Post-training <quantizing-models-post-training>`
- :doc:`Model Optimization - NNCF <../model-optimization>`
- `NNCF GitHub repository <https://github.com/openvinotoolkit/nncf>`__
- `NNCF FAQ <https://github.com/openvinotoolkit/nncf/blob/develop/docs/FAQ.md>`__
- `Quantization Aware Training with NNCF and PyTorch <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pytorch-quantization-aware-training>`__.
- `Quantization Aware Training with NNCF and TensorFlow <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/tensorflow-quantization-aware-training>`__.

