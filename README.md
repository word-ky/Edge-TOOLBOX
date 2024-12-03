## Project Overview

**Efficient Detection Framework Adaptation for Edge Computing: A Plug-and-play Neural Network Toolbox Enabling Edge Deployment**

Existing edge detection methods suffer from several issues, including:
1. **Imbalance between performance and scalability**
2. **Lack of dedicated design for edge deployment**
3. **No validation in real-world scenarios**

We present a ** Edge Detection Toolbox (ED-TOOLBOX)**, which includes plug-and-play components tailored specifically for detection models. This toolbox enables edge optimization without altering the original model architecture and, most importantly, maintains or even improves model performance.

### Key Contributions

- **Rep-DConvNet Module**: A feature extraction network lightweighted based on a reparameterization strategy, significantly reducing parameters while ensuring perceptual performance.
- **Sparse Cross-Attention (SC-A) Module**: Efficient adaptive connections between different model modules through high-performance self-attention calculation, with no parameters.
- **Joint Module**: Optimized detection module named following the YOLO (You Only Look Once) family naming convention.
- **Efficient Head**: For the most popular YOLO detection methods, an **Efficient Head** is proposed to achieve full-edge optimization.

### Helmet Strap Detection Dataset (HBDD)

We found that existing helmet detection tasks focus only on whether a helmet is worn, but overlook the wearing of helmet straps. This not only violates safety regulations but also poses safety risks. To address this, we contribute the **HBDD** dataset and plan to apply **GL-TOOLBOX** for edge detection in this real industrial scene.

### Code and Dataset

- **Code**: The code is currently under review for 'software publication' for potential future commercial collaboration. Therefore, it is subject to a 'proprietary licensing agreement (PLA)', which means we are unable to provide the source code directly. However, we are offering a similar version that closely mirrors the one described in the paper.
- **Dataset**: The HBDD Dataset was developed in collaboration with Victorysoft. As a result, we are unable to release the full dataset. However, we will open-source a portion of the unlabeled test set. To gain access to the complete dataset, please contact us at: wjq11346@student.ubc.ca, with the assurance that the data will not be used for commercial purposes. We reserve the right to take legal action in the event of any infringement.

### TODO

- We will open-source our application project on **Hugging Face** in the SPACE Section. Stay tuned.


If you are interested in this project, or have any inquiries or requests, please feel free to contact us directly: wjq11346@student.ubc.ca
