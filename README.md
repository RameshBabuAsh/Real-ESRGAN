# Real-ESRGAN: Enhanced Image Generation from Low-Light Images

## Overview
Real-ESRGAN is a PyTorch-based implementation of an advanced image enhancement model tailored to generate high-quality images from low-light inputs. The model integrates machine learning regressors with a lightweight Real-ESRGAN architecture to achieve superior results. Initial preprocessing steps, such as brightness and contrast adjustments, are performed by machine learning regressors before passing the image to the Real-ESRGAN for final enhancement. This hybrid approach demonstrates exceptional performance on the LOL dataset, achieving a remarkable Peak Signal-to-Noise Ratio (PSNR) of 25.9.

---

## Features
- **Hybrid Architecture:** Combines traditional machine learning regressors with Real-ESRGAN for effective low-light image enhancement.
- **Preprocessing Pipeline:** Applies basic image adjustments, such as brightness and contrast, through machine learning techniques.
- **Lightweight Design:** Optimized for performance while maintaining high-quality outputs.
- **State-of-the-Art Performance:** Outperforms existing models on the LOL dataset with a PSNR of 25.9.

---

## Architecture
The model consists of two primary components:

### 1. **Machine Learning Regressor**
   - Enhances the input image by adjusting parameters like brightness, contrast, and saturation.
   - Acts as a preprocessing step to provide a better starting point for the deep learning model.

### 2. **Real-ESRGAN**
   - A lightweight super-resolution GAN optimized for low-light image enhancement.
   - Generates high-quality enhanced images by learning from the preprocessed outputs of the regressor.

---

## Dataset
The model is trained and evaluated on the LOL (Low-Light) dataset, which includes:
- Low-light input images.
- Corresponding high-quality reference images.

---

## Performance
- **PSNR:** 25.9 (on the LOL dataset).
- **SSIM:** 0.81 (on the LOL dataset).
- **Qualitative Results:** Produces visually appealing enhanced images with well-preserved details and reduced noise.