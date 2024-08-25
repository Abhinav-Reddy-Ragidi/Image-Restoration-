Image Restoration Using Generative Adversarial Networks (GAN)
Overview
This project implements an image restoration pipeline using Generative Adversarial Networks (GAN). The goal is to restore low-quality or degraded images to their original, high-quality state. The project includes a custom-designed architecture for both the generator and discriminator models, which are the core components of the GAN.

Project Structure
1. Generator Network
Purpose: The generator is responsible for producing restored images from low-quality inputs. It leverages convolutional layers, residual blocks, and upsampling layers to generate high-resolution images that closely resemble real images from the training dataset.
Key Features:
Feature Extraction: Captures important features from the input image using convolutional layers.
Residual Blocks: Enhances the ability of the network to learn complex patterns and fine details in images.
Upsampling: Transposed convolutional layers are used to increase the resolution of the generated images.
Output: Produces an image with the same dimensions as the input but with enhanced quality.
2. Discriminator Network
Purpose: The discriminator acts as a binary classifier that distinguishes between real images (from the dataset) and fake images (generated by the generator). It is trained to improve its accuracy in identifying fake images.
Key Features:
Convolutional Layers: Extract features from input images while progressively reducing their spatial dimensions.
LeakyReLU Activation: Provides a non-zero gradient for negative inputs, aiding in better learning.
Batch Normalization: Stabilizes and accelerates training by normalizing outputs from convolutional layers.
Output: A single probability value indicating whether the input image is real or fake.
3. Dataset
Dataset Used: The DIV2K dataset is used for training and testing the GAN model. The dataset contains high-resolution images that serve as the ground truth for training the generator.
4. Training Process
The training process involves alternating between training the generator and the discriminator. The generator aims to produce realistic images to fool the discriminator, while the discriminator tries to correctly classify images as real or fake.
Loss Functions:
Generator Loss: Mean Squared Error (MSE) is used to minimize the difference between generated and real images.
Discriminator Loss: Binary Crossentropy is used to maximize the discriminator's ability to differentiate between real and fake images.
5. Image Restoration
After training, the generator can be used to restore degraded images to a higher quality. The restored images are compared to their original counterparts to evaluate the performance of the model.
