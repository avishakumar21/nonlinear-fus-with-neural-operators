# learning-nonlinear-fus-with-neural-operators

This repository contains the necessary code for training and testing a convolutional deep operator network for predicting focused ultrasound propagation in patient-specific spinal cord anatomy. 

To run the code
0. Split data have data in data/, split into train/, val/, test/. Have a result/ folder
1. Set hyperparameters, input output directories, and device (cpu or gpu) in config.json
2. Run the following in the root directory:
   ```
    python3 main.py <a_nane_for_this_trail_NOSPACE>
   ```

The data and a trained model checkpoint can be downloaded here: https://drive.google.com/drive/folders/1ZHBKw1G_ItsYILUMRQUEZik-PrqKsz2m?usp=drive_link


The support models used for benchmarking are in support_models.py


Abstract:
Focused ultrasound (FUS) therapy is a promising tool for optimally targeted treatment of spinal cord injuries (SCI), offering submillimeter precision to enhance blood flow at injury sites while minimizing impact on surrounding tissues. However, its efficacy is highly sensitive to the placement of the ultrasound source, as the spinal cord's complex geometry and acoustic heterogeneity distort and attenuate the FUS signal. Current approaches rely on computer simulations to solve the governing wave propagation equations and compute patient-specific pressure maps using ultrasound images of the spinal cord anatomy. While accurate, these high-fidelity simulations are computationally intensive, taking up to hours to complete parameter sweeps, which is impractical for real-time surgical decision-making. To address this bottleneck, we propose a convolutional deep operator network (DeepONet) to rapidly predict FUS pressure fields in patient spinal cords. Unlike conventional neural networks, DeepONets are well equipped to approximate the solution operator of the parametric partial differential equations (PDEs) that govern the behavior of FUS waves with varying initial and boundary conditions (i.e., new transducer locations or spinal cord geometries) without requiring extensive simulations. Trained on simulated pressure maps across diverse patient anatomies, this surrogate model achieves real-time predictions with only a 2% error on the test set, significantly accelerating the modeling of nonlinear physical systems in heterogeneous domains. By facilitating rapid parameter sweeps in surgical settings, this work provides a crucial step toward precise and individualized solutions in neurosurgical treatments.



   
