
# NEP_GT

This repository records post-processing operations for NEP (Neural Equivariant Potential) functions. Currently, the following functionalities are included:

- **NEP_Active_learning**: Module for active learning of NEP potentials.
- **NEP_Order_parameter**: Calculation of order parameters for different structures in the system.
- **NEP_Polarization**: Analysis of polarization behavior in the system.
- **NEP_Elastic_properties**: Calculation of elastic properties of materials, including Young's modulus, shear modulus, etc.
- **NEP_Phon**: Handling properties related to phonons.
- **NEP_Prediction**: Prediction of material properties using NEP models.
- **NEP_Loss**: Analysis of the loss function behavior during NEP model training.
- **NEP_Polar_snapshot**: Capturing transient changes in system polarization.

## Directory Structure

```
jupyter notebook/
├── NEP_Active_learning/
├── NEP_Order_parameter/
├── NEP_Polarization/
├── NEP_Elastic_properties/
├── NEP_Phon/
├── NEP_Prediction/
├── NEP_Loss/
└── NEP_Polar_snapshot/
```

All the functionalities are contained within the 'jupyter notebook' directory, which is the main folder for running NEP-related tasks.

## Installation Instructions

1. Clone this repository to your local machine:

   ```bash
   git clone git@github.com:wangchr1617/NEP_GT.git
   ```

2. Navigate to the repository directory and install the required dependencies based on the project needs.

## Usage Guide

Each submodule corresponds to different post-processing operations. Please navigate to the corresponding folder within the 'jupyter notebook' directory for usage. Below is a brief description of each submodule:

- **NEP_Active_learning**: Contains scripts and tools for active learning, helping users iteratively improve NEP potential models.
- **NEP_Order_parameter**: Calculates order parameters of the system to analyze transitions between different structural phases.
- **NEP_Polarization**: Computes polarization vectors and related properties of the system.
- **NEP_Elastic_properties**: Provides scripts and workflows for calculating elastic properties of materials.
- **NEP_Phon**: Computation and analysis of phonon-related properties.
- **NEP_Prediction**: Uses NEP models to predict energy, forces, stress, and other system properties.
- **NEP_Loss**: Analyzes the loss function behavior during training to evaluate model convergence.
- **NEP_Polar_snapshot**: Captures and analyzes snapshots of system polarization behavior.

## Contribution Guidelines

We welcome issues and pull requests to help improve and extend this repository. When contributing code, please follow these guidelines:

1. Develop on the `dev` branch and ensure that all tests pass before submission.
2. Before submitting a PR, ensure that the code is properly formatted and that the relevant documentation is updated.
3. Provide a brief description of the changes and their purpose when submitting a PR.

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](./LICENSE) file.
