
================================================================================================================================================
IRAW Project - Code and Environment Guide
================================================================================================================================================

Dear Reviewers,

We would like to express our sincere gratitude for your valuable time and meticulous efforts to review our manuscript (IRAW). Regarding the attached demonstration code, please note:

1. The demo facilitates verification of the IRAW framework's capability to generate high-fidelity adversarial perturbations that effectively deceive state-of-the-art deep learning architectures while preserving image fidelity.
2. Raw images and generated adversarial examples are stored in designated folders (see details in the File Organization section below).
3. The code is natively optimized for CUDA acceleration by default to ensure computational efficiency; CPU-compatible adjustments are provided for universal compatibility across different computing environments.

To verify IRAW's performance and validate its core capabilities:
- Conduct a comparative analysis of images in 'Raw Data/' and 'Results/' to evaluate IRAW's performance metrics in generating high-fidelity adversarial perturbations.
- Ensure the 'Raw Data' folder and the watermark image files (AIR_32.png / AIR_64.png) are placed in the same root directory as the executable script prior to execution.
- Replace all `.cuda()` invocations with `.cpu()` if your environment is limited to CPU resources to ensure full functionality.

------------------------------------------------------------------------------------------------------------------------------------------------
1. System Requirements
------------------------------------------------------------------------------------------------------------------------------------------------
- Operating System: Windows/Linux/macOS (fully compatible with all mainstream distributions).
- Computational Acceleration: CUDA acceleration (default, recommended); CPU-only compatible by modifying `.cuda()` calls to `.cpu()` accordingly.

------------------------------------------------------------------------------------------------------------------------------------------------
2. Required Packages
------------------------------------------------------------------------------------------------------------------------------------------------
To guarantee the accurate reproduction of the algorithmic logic presented in our manuscript, please install the required dependencies using the following command:

pip install torch==1.12.1 torchvision==0.13.1 numpy==1.21.6 Pillow==9.2.0 opencv-python==4.6.0.66 scipy==1.7.3

- Version Note: Maintaining these specified package versions is critical to ensuring the reproducibility of experimental results and consistent performance of the IRAW framework.

------------------------------------------------------------------------------------------------------------------------------------------------
3. File Organization
------------------------------------------------------------------------------------------------------------------------------------------------
- Raw Data/: Houses the original input images utilized in the experimental evaluations of the IRAW framework.
- Results/: Stores adversarial examples produced by IRAW (located within the automatically generated 'scale_0.8' folder).
- Watermark Files: AIR_32.png / AIR_64.png (indispensable for the IRAW perturbation generation pipeline and core functionality).

------------------------------------------------------------------------------------------------------------------------------------------------
4. Data & Running Notes
------------------------------------------------------------------------------------------------------------------------------------------------
- File Paths: The code uses relative paths. PLEASE ENSURE ALL SCRIPTS AND DATA FOLDERS ARE KEPT IN THE SAME ROOT DIRECTORY prior to execution to avoid path-related errors.
- CUDA Acceleration: Note that the code is configured for CUDA by default; please modify the script for CPU-only environments as instructed above.
- Reproducibility: Strictly adhere to the specified package versions for the precise reproduction of the algorithmic logic and experimental results presented in our manuscript.

------------------------------------------------------------------------------------------------------------------------------------------------
We highly appreciate your consideration of our work and thank you again for your valuable time and meticulous review. We look forward to your insightful comments and constructive suggestions to further improve our research.

Sincerely yours,

Dr. Xiao-long Liu
College of Computer and Information Sciences
Fujian Agriculture and Forestry University
Email: xlliu@fafu.edu.cn
==========================================================================================
