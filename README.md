Design and Implementation of a Cross-Platform Biometric Analysis Engine

Project Code Name: PoseShow Pro

1. Abstract

PoseShow Pro is a cross-platform human pose estimation application developed to provide real-time biometric feedback. The system integrates state-of-the-art neural networks, specifically targeting a minimum of 15 FPS on mobile devices to satisfy real-world performance requirements . This project explores the trade-offs between model complexity and inference latency across diverse hardware environments.

2. System ArchitectureFollowing a modular software engineering approach, the application utilizes:  

-Core Engine: MediaPipe Pose (BlazePose) and MoveNet-inspired topologies.  

-Deployment: Streamlit-based Web-Native Hybrid for instant cross-platform compatibility on Android and iOS.  

-Data Layer: Local on-device processing to ensure user privacy and biometric security.

3. Key Features (Requirement #2)Symmetrical Analysis:

-Side-by-side rendering of raw input and analyzed pose. 

-Multi-Model Support: Toggle between 33-point (MediaPipe), 17-point (MoveNet), and 18-point (OpenPose) skeletons. 

-Quantitative Logging: Session data export to CSV (Latency, Confidence, and FPS) for academic evaluation.  

-Mobile-Optimized UI: Responsive design with health app aesthetics. 

4. Installation & Reproducibility 
To replicate the development environment and run the application:  

-PrerequisitesPython 3.9 or higher

-Pip package manager

Steps

1. Clone the repository:

2. Install Dependencies
pip install -r requirements.txt

3. Launch the application

Bash
streamlit run [name of file].py

5. Performance Benchmarks
Metric             Target Requirement        Measured Result (Avg)
Inference Latency       < 100ms             ~45ms - 80ms
System FPS>             15 FPS               18 - 24 FPS 
Model Accuracy        >80% Confidence        88% - 92%

6. Ethical Considerations & Privacy

In compliance with the AUTHOR'S STATEMENT on academic ethics, this system processes all biometric data locally. No image data or skeletal coordinates are transmitted to external servers.

Author: 赞恩

Supervisor: 袁玉波

Institution: Information Science and Engineering, Computer Science Department