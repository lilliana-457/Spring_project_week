Integrated Robotics System: Line Following & Computer Vision
This project focuses on the development and integration of an autonomous robotics system using a Raspberry Pi 4. It combines precise PID-controlled line following with real-time symbol classification using computer vision.

Project Overview:

The primary objective is to integrate separate robotics tasks—motion control and object detection—into a single, cohesive system utilizing Python Threading for concurrent execution.

Key Features:

PID Line Following: Implements Proportional-Integral-Derivative control to maintain smooth navigation.

Computer Vision (ORB): 
Uses Oriented FAST and Rotated BRIEF (ORB) feature matching for shape and symbol classification.

Multi-threaded Integration: 
High-level synchronization of vision and motion sub-processes.

Hardware Interface: 
GPIO control of L298N motor drivers for precise wheel movement.

