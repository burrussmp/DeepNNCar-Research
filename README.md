# DeepNNCar-Research
DeepNNCar provides a low cost physical platform to experiment with autonomous driving concepts,
specifically 1) collecting data, 2) implementing learning algorithms for autonomous driving, and 3) testing
safety critical protocols necessary for fault tolerance in autonomous systems. DeepNNCar is a flexible
framework that can be extended to support advanced inference systems that can be supported on small
single board computers such as the Raspberry Pi (RPi) with additional communication with edge or cloud
computers. Another purpose of this research was to provide a detailed setup tutorial and software
installation guide to allow other research communities to use DeepNNCar in research involving smart
cities, traffic analysis for autonomous driving, adaptive cruise control, power analysis, or resource
allocation problems.
This research is summarized in [this slideshow](https://docs.google.com/presentation/d/1GtdWFMxtsOxswUmNY0HNnKz7B1AO7px7Pbp4MSIUf90/edit#slide=id.g3eb76a4e6b_3_316).
## About
![DeepNNCar](/Images/DeepNNCar.PNG)
We present a multi-layered framework for quantifying the resilience of an end-to-end learning based autonomous system. We deploy the architecture on DeepNNCar, which is a self-driving RC car that replicates the NVIDIA's DAVE-2 Convolutional Neural Network model for steering prediction's. The framework we present here uses a Component Based Simplex Architecture (CSBA) to optimally trade-off between the performance, safety and liveness requirements of the car. The Base Controller for this simplex architecture is the Convolutional Neural Network (CNN) model which predicts the steering, and the Complex Controller is a safety monitor, which is designed with some algorithms like lane detector and blur detector, to maintain the safety properties of the car. This is supported by a decision manager, which is responsible to make a decision on the actuation (speed, steering) of the car. This framework is deployed on a resource constrained computing unit like Raspberry Pi which is placed on the chassis of the car.
## Architectural Design
![High Level Overview](/Images/HighLevelOverview.PNG)
DeepNNCar is built using the chassis of the Traxxas RC Car. The control platform is implemented on a RPi
and uses videos as the primary sensory input. Currently, the platform is capable of 3 driving modes:
remote data collection mode, autonomous steering mode, and livestream mode; all of which have their
own unique capabilities and provide a networked thin client connection. DeepNNCar is written in Python
and uses OpenCV, an open source image processing toolbox and TensorFlow, an open source machine
learning toolbox, to implement the CNN and the safety protocols.
![Controller](/Images/Controller.PNG)
## Hardware Architecture
![Circuit Schematic](/Images/CircuitSchematic.PNG)
## Software Architecture
DeepNNCar has several levels of fault tolerance. Primarily, consistent client-server communication
ensures that emergency stops can always be signaled. DeepNNCar is also designed to never exceed a set
throttle position or a maximum speed (m/s). A lane detection algorithm is used to stop DeepNNCar
automatically and to correct highly erroneous steering predictions that could lead to catastrophic failures.
Each captured image is labelled in real time with the variance of the Laplacian operator to measure image
blurriness. The results of the image quality analysis are used to adjust the throttle of DeepNNCar. Finally,
because the camera can slow down unexpectedly or the RPi can throttle the frequency of its clock due to
external conditions (weak power source or overheating) the inference timeline is monitored to ensure
safe operating conditions.
![Software Architecture](/Images/Internal.PNG)
![Safety Manager](/Images/SafetyManager.PNG)
![Decision Manager](/Images/DecisionManager.PNG)
## Potential Areas of Research
This framework can be extended in the future for other areas of research involving autonomous driving
and neural networks, including reinforcement learning, geometric-path tracking algorithms, intelligent
power planning algorithms, indoor positioning algorithms, ensemble learning, safety analysis of deep
neural networks, and more generally extending computer simulations such as CARLA or TORCS to the real
world.
## Bill of Materials
The [bill of materials] (https://docs.google.com/spreadsheets/d/1XsIUoWFDj45tv_6_x6FvWFg62A-szoD_1Ib5_Cinvhk/edit?usp=sharing)
