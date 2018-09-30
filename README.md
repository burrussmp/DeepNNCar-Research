# DeepNNCar-Research
This research is summarized in [this slideshow](https://docs.google.com/presentation/d/1GtdWFMxtsOxswUmNY0HNnKz7B1AO7px7Pbp4MSIUf90/edit#slide=id.g3eb76a4e6b_3_316).
## About
![DeepNNCar](/Images/DeepNNCar.PNG)
We present a multi-layered framework for quantifying the resilience of an end-to-end learning based autonomous system. We deploy the architecture on DeepNNCar, which is a self-driving RC car that replicates the NVIDIA's DAVE-2 Convolutional Neural Network model for steering prediction's. The framework we present here uses a Component Based Simplex Architecture (CSBA) to optimally trade-off between the performance, safety and liveness requirements of the car. The Base Controller for this simplex architecture is the Convolutional Neural Network (CNN) model which predicts the steering, and the Complex Controller is a safety monitor, which is designed with some algorithms like lane detector and blur detector, to maintain the safety properties of the car. This is supported by a decision manager, which is responsible to make a decision on the actuation (speed, steering) of the car. This framework is deployed on a resource constrained computing unit like Raspberry Pi which is placed on the chassis of the car.
## Architectural Design
![High Level Overview](/Images/HighLevelOverview.PNG)
![Controller](/Images/Controller.PNG)
## Hardware Architecture
![Circuit Schematic](/Images/CircuitSchematic.PNG)
## Software Architecture
![Software Architecture](/Images/Internal.PNG)
![Safety Manager](/Images/SafetyManager.PNG)
![Decision Manager](/Images/DecisionManager.PNG)
## Bill of Materials
The [bill of materials] (https://docs.google.com/spreadsheets/d/1XsIUoWFDj45tv_6_x6FvWFg62A-szoD_1Ib5_Cinvhk/edit?usp=sharing)
