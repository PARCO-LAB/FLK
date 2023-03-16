# FLK: a Low-Latency Biomechanics-Aware Filter for Real-time 3D Human Pose Estimation

There is a growing interest in adopting 3D human pose estimation (HPE) in safety-critical systems, from Healthcare to Industry 5.0. Nevertheless, when applied in such scenarios, state-of-the-art HPE platforms suffer from estimation inaccuracy due to different reasons. Besides imprecise or inconsistent annotations in the training dataset, the inaccuracy is due to poor image quality, rare poses, dropped frames, or heavy occlusions in the real scene. In addition, these scenarios often require the software results with temporal constraints, such as real-time and zero- or low-latency, which make many of the filtering solutions proposed in literature inapplicable.

We propose FLK, a spatio-temporal filter to refine 3D human motion data in real-time and at zero/low latency. The temporal core combines an adaptive Kalman filter, in which the motion model is learnt through a recurrent neural network, and a low-pass filter. The spatial core takes advantage of biomechanical constraints of the human body to provide spatial coherency between keypoints. The combination of the cores allows the filter to properly address different types of noise, from jittering to dropped frames.

## Requirements
To install the requirements:

```
pip install -r requirements.txt
```

Also, you need to install **tensorflow** following the [official instructions](https://www.tensorflow.org/install).

## How to run the demo

To run the example  `example.py` in the main directory:

```
python3 example.py
```