# FLK: a Filter with Learned Kinematics for Real-time 3D Human Pose Estimation
[Official website](https://parco-lab.github.io/FLK/)
[Paper](http://dx.doi.org/10.1016/j.sigpro.2024.109598)

There is a growing interest in adopting 3D human pose estimation (HPE) in safety-critical systems, from Healthcare to Industry 5.0. Nevertheless, when applied in such scenarios, state-of-the-art HPE platforms suffer from estimation inaccuracy due to different reasons. Besides imprecise or inconsistent annotations in the training dataset, the inaccuracy is due to poor image quality, rare poses, dropped frames, or heavy occlusions in the real scene. In addition, these scenarios often require the software results with temporal constraints, such as real-time and zero- or low-latency, which make many of the filtering solutions proposed in literature inapplicable.

We propose FLK, a spatio-temporal filter to refine 3D human motion data in real-time and at zero/low latency. The temporal core combines an adaptive Kalman filter, in which the motion model is learnt through a recurrent neural network, and a low-pass filter. The spatial core takes advantage of biomechanical constraints of the human body to provide spatial coherency between keypoints. The combination of the cores allows the filter to properly address different types of noise, from jittering to dropped frames.

## Installation
Create a virtual environment:
```
git clone git@github.com:PARCO-LAB/FLK.git
cd FLK
python3.8 -m virtualenv .venv
source .venv/bin/activate
```

Install the requirements:

```
pip install -r requirements.txt
```

PS: if you encounter any trouble, you need to install **tensorflow** following the [official instructions](https://www.tensorflow.org/install).

Install the package:

```
python3 -m pip install --upgrade build
python3 -m build
cd dist
pip3 install flk-0.0.1-py3-none-any.whl
```

## How to run the demo

To run the example  `example.py` in the main directory:

```
python3 example.py
```

## Citation

```
@article{Martini2024FLK,
  title = {FLK: A filter with learned kinematics for real-time 3D human pose estimation},
  ISSN = {0165-1684},
  url = {http://dx.doi.org/10.1016/j.sigpro.2024.109598},
  DOI = {10.1016/j.sigpro.2024.109598},
  journal = {Signal Processing},
  publisher = {Elsevier BV},
  author = {Martini,  Enrico and Boldo,  Michele and Bombieri,  Nicola},
  year = {2024},
  month = jul,
  pages = {109598}
}
```

