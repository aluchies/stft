# stft

Short-time Fourier transform (STFT) and inverse short-time Fourier transform (ISTFT) in python.

## Overview

This code is based on the algorithm proposed in [1], the reader is referred there for theoretical details. The input data is assumed to be real (use rfft/irfft instead of fft/ifft), it can be applied to an n-d array, the segmentation is always along the first dimension, overlapping segments are allowed, zero padding is allowed, and the ISTFT can be performed using overlap-and-add (p=1) or a least squares approach (p=2). Additional works that use this stft/istft approach include [2-4].

## References

[1] B Yang, "A study of inverse short-time Fourier transform," in Proc. of ICASSP, 2008.

[2] B Byram et al, "Ultrasonic multipath and beamforming clutter reduction: a chirp model approach," IEEE UFFC, 61, 3, 2014.

[3] B Byram et al, "A model and regularization scheme for ultrasonic beamforming clutter reduction, " IEEE UFFC, 62, 11, 2015.

[4] A Luchies et al, "Deep neural networks for ultrasound beamforming," IEEE TMI, 37, 9, 2018.
