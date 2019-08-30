# stft

Short-time Fourier transform (STFT) and inverse short-time Fourier transform (ISTFT) in python. This code is based on the algorithm proposed in [1], the reader is referred there for theoretical details. It can be applied to an n-d array, the input data is assumed to be real (use rfft/irfft instead of fft/ifft), the segmentation is always along the first dimension, overlapping segments are allowed, and the ISTFT can be performed using overlap-and-add or a least squares approach. Additional works that use this stft/istft approach include [2-4].

[1] B Yang, "A study of inverse short-time Fourier transform," in Proc. of ICASSP, 2008.
[2] B Byram et al, "Ultrasonic multipath and beamforming clutter reduction: a chirp model approach," IEEE UFFC, 61, 3, 2014.
[3] B Byram et al, "A model and regularization scheme for ultrasonic beamforming clutter reduction, " IEEE UFFC, 62, 11, 2015.
[4] A Luchies et al, "Deep nerual networks for ultrasound beamforming," IEEE TMI, 37, 9, 2018.
