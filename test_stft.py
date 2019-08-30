#!/usr/bin/env python

import unittest
from stft import window_nonzero, create_overlapping_segments, stft, istft
from stft import istft
import numpy as np
from scipy.signal.windows import boxcar, hann, tukey

class TestCode(unittest.TestCase):

    def test_window_nonzero(self):
        """Test window_nonzero function"""

        # boxcar window
        segment_length = 16        
        window_function = boxcar
        window_vector = window_nonzero(window_function, segment_length)
        self.assertTrue(segment_length == np.count_nonzero(window_vector))

        # hann window has zeros on the ends
        segment_length = 16        
        window_function = hann
        window_vector = window_nonzero(window_function, segment_length)
        self.assertTrue(segment_length == np.count_nonzero(window_vector))

        # tukey has zeros on the ends
        segment_length = 16        
        window_function = tukey
        window_vector = window_nonzero(window_function, segment_length)
        self.assertTrue(segment_length == np.count_nonzero(window_vector))
        
    def test_create_overlapping_segments(self):
        """Test create_overlapping_segments"""


        x = np.arange(16)
        segment_length = 4
        shift_length = 2
        x_segments, start_list, stop_list = create_overlapping_segments(x,
            segment_length, shift_length)
        self.assertTrue(x_segments.shape[0] == segment_length)
        self.assertTrue(x_segments.shape[1] == 7)

        x = np.arange(16)
        segment_length = 16
        shift_length = 2
        x_segments, start_list, stop_list = create_overlapping_segments(x,
            segment_length, shift_length)
        self.assertTrue(x_segments.shape[0] == segment_length)
        self.assertTrue(x_segments.shape[1] == 1)

        x = np.arange(16)
        segment_length = 8
        shift_length = 5
        x_segments, start_list, stop_list = create_overlapping_segments(x,
            segment_length, shift_length)
        self.assertTrue(x_segments.shape[0] == segment_length)
        self.assertTrue(x_segments.shape[1] == 3)

        
        x = np.random.randn(16, 16, 16)
        segment_length = 8
        shift_length = 5
        x_segments, start_list, stop_list = create_overlapping_segments(x,
            segment_length, shift_length)
        # check shape of x_segments
        self.assertTrue(x_segments.shape[0] == segment_length)
        self.assertTrue(x_segments.shape[1] == 3)
        self.assertTrue(x_segments.shape[2] == 16)
        self.assertTrue(x_segments.shape[3] == 16)
        # check first segment
        self.assertTrue(np.allclose(x[0:segment_length, 0, 0], x_segments[:, 0, 0, 0]))
        # check last segment
        self.assertTrue(np.allclose(x[8:16, -1, -1], x_segments[:, -1, -1, -1]))


    def test_stft(self):
        """Test stft"""
        x = np.ones(16)
        segment_length = 4
        shift_length = 2
        segment_length_padded = 4
        window_function = boxcar
        # take stft
        x_stft, start_list, stop_list = stft(x, segment_length,
            segment_length_padded, shift_length, window_function)
        # setup what stft output should be
        x_stft_true = np.zeros((3, 7))
        x_stft_true[0] = 4
        self.assertTrue( np.allclose(x_stft, x_stft_true) )

        x = np.ones(16)
        segment_length = 4
        shift_length = 2
        segment_length_padded = 4
        window_function = hann
        # take stft
        x_stft, start_list, stop_list = stft(x, segment_length,
            segment_length_padded, shift_length, window_function)
        # setup what stft output should be
        window_vector = window_nonzero(hann, segment_length)
        x_stft_true = np.zeros((3, 7), dtype=np.complex)
        window_vector_fft = np.fft.rfft(window_vector)
        x_stft_true[0] = window_vector_fft[0]
        x_stft_true[1] = window_vector_fft[1]
        x_stft_true[2] = window_vector_fft[2]
        self.assertTrue( np.allclose(x_stft, x_stft_true) )

        x = np.ones((16, 2, 2))
        segment_length = 4
        shift_length = 2
        segment_length_padded = 4
        window_function = boxcar
        # take stft
        x_stft, start_list, stop_list = stft(x, segment_length,
            segment_length_padded, shift_length, window_function)
        # setup what stft output should be
        x_stft_true = np.zeros((3, 7, 2, 2))
        x_stft_true[0] = 4
        self.assertTrue( np.allclose(x_stft, x_stft_true) )

        x = np.ones((16, 2, 2))
        segment_length = 4
        shift_length = 2
        segment_length_padded = 4
        window_function = hann
        # take stft
        x_stft, start_list, stop_list = stft(x, segment_length,
            segment_length_padded, shift_length, window_function)
        # setup what stft output should be
        window_vector = window_nonzero(hann, segment_length)
        x_stft_true = np.zeros((3, 7, 2, 2), dtype=np.complex)
        window_vector_fft = np.fft.rfft(window_vector)
        x_stft_true[0] = window_vector_fft[0]
        x_stft_true[1] = window_vector_fft[1]
        x_stft_true[2] = window_vector_fft[2]
        self.assertTrue( np.allclose(x_stft, x_stft_true) )


    def test_istft(self):

        """Test istft"""
        x = np.ones(16)
        segment_length = 4
        shift_length = 2
        segment_length_padded = 4
        window_function = boxcar
        original_size = x.shape
        p = 1
        x_stft, start_list, stop_list = stft(x, segment_length,
            segment_length_padded, shift_length, window_function)
        x_out = istft(x_stft, segment_length, segment_length_padded,
            start_list, stop_list,
            original_size, window_function, p)
        self.assertTrue( np.allclose(x_out, x))

        x = np.random.randn(16)
        segment_length = 4
        shift_length = 2
        segment_length_padded = 4
        window_function = boxcar
        original_size = x.shape
        p = 1
        x_stft, start_list, stop_list = stft(x, segment_length,
            segment_length_padded, shift_length, window_function)
        x_out = istft(x_stft, segment_length, segment_length_padded,
            start_list, stop_list,
            original_size, window_function, p)
        self.assertTrue( np.allclose(x_out, x))


        x = np.random.randn(16, 2, 2)
        segment_length = 4
        shift_length = 2
        segment_length_padded = 4
        window_function = boxcar
        original_size = x.shape
        p = 1
        x_stft, start_list, stop_list = stft(x, segment_length,
            segment_length_padded, shift_length, window_function)
        x_out = istft(x_stft, segment_length, segment_length_padded,
            start_list, stop_list,
            original_size, window_function, p)
        self.assertTrue( np.allclose(x_out, x))

        x = np.ones(16)
        segment_length = 4
        shift_length = 2
        segment_length_padded = 4
        window_function = hann
        original_size = x.shape
        p = 1
        x_stft, start_list, stop_list = stft(x, segment_length,
            segment_length_padded, shift_length, window_function)
        x_out = istft(x_stft, segment_length, segment_length_padded,
            start_list, stop_list,
            original_size, window_function, p)
        self.assertTrue( np.allclose(x_out, x))

        x = np.ones(16)
        segment_length = 4
        shift_length = 4
        segment_length_padded = 4
        window_function = hann
        original_size = x.shape
        p = 2
        x_stft, start_list, stop_list = stft(x, segment_length,
            segment_length_padded, shift_length, window_function)
        x_out = istft(x_stft, segment_length, segment_length_padded,
            start_list, stop_list,
            original_size, window_function, p)
        self.assertTrue( np.allclose(x_out, x))

        x = np.ones(16)
        segment_length = 4
        shift_length = 2
        segment_length_padded = 4
        window_function = boxcar
        original_size = x.shape
        p = 2
        x_stft, start_list, stop_list = stft(x, segment_length,
            segment_length_padded, shift_length, window_function)
        x_out = istft(x_stft, segment_length, segment_length_padded,
            start_list, stop_list,
            original_size, window_function, p)
        self.assertTrue( np.allclose(x_out, x))


        x = np.ones(16)
        segment_length = 4
        shift_length = 2
        segment_length_padded = 4
        window_function = hann
        original_size = x.shape
        p = 2
        x_stft, start_list, stop_list = stft(x, segment_length,
            segment_length_padded, shift_length, window_function)
        x_out = istft(x_stft, segment_length, segment_length_padded,
            start_list, stop_list,
            original_size, window_function, p)
        self.assertTrue( np.allclose(x_out, x))

        x = np.random.randn(16)
        segment_length = 4
        shift_length = 2
        segment_length_padded = 4
        window_function = hann
        original_size = x.shape
        p = 2
        x_stft, start_list, stop_list = stft(x, segment_length,
            segment_length_padded, shift_length, window_function)
        x_out = istft(x_stft, segment_length, segment_length_padded,
            start_list, stop_list,
            original_size, window_function, p)
        self.assertTrue( np.allclose(x_out, x))

        x = np.random.randn(16, 3)
        segment_length = 4
        shift_length = 2
        segment_length_padded = 4
        window_function = hann
        original_size = x.shape
        p = 2
        x_stft, start_list, stop_list = stft(x, segment_length,
            segment_length_padded, shift_length, window_function)
        x_out = istft(x_stft, segment_length, segment_length_padded,
            start_list, stop_list,
            original_size, window_function, p)
        self.assertTrue( np.allclose(x_out, x))

        x = np.ones(16)
        segment_length = 4
        shift_length = 2
        segment_length_padded = 7
        window_function = boxcar
        original_size = x.shape
        p = 1
        x_stft, start_list, stop_list = stft(x, segment_length,
            segment_length_padded, shift_length, window_function)
        x_out = istft(x_stft, segment_length, segment_length_padded, 
            start_list, stop_list,
            original_size, window_function, p)
        self.assertTrue( np.allclose(x_out, x))




if __name__ == '__main__':
    print("Running unit tests for stft.py")
    unittest.main()