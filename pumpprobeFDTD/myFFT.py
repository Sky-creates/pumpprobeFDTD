import numpy as np
import numpy.fft as fft
def FFT1D(t,s,n=0):
    if n<=0:
        n = len(t)
    scale = np.abs(t[1]-t[0])
    s1 = (fft.fft(s,n))*scale
    s2 = fft.fftshift(s1)
    f = fft.fftshift(fft.fftfreq(n))/scale
    return (f,s2)

def IFFT1D(f,s):
    scale = np.abs(f[1]-f[0])
    s0 = fft.ifftshift(s)
    s1 = (fft.ifft(s0))*scale
    tnew = np.arange(len(f))/scale
    return (tnew,s1)

def FFT2D(t1,t2,s,n1=0,n2=0):
    if n1<=0:
        n1 = len(t1)
    if n2<=0:
        n2 = len(t2)
    scale1 = np.abs(t1[0]-t1[1])
    scale2 = np.abs(t2[0]-t2[1])
    # note that s is real, conj converts frequency sign in the transformation
    # we correct for prefactor to get a good fourier transformation
    s1 = np.conj(fft.fft2(s,[n1,n2]))*scale1*scale2
    s2 = fft.fftshift(s1)
    f1 = fft.fftshift(fft.fftfreq(n1))/scale1
    f2 = fft.fftshift(fft.fftfreq(n2))/scale2
    return (f1,f2,s2)