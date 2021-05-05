import cv2  # (OpenCV3)
from LER import edge_roughness
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image

#image = cv2.imread("Spec_test.bmp")
binary_image = Image.open('test.jpg').convert('L')

LER = edge_roughness()
# image, image width (m), threshold for outliers
# (Xcln, Ycln, freq, FourierPow) = LER.LER_analysis(image, 50e-6, 2)

# use binary image function
(Xcln, Ycln, freq, FourierPow) = LER.analyse_binary(binary_image, 50e-6, 2)

plt.plot(Xcln, Ycln, '-')
plt.plot(Xcln, Ycln, 'r.')
plt.show()

plt.plot(freq[2:100], FourierPow[2:100], '-')
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
plt.xlabel('cycles (m$^{-1}$)')
plt.ylabel('Fourier power (au)')
plt.title('Fourier power spectrum')
plt.show()

# cv2.imshow('Origional Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
