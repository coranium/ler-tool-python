
import numpy as np
from PIL import Image
from skimage.feature import canny


class LER():

    def analyse(self, imgpath):

        # Open image (should be binary mask image) and convert to np array.
        img_pil = Image.open(imgpath).convert('L')
        img_np = np.array(img_pil)

        # Get edge image using canny detector
        edges = canny(img_np, sigma=3)
        edge_image = Image.fromarray(edges)
        edge_image.save('edges.jpg')

        # Coordinate points of edges
        coords = np.where(edges == True)
        coords = np.array([coords[0], coords[1]])

        line_stats = self.get_lines(coords.T)
        computed_stats = self.process_lines(line_stats)

        return np.array(computed_stats)

    def get_lines(self, coords):
        xcoords, xcounts = np.unique(coords[:, 1], return_counts=True)

        line_count = 0
        line_stats = []
        line_stat = []

        for i, xpos in enumerate(xcoords):
            count = xcounts[i]
            line_stat.append([xpos, count])

            if i != len(xcoords)-1:  # if not last xposition
                if xcoords[i+1] - xpos != 1:  # if next xpos is not 1 away
                    line_count += 1
                    line_stats.append(np.array(line_stat))
                    line_stat = []

        line_stats.append(np.array(line_stat))

        return line_stats

    def process_lines(self, linestats):
        computed_stats = []
        for stat in linestats:
            val = stat[:, 0]
            freq = stat[:, 1]
            if freq.sum() < 254:
                continue

            mean = np.average(val, weights=freq)
            dev = freq * (val-mean)**2  # squared deviations from mean
            var = dev.sum()/freq.sum()
            std = np.sqrt(var)

            computed_stats.append([mean, var, std])

        return computed_stats
