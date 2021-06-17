import numpy as np

class MaxPool2:
    # A Max Pooling layer using a pool size of 2

    def iterate_regions(self, image):
        '''
        Generates non-overlapping 2x2 image regions to pool over
        - image is a 2d nump array
        '''
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                image_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield image_region, i, j

    def forward(self, input):
        '''
        Performs forward pass of the maxpool layer using the given input.
        Returns 3d numpy array with dimensions (h / 2, w / 2, num_filters)
        - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for image_region, i, j in self.iterate_regions(input):

            # uses numpy array max method
            # set axis=(0, 1) to maximize over height and weight only (not num_filters)
            output[i, j] = np.amax(image_region, axis=(0, 1))

        return output