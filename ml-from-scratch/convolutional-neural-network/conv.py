import numpy as np

class Conv3x3:
    # A Convolution layer using 3x3 filters

    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters is a 3d array with dimension (num_filters, 3, 3)
        # divide by 9 to reduce variance of intial values 
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        '''
        Generates all possible 3x3 image regions using valid padding
        - image is 2d numpy array
        '''
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                image_region = image[i:(i + 3), j:(j + 3)]

                # generator --> iterable you can only iterate over once
                    # handy when you know funct will return huge set of values
                    # that we will only need to read once
                yield image_region, i , j 

    def forward(self, input):
        '''
        Performs forward pass of the conv layer using given input.
        Returns a 3d numpy array with dimensions (h, w, num_filters)
        - input is 2d numpy array
        '''
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for image_region, i, j, in self.iterate_regions(input):

            # contains convolution results for pixel(i, j)
            # image_region * self.filters uses numpy broadcasting feature
            # np.sum(...) prod 1d array of length num_filters 
                # ea elem contains convolution result for corresponding filter
            output[i, j] = np.sum(image_region * self.filters, axis=(1,2))

        return output

