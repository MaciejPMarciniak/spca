import numpy as np
from data_acquisition import GetDataFromCSV, GetDataFromMAT
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# TODO: write a report and describe current issues.


class SPCA:

    def __init__(self, root='C:\Data\GenerationR\Atlas800', data_file1='AlignedShapes.csv',
                 data_file2='Phases.csv', components=None):

        self.root = root
        self.data_file1 = data_file1
        self.data_file2 = data_file2
        self.components = components
        self.data = self.get_data()

        self.X = self.get_x()
        self.H = self.get_h()
        self.L = self.get_l(self.ed_es_kernel, scale=True, case='no_samples_squared')

        self.Q = self.calculate_q(make_pd=True)
        self.eigenvalues_q, self.U = self.compute_basis()
        # self.Z = self.get_z()
        # self.save_results()

    # -----Input-Output management----------------------------------------------------------------------------
    def get_data(self):
        _, ext = os.path.splitext(self.data_file1)
        if ext == '.csv':
            return GetDataFromCSV(self.root, self.data_file1, self.data_file2)
        elif ext == '.mat':
            return GetDataFromMAT(self.root, self.data_file1)
        else:
            exit('File extension: {} not recognized'.format(ext))

    def save_results(self):
        np.savetxt('SPCA_eigenvalues.csv', self.eigenvalues_q, delimiter=',')
        np.savetxt('SPCA_eigenvectors.csv', self.U, delimiter=',')
    # --------------------------------------------------------------------------------------------------------

    def get_x(self):
        """
        Retrieve X from data imported in a form N x M.
        :return: Function returns a transposed representation of the data set, to comply with the SPCA article
        algorithm. Size: M x N.
        """
        x_t = self.data.shape_vectors
        x = x_t.T
        return x

    def get_h(self):
        """
        Create centering matrix of size corresponding to input data.
        :return: Square centering matrix H. Size: N x N.
        """
        i = np.eye(self.X.shape[1])
        ee = np.ones(i.shape) * (1 / i.shape[0])
        return i - ee

    def get_l(self, kernel=None, **kwargs):
        """
        Create target matrix using a kernel. Default is just vector multiplication.
        :param kernel: Defines a function that target matrix should follow.
        :return: _L - Target matrix. Size: N x N
        :return: sort - the sorting order if target vector required permutation.
        """
        if kernel is not None:
            _L = kernel(**kwargs)
        else:
            exit('Provide viable kernel function')
        return _L

    def calculate_q(self, make_pd=True):
        """
        Multiplication of input and target matrices.
        :param make_pd: Correct for positive definiteness if necessary.
        :return: Result of multiplication. Size: M x M
        """
        q = self.X @ self.H @ self.L @ self.H @ self.X.T
        print(is_pd(q))
        if make_pd:
            q = nearest_pd(q)
        return q

    def compute_basis(self):
        """
        Calculation of eigenvalues and corresponding eigenvectors.
        :return: Tuple with eigenvalues and as many eigenvectors as set in 'components' argument.
        """
        print('Calculating eigenvectors...')
        w, v = np.linalg.eigh(self.Q)
        _w_sort = w.argsort()
        w = w[_w_sort]
        v = v[:, _w_sort]
        return w, v[:, :self.components]

    def get_z(self):
        """
        Calculate latent variables
        :return: Samples transformed with some version of PCA. Size depends on the number of components.
        """
        return self.U.T @ self.X

    # -----Kernels used for calculating the target matrix-----------------------------------------------------
    def identity_kernel(self):
        """
        An identity matrix to perform regular PCA (and confirm correctness)
        :return: Identity matrix. Size: N x N.
        """
        return np.eye((self.H.shape[0]))

    def ed_es_kernel(self, **kwargs):
        """Rows of the matrix contain as many ones as there are samples from the same group, either on the
        left or on the right side. The labels of end-diastole phase and end-systole.
        :return: delta - kernel matrix as described in the SPCA article. Size N x N.
        """
        _target = self.data.labels
        _scale = kwargs['scale'] if 'scale' in kwargs else False
        _case = kwargs['case'] if 'case' in kwargs else None

        _target = _target.reshape((-1, 1))  # reshape - to create dimension for flattened array
        _target[_target == 2] = -1
        delta = (_target @ _target.T)
        delta[delta == -1] = 0

        if _scale:
            print('Scaling non-diagonal values...')
            if _case == 'no_samples':
                denom = len(_target)
            elif _case == 'no_samples_squared':
                denom = len(_target) ** 2
            elif _case == 'sum':
                denom = np.sum(delta)
            elif _case == 'squared_sum':
                denom = np.sum(delta) ** 2
            else:
                denom = 1.0

            print(delta[:10, :10])
            # np.fill_diagonal(delta, 0)
            delta = delta / denom

        print(delta[:10, :10])
        return delta

    def diagonal_kernel(self, **kwargs):
        """
        This kernel creates a diagonal matrix of the enhanced target and its transposition. The values
        connected with labels are reciprocals.
        :param kwargs:
        **'enhancement' - this parameter changes the values of the labels to 'en' and 1/'en'.
        :return: Diagonal matrix containing the enhanced values of labels of the sample. Size N x N.
        """
        _enhancement = kwargs['enhancement'] if 'enhancement' in kwargs else np.sqrt(2)
        _target = self.data.labels.astype(float)
        _enhanced_target = np.zeros(len(_target))

        _enhanced_target[_target == 2] = _enhancement
        _enhanced_target[_target == 1] = 1/_enhancement
        delta = np.diag(_enhanced_target)
        # -----Testing convoluted kernels -----
        # kernel_e = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        # convolution = convolve2d(delta, kernel_e, mode='same', boundary='wrap')
        # delta = nearest_pd(convolution)
        # plt.imshow(delta[:20, :20])
        # -------------------------
        print('Labels: ED = {}, ES = {}'.format(delta[0, 0], delta[1, 1]))
        return delta

    def dot_product_kernel(self, **kwargs):
        """
        This kernel performs vector multiplication of the enhanced target and its transposition. The values
        connected with labels are reciprocals.
        :param kwargs:
        **'enhancement' - this parameter changes the values of the labels to 'en' and 1/'en'.
        **'scale' - given as 'True', all values that result in 1 (multiplications of inverted values)
        are divided by the sum of all elements in the delta matrix.
        :return: Full matrix containing the enhanced values of labels of the sample. Size N x N.
        """
        _enhancement = kwargs['enhancement'] if 'enhancement' in kwargs else np.sqrt(2)
        _scale = kwargs['scale'] if 'scale' in kwargs else False
        _target = self.data.labels.astype(float)

        _enhanced_target = np.zeros((len(_target), 1))
        _enhanced_target[_target == 1] = 1/_enhancement
        _enhanced_target[_target == 2] = _enhancement
        delta = _enhanced_target @ _enhanced_target.T

        if _scale:
            print('Scaling non-diagonal values...')
            _delta_sum = np.sum(_enhanced_target)
            print(delta[:10, :10])
            # delta[delta == 1] = 0
            np.fill_diagonal(delta, 0)
            delta /= _delta_sum
            delta = delta + np.diag(_enhanced_target ** 2)
        return delta

    def centroid_kernel(self, **kwargs):
        _scale = kwargs['scale'] if 'scale' in kwargs else 1.0
        _target = self.data.labels.astype(float)
        group_a = self.X[:, _target == 1]
        group_b = self.X[:, _target != 1]

        diff_a = get_centroids(group_a, group_b, show_populations=True) * _scale
        diff_b = get_centroids(group_b, group_a, show_populations=True) * _scale

        diff_vector = np.zeros(len(_target))
        diff_vector[_target == 1] = diff_a
        diff_vector[_target != 1] = diff_b
        print(diff_vector[:20])
        delta = np.diag(diff_vector)
        return delta

    def rbf_kernel(self, **kwargs):
        _sigma = kwargs['scale'] if 'scale' in kwargs else 1.0
        delta = np.zeros(self.H.shape)
        for i, x_i in enumerate(self.X.T):
            for j, x_j in enumerate(self.X.T):
                norm_ij = np.linalg.norm(x_i - x_j, axis=0)

                delta[i, j] = np.exp(-(norm_ij/(2*_sigma)) ** 2)
                delta[j, i] = delta[i, j]
                if j == i:
                    break
        print(delta[:10, :10])
        return delta

    # ----END OF SPCA CLASS-----------------------------------------------------------------------------------


# -----Correcting for positive definiteness-------------------------------------------------------------------
def nearest_pd(input_matrix):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    print('Correcting the matrix to be positive definite...')
    mean_symmetric_matrix = (input_matrix + input_matrix.T) / 2
    _, singular_values, row_eigenvectors = np.linalg.svd(mean_symmetric_matrix)
    symmetric_polar_factor = np.dot(row_eigenvectors.T, np.dot(np.diag(singular_values), row_eigenvectors))

    refined_mean_symmetric_matrix = (mean_symmetric_matrix + symmetric_polar_factor) / 2
    output_matrix = (refined_mean_symmetric_matrix + refined_mean_symmetric_matrix.T) / 2

    if is_pd(output_matrix):
        summarize_nearest_pd_transformation(input_matrix, output_matrix)
        return output_matrix
    spacing = np.spacing(np.linalg.norm(input_matrix))
    i = np.eye(input_matrix.shape[0])
    k = 1
    while not is_pd(output_matrix):
        min_eig = np.min(np.real(np.linalg.eigvals(output_matrix)))
        output_matrix += i * (-min_eig * k**2 + spacing)
        k += 1
        summarize_nearest_pd_transformation(input_matrix, output_matrix)
    return output_matrix


def is_pd(matrix):
    """Returns true when input is positive-definite, via Cholesky decomposition attempt"""
    try:
        _ = np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError as e:
        print(e)
        return False


def summarize_nearest_pd_transformation(matrix, nearest_pd_matrix):
    _is_dif = np.isclose(matrix, nearest_pd_matrix, atol=1e-100)
    _dif = np.sqrt((matrix[np.invert(_is_dif)] - nearest_pd_matrix[np.invert(_is_dif)]) ** 2)
    print('Number of altered cells: {}'.format(len(nearest_pd_matrix[np.invert(_is_dif)])))
    print('Sum of all absolute differences: {}'.format(np.sum(_dif)))
    if np.any(_dif):
        print('Maximum difference between cells: {}'.format(np.max(_dif)))
        print('Minimum difference between cells: {}'.format(np.min(_dif)))
# ------------------------------------------------------------------------------------------------------------


# Additional functions:
def get_centroids(population_a, population_b, show_populations=False):
    population_a_mean = np.mean(population_a, axis=1).reshape(-1, 1)
    population_b_mean = np.mean(population_b, axis=1).reshape(-1, 1)
    norms_from_opposite_centroid = np.linalg.norm(population_a - population_b_mean, axis=0)
    norms_from_own_centroid = np.linalg.norm(population_a - population_a_mean, axis=0)

    distance = norms_from_opposite_centroid - norms_from_own_centroid
    distance = distance + np.abs(np.min(distance))

    if show_populations:
        print('mean: {}, median: {}, max: {}, min: {}'.format(np.mean(norms_from_opposite_centroid),
                                                              np.median(norms_from_opposite_centroid),
                                                              np.max(norms_from_opposite_centroid),
                                                              np.min(norms_from_opposite_centroid)))
        print('mean: {}, median: {}, max: {}, min: {}'.format(np.mean(norms_from_own_centroid),
                                                              np.median(norms_from_own_centroid),
                                                              np.max(norms_from_own_centroid),
                                                              np.min(norms_from_own_centroid)))
        sorted_index_norms = np.argsort(norms_from_opposite_centroid)
        sorted_diff_b = np.sort(norms_from_opposite_centroid)  # [sorted_index_norms]
        sorted_diff_bplus = np.sort(norms_from_own_centroid)  # [sorted_index_norms]
        sorted_distance = np.sort(distance)
        plt.plot(sorted_diff_b)
        plt.plot(sorted_diff_bplus)
        plt.plot(sorted_distance)
        d_0 = np.percentile(sorted_distance, 0, interpolation='nearest')
        d_25 = np.percentile(sorted_distance, 25, interpolation='nearest')
        d_50 = np.percentile(sorted_distance, 50, interpolation='nearest')
        d_75 = np.percentile(sorted_distance, 75, interpolation='nearest')
        d_100 = np.percentile(sorted_distance, 100, interpolation='nearest')
        plt.axhline(y=d_0, xmin=0, xmax=50/850, color='red')
        plt.axhline(y=d_25, xmin=0, xmax=(np.where(sorted_distance == d_25)[0]) / 800,
                    color='orange')
        plt.axhline(y=d_50, xmin=0, xmax=(np.where(sorted_distance == d_50)[0]) / 800,
                    color='black')
        plt.axhline(y=d_75, xmin=0, xmax=(np.where(sorted_distance == d_75)[0]) / 800,
                    color='lightblue')
        plt.axhline(y=d_100, xmin=0, xmax=(np.where(sorted_distance == d_100)[0]) / 800,
                    color='blue')
        plt.xlim((-50, 800))
        plt.show()
        return norms_from_opposite_centroid


def is_symmetric(matrix, tol=1e-8):
    return 'is symmetric' if np.allclose(matrix, matrix.T, atol=tol) else 'is not symmetric'


def normalize_0_1(vector):
    return (vector-min(vector))/(max(vector)-min(vector))


def normalize_div_by_sum(vector):
    return vector/np.sum(vector)


if __name__ == '__main__':
    atlas_mat = SPCA(root='C:\Data\GenerationR\Atlas800', data_file1='AtlasOutputLVLrv\AtlasGenRLVL.mat',
                     components=None)
    atlas_mat.save_results()
    # print(np.sum(np.iscomplex(atlas_mat.U)))
    print('First 10 normalized eigenvalues')
    print(normalize_div_by_sum(atlas_mat.eigenvalues_q)[-10:])
    print('First 5 eigenvalues')
    print(atlas_mat.eigenvalues_q[-5:])
    print('Sum of first 10 eigenvalues')
    print(sum(normalize_div_by_sum(atlas_mat.eigenvalues_q)[-10:]))
