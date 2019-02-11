import os.path
import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat

class GetDataFromCSV:
    """
    Data acquisition and initial processing for SPCA proof of concept. Data is imported from AtlasClass from BuildPCA
    function, at the first iteration:
    - the energy of derivative components has already been equalized in terms of variability (1, 1/6, 1/36, 1/216);
    - the csv file is saved in the root directory as AlignedShapes.csv.
    """
    def __init__(self,  root=None, input_file=None, target_file=None):

        self.root = root
        self.input_file = input_file
        self.target_file = target_file

        self.input = self.get_input_data()
        self.shape_vectors = self.input.values # Specific for ATLAS
        if self.target_file is not None:
            self.target = self.get_target_data()
            self.labels = self.target.values.reshape((-1)) # Specific for ATLAS

    def go_to_exit(self, func):
        exit('Data acquisition failed at {} function in {} class'.format(func, type(self).__name__))

    def get_input_data(self, clean=False):
        df = self._import_data(self.input_file)
        if clean:
            df = df.loc[:, (df != 0).any(axis=0)] # if all values are 0, remove the column from df
        return df

    def get_target_data(self):
        df = self._import_data(self.target_file)
        assert len(df.index) == len(self.input.index), \
            'The number of target labels ({}) provided does not match the number of data points ({})'.format(
                len(df.index),
                len(self.input.index))
        return df

    def _import_data(self, file):
        try:
            _df = pd.read_csv(os.path.join(self.root, file), header=None)
            return _df
        except FileNotFoundError:
            print('File {} does not exist in folder {}. Check your path and file name'.format(file, self.root))
            self.go_to_exit(self._import_data.__name__)

    def center_columns(self, df):
        return df.subtract(df.mean())


class GetDataFromMAT:
    """
       Importing data from MAT files contatining PCA axes and coefficients (to provide compatibility with ATLAS class).
       - axes are saved in 'AtlasOutput<Mesh_type>\Atlas<Proj_name><<Mesh_type>.mat';
            ^ they contain information about eigenvalues and eigenvectors
       - coefficiens are saved in 'AtlasOutput<Mesh_type>\<Coordinates<Proj_name><Mesh_type>.mat;
            ^ it contains array with transformed data
       """

    def __init__(self, root=None, axis_file=None):

        self.root = root
        self.axis_file = axis_file

        self.axis = self.get_axis_data()
        self.shape_vectors = self.axis.ValidDofsInLine
        self.labels = self.get_target_data()

    def go_to_exit(self, func):
        exit('Data acquisition failed at {} function in {} class'.format(func, type(self).__name__))

    def get_axis_data(self):
        axis = self._import_data(self.axis_file)
        # assert np.array_equal(axis['ss'], axis['PCAaxis'].ss),\
        #     'Faulty eigenvalues stored in {}, recalculate!'.format(self.axis_file)
        assert np.array_equal(axis['V'], axis['PCAaxis'].V),\
            'Faulty eigenvectors stored in {} file, recalculate!'.format(self.axis_file)
        assert axis['V'].shape[0] == axis['ss'].shape[0], \
            'Number of eigenvalues ({}) does not match number of eigenvectors({}). Check the {} file'.format(
                axis['ss'].shape[0],
                axis['V'].shape[0],
                self.axis_file)
        return axis['PCAaxis']

    def get_target_data(self):
        cases = self.axis.ListCases
        target = cases % 10
        return target

    def _import_data(self, file):
        try:
            _mat = loadmat(os.path.join(self.root, file), struct_as_record=False, squeeze_me=True)
            return _mat
        except FileNotFoundError:
            print('File {} does not exist in folder {}. Check your path and file name'.format(file, self.root))
            self.go_to_exit(self._get_data.__name__)


if __name__ == '__main__':
    atlas = GetDataFromCSV(root='C:\Data\GenerationR\Atlas800', input_file='AlignedShapes.csv',
                           target_file='Phases.csv')
    print(atlas.input.head())
    print(atlas.input.shape)
    print(atlas.input.sum())

    ac = GetDataFromMAT(root='C:\Data\GenerationR\Atlas800', axis_file='AtlasOutputLVLrv\AtlasGenRLVL.mat')
    print(ac.axis.ss)
    print(ac.axis.KeyName)
    print(atlas.target.head())
    print(ac.target[:10])

    # np.savetxt('C:\Data\GenerationR\Atlas800\Differently_aligned.csv',
    #            ac.axis['PCAaxis'].ValidDofsInLine, delimiter=',')
    # savemat('C:\Data\GenerationR\Atlas800\Transformed_PCAaxis.mat', ac.axis)




