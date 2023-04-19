from os.path import join

from gpmap.src.utils import check_error, read_dataframe
from gpmap.src.settings import DATASETS, RAW_DATA_DIR, LANDSCAPES_DIR
from gpmap.src.space import SequenceSpace


class DataSet(object):
    def __init__(self, dataset):
        msg = 'Dataset not recognized: try one of {}'.format(DATASETS)
        check_error(dataset in DATASETS, msg=msg)
        
        self.name = dataset
        self.load_data()
        self.load_landscape()
    
    def load_data(self):
        fpath = join(RAW_DATA_DIR, '{}.pq'.format(self.name))
        self.data = read_dataframe(fpath)
        
        msg = '"m" column missing from data table'
        check_error('m' in self.data.columns, msg=msg)
        
        msg = '"var" column missing from data table'
        check_error('var' in self.data.columns, msg=msg)
    
    def load_landscape(self):
        fpath = join(LANDSCAPES_DIR, '{}.pq'.format(self.name))
        self.landscape = read_dataframe(fpath)
    
    def to_sequence_space(self):
        space = SequenceSpace(X=self.landscape.index.values,
                              y=self.landscape.iloc[:, 0].values)
        return(space)
