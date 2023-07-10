from os.path import join, exists

from gpmap.src.utils import check_error, read_dataframe
from gpmap.src.settings import DATASETS, RAW_DATA_DIR, LANDSCAPES_DIR
from gpmap.src.space import SequenceSpace


class DataSet(object):
    def __init__(self, dataset):
        msg = 'Dataset not recognized: try one of {}'.format(DATASETS)
        check_error(dataset in DATASETS, msg=msg)
        
        self.name = dataset
        self.load_data()
    
    def load_data(self):
        fpath = join(RAW_DATA_DIR, '{}.pq'.format(self.name))
        self.data = read_dataframe(fpath)
        
        if not 'X' in self.data.columns:
            msg = '"m" column missing from data table'
            check_error('m' in self.data.columns, msg=msg)
            
            msg = '"var" column missing from data table'
            check_error('var' in self.data.columns, msg=msg)
    
    @property
    def landscape(self):
        if not hasattr(self, '_landscape'):
            self.load_landscape()
        return(self._landscape)
    
    def load_landscape(self):
        fpath = join(LANDSCAPES_DIR, '{}.pq'.format(self.name))
        
        if not exists(fpath):
            msg = 'estimated landscape for dataset {} not found'.format(self.name)
            raise ValueError(msg)
        
        self._landscape = read_dataframe(fpath)
    
    def to_sequence_space(self):
        space = SequenceSpace(X=self.landscape.index.values,
                              y=self.landscape.iloc[:, 0].values)
        return(space)
