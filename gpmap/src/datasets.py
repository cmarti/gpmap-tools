from os.path import join, exists

from gpmap.src.space import SequenceSpace
from gpmap.src.utils import check_error, read_dataframe
from gpmap.src.settings import (DATASETS, RAW_DATA_DIR, LANDSCAPES_DIR,
                                PROCESSED_DIR)


class DataSet(object):
    def __init__(self, dataset):
        msg = 'Dataset not recognized: try one of {}'.format(DATASETS)
        check_error(dataset in DATASETS, msg=msg)
        
        self.name = dataset
        self.load_data()
    
    def load_data(self):
        fpath = join(PROCESSED_DIR, '{}.pq'.format(self.name))
        self.data = read_dataframe(fpath)
        
        if not 'X' in self.data.columns:
            msg = '"m" column missing from data table'
            check_error('m' in self.data.columns, msg=msg)
            
            msg = '"var" column missing from data table'
            check_error('var' in self.data.columns, msg=msg)
    
    def _load(self, fdir, label):
        fpath = join(fdir, '{}.pq'.format(self.name))
        
        if not exists(fpath):
            msg = '{} for dataset {} not found'.format(label, self.name)
            raise ValueError(msg)
        
        return(read_dataframe(fpath))
    
    @property
    def landscape(self):
        if not hasattr(self, '_landscape'):
            self._landscape = self._load(fdir=LANDSCAPES_DIR,
                                         label='estimated landscape')
        return(self._landscape)
    
    @property
    def raw_data(self):
        if not hasattr(self, '_raw_data'):
            self._raw_data = self._load(fdir=RAW_DATA_DIR,
                                         label='estimated landscape')
        return(self._raw_data)
    
    def to_sequence_space(self):
        space = SequenceSpace(X=self.landscape.index.values,
                              y=self.landscape.iloc[:, 0].values)
        return(space)
