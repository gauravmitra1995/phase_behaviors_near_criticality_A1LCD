__author__ = 'Furqan Dar'
__version__ = '0.0.2'

import numpy as np
import sys

_INT_SIZE = 4
_BYTE_ORDER = sys.byteorder

class TrjReader(object):
    """
    General purpose trajectory reader for binary LaSSI trajectories. The trajectories are formatted as such:
    
    N,X_1,Y_1,Z_1,F_1,...,X_N,Y_N,Z_N,F_N,
    
    where in each frame the first int corresponds to the total number of atoms, and then every four ints corresponds to
    a bead where X, Y, Z are coordinates, and F is the beadPartner.
    
    """
    
    def __init__(self, file_path: str = "", num_atoms: int = None, num_frames: int = None):
        """
        Can be used to instantiate the class. If we know both the number of
        atoms per frame, and the total number of frames, we have enough information
        to extract any frame.
        """
        self._f_path = file_path
        if (num_atoms is None) or (num_frames is None):
            self._n_atoms, self._n_frms = self._get_atoms_and_frames(file_path=self._f_path)
        else:
            self._n_atoms = num_atoms
            self._n_frms  = num_frames
    
    @staticmethod
    def _get_atoms_and_frames(file_path:str = None):
        """
        Get the total number of atoms per frame, and number of frames for this trajectory.
        :param file_path:
        :return:
        """
        with open(file_path, 'rb') as tF:
            num_atoms = tF.read(1 * _INT_SIZE)
            num_atoms = int.from_bytes(num_atoms, byteorder=_BYTE_ORDER)
            frm_size = num_atoms * 4 * _INT_SIZE
            tF.seek(0)
            num_frames = len(tF.read()) // (frm_size + _INT_SIZE)
            
        return num_atoms, num_frames
    
    def __str__(self):
        """
        String representation of this particular trajectory.
        :return:
        'Num of atoms  :
         Num of frames :
         File Path     :
         '
        """
        
        line1 = f'Num of atoms  : {self._n_atoms:<10}'
        line2 = f'Num of frames : {self._n_frms:<10}'
        line3 = f'File Path     : {self._f_path}'
        
        all_lines = [line1, line2, line3]
        return "\n".join(all_lines)
    
    def __eq__(self, other):
        """
        If the trajectories have the same paths,they are they same.
        :param other:
        :return:
        """
        
        return self._f_path == other._f_path
    
    def __repr__(self):
        """
        A way to instantiate this current trajectory
        :return:
        """
        
        return f'tp.TrjReader(file_path={self._f_path}, num_atoms={self._n_atoms}, num_frames={self._n_frms})'

class TrjExtractor(TrjReader):
    """
    General purpose class that has methods for extracting frames from a particular binary trajectory.
    """
    
    def __init__(self, file_path: str = "", num_atoms: int = None, num_frames: int = None, frames_list: list = None):
        """
        :param file_path:
        :param num_atoms:
        :param num_frames:
        :param frames_list:
        :return:
        """
        super().__init__(file_path=file_path, num_atoms=num_atoms, num_frames=num_frames)
        
        if frames_list is None:
            self._frm_list = np.arange(self._n_frms)
        else:
            self._frm_list = frames_list
            
    def __str__(self):
        """
        
        :return:
        """
        # line0 = f'Num of atoms  : {self._n_atoms:<10}'
        line1 = f"Frame List    : {str(list(self._frm_list))}"
        
        return f'{super().__str__()}\n{line1}'
    
    def __repr__(self):
        """
        
        :return:
        """
        return f'{super().__repr__()[:-1]}, frames_list={str(list(self._frm_list))}'
    
    @staticmethod
    def get_frm_idx(frame_id: int = 0, num_atoms: int = None):
        """
        
        :param frame_id:
        :param num_atoms:
        :return:
        """
        
        frmSize = num_atoms * 4 * _INT_SIZE + 1 * _INT_SIZE
        
        start_idx = frame_id * (frmSize)
        end_idx   = (frame_id+1) * (frmSize)
        return start_idx, end_idx
    
    def extract_frames(self):
        """
        Given the frames_list, we extract each of those frames.
        :return:
        """
        all_frames = np.zeros(( len(self._frm_list), self._n_atoms, 4), dtype=f'i{_INT_SIZE}')
        
        with open(self._f_path, 'rb') as tF:
            fullBuff = tF.read()
            for frmID in self._frm_list:
                frmStart, frmEnd = self.get_frm_idx(frame_id=frmID, num_atoms=self._n_atoms)
                thisFrm = np.frombuffer(fullBuff[frmStart + _INT_SIZE: frmEnd], dtype=f'i{_INT_SIZE}')
                all_frames[frmID] = thisFrm.reshape((self._n_atoms, 4))
        return all_frames
    
    def extract_specific_frame(self,frmID):
        """
        Given the frames_list, we extract each of those frames.
        :return:
        """
        #all_frames = np.zeros(( len(self._frm_list), self._n_atoms, 4), dtype=f'i{_INT_SIZE}')
        
        with open(self._f_path, 'rb') as tF:
            
            
            frmStart, frmEnd = self.get_frm_idx(frame_id=frmID, num_atoms=self._n_atoms)
            
            tF.seek(frmStart+_INT_SIZE)
            fullBuff = tF.read(frmEnd-frmStart-_INT_SIZE)
            thisFrm = np.frombuffer(fullBuff, dtype=f'i{_INT_SIZE}')
            size = len(thisFrm)
            thisFrm = thisFrm.reshape((size//4, 4))
        return thisFrm[:,:-1]

    def extract_coords(self):
        """
        Given the frames_list, we extract only the coordinates from each frame.
        :return:
        """
        all_frames = np.zeros((len(self._frm_list), self._n_atoms, 4), dtype=f'i{_INT_SIZE}')
    
        with open(self._f_path, 'rb') as tF:
            fullBuff = tF.read()
            for frmID in self._frm_list:
                frmStart, frmEnd = self.get_frm_idx(frame_id=frmID, num_atoms=self._n_atoms)
                thisFrm = np.frombuffer(fullBuff[frmStart + _INT_SIZE: frmEnd], dtype=f'i{_INT_SIZE}')
                all_frames[frmID] = thisFrm.reshape((self._n_atoms, 4))
        return all_frames[:, :, :-1].copy()


