import random
import numpy as np
import pandas as pd
import os

PATH_TO_SAVE = '../data/regularizer_limit.csv'

class LimitTracker():

    def __init__(
        self,
        low_decor=0,
        high_decor=1e5,
        low_n=0,
        high_n=30,
        low_back=0,
        high_back=5,
        low_spb=0,
        high_spb=1e2,
        low_spm=-1e-3,
        high_spm=1e2,
        low_sp_phi=-1e3,
        high_sp_phi=1e3,
        low_prob=0,
        high_prob=1,
        update_rate=0.25
    ):
        self.high_decor = high_decor
        self.low_decor = low_decor
        self.low_n = low_n
        self.high_n = high_n
        self.low_back = low_back
        self.high_back = high_back
        self.high_spb = high_spb
        self.low_spb = low_spb
        self.low_spm = low_spm
        self.high_spm = high_spm
        self.low_sp_phi = low_sp_phi
        self.high_sp_phi = high_sp_phi
        self.low_prob = low_prob
        self.high_prob = high_prob
        
        self.update_rate = update_rate
        self.params = {}
        self._init_tracker()

    def _init_tracker(self):
        self.params['low_decor'] = []
        self.params['high_decor'] = []
        self.params['low_n'] = []
        self.params['high_n'] = []
        self.params['low_back'] = []
        self.params['high_back'] = []
        self.params['low_spb'] = []
        self.params['high_spb'] = []
        self.params['low_spm'] = []
        self.params['high_spm'] = []
        self.params['low_sp_phi'] = []
        self.params['high_sp_phi'] = []
        self.params['low_prob'] = []
        self.params['high_prob'] = []
        self.params['fitness'] = []

    def log_limits(self, **kwargs):
        self.params['low_decor'].append(kwargs.get('low_decor'))
        self.params['high_decor'].append(kwargs.get('high_decor'))
        self.params['low_n'].append(kwargs.get('low_n'))
        self.params['high_n'].append(kwargs.get('high_n'))
        self.params['low_back'].append(kwargs.get('low_back'))
        self.params['high_back'].append(kwargs.get('high_back'))
        self.params['low_spb'].append(kwargs.get('low_spb'))
        self.params['high_spb'].append(kwargs.get('high_spb'))
        self.params['low_spm'].append(kwargs.get('low_spm'))
        self.params['high_spm'].append(kwargs.get('high_spm'))
        self.params['low_sp_phi'].append(kwargs.get('low_sp_phi'))
        self.params['high_sp_phi'].append(kwargs.get('high_sp_phi'))
        self.params['low_prob'].append(kwargs.get('low_prob'))
        self.params['high_prob'].append(kwargs.get('high_prob'))
        self.params['fitness'].append(kwargs.get('fitness'))

        print('limits: ', self.params)

    def get_limits(self):
        limits = {
            'high_decor' : self.high_decor,
            'low_decor' : self.low_decor,
            'low_n' : self.low_n,
            'high_n' : self.high_n,
            'low_back' : self.low_back,
            'high_back' : self.high_back,
            'high_spb' : self.high_spb,
            'low_spb' : self.low_spb,
            'low_spm' : self.low_spm,
            'high_spm' : self.high_spm,
            'low_sp_phi' : self.low_sp_phi,
            'high_sp_phi' : self.high_sp_phi,
            'low_prob' : self.low_prob,
            'high_prob' : self.high_prob,
        }

        return limits
    
    def update_limit(self):        

        # get correlation
        df = pd.read_csv(PATH_TO_SAVE)
        corr = df.corr().iloc[:-1, -1].fillna(0).to_dict()

        if corr['low_decor'] != 0.0:
            self.low_decor = self.low_decor - (self.low_decor * self.update_rate * corr['low_decor']) if corr['low_decor'] < 0 else self.low_decor + (self.low_decor * self.update_rate * corr['low_decor'])
        
        if corr['high_decor'] != 0.0:
            self.high_decor = self.high_decor - (self.high_decor * self.update_rate * corr['high_decor']) if corr['high_decor'] < 0 else self.high_decor + (self.high_decor * self.update_rate * corr['high_decor'])
        
        if corr['low_n'] != 0.0:
            self.low_n = self.low_n - (self.low_n * self.update_rate * corr['low_n']) if corr['low_n'] < 0 else self.low_n + (self.low_n * self.update_rate * corr['low_n'])
        
        if corr['high_n'] != 0.0:
            self.high_n = self.high_n - (self.high_n * self.update_rate * corr['high_n']) if corr['high_n'] < 0 else self.high_n + (self.high_n * self.update_rate * corr['high_n'])
        
        if corr['low_back'] != 0.0:
            self.low_back = self.low_back - (self.low_back * self.update_rate * corr['low_back']) if corr['low_back'] < 0 else self.low_back + (self.low_back * self.update_rate * corr['low_back'])
        
        if corr['high_back'] != 0.0:
            self.high_back = self.high_back - (self.high_back * self.update_rate * corr['high_back']) if corr['high_back'] < 0 else self.high_back + (self.high_back * self.update_rate * corr['high_back'])
        
        if corr['low_spb'] != 0.0:
            self.low_spb = self.low_spb - (self.low_spb * self.update_rate * corr['low_spb']) if corr['low_spb'] < 0 else self.low_spb + (self.low_spb * self.update_rate * corr['low_spb'])
        
        if corr['high_spb'] != 0.0:
            self.high_spb = self.high_spb - (self.high_spb * self.update_rate * corr['high_spb']) if corr['high_spb'] < 0 else self.high_spb + (self.high_spb * self.update_rate * corr['high_spb'])
        
        if corr['low_spm'] != 0.0:
            self.low_spm = self.low_spm - (self.low_spm * self.update_rate * corr['low_spm']) if corr['low_spm'] < 0 else self.low_spm + (self.low_spm * self.update_rate * corr['low_spm'])
        
        if corr['high_spm'] != 0.0:
            self.high_spm = self.high_spm - (self.high_spm * self.update_rate * corr['high_spm']) if corr['high_spm'] < 0 else self.high_spm + (self.high_spm * self.update_rate * corr['high_spm'])
        
        if corr['low_sp_phi'] != 0.0:
            self.low_sp_phi = self.low_sp_phi - (self.low_sp_phi * self.update_rate * corr['low_sp_phi']) if corr['low_sp_phi'] < 0 else self.low_sp_phi + (self.low_sp_phi * self.update_rate * corr['low_sp_phi'])
        
        if corr['high_sp_phi'] != 0.0:
            self.high_sp_phi = self.high_sp_phi - (self.high_sp_phi * self.update_rate * corr['high_sp_phi']) if corr['high_sp_phi'] < 0 else self.high_sp_phi + (self.high_sp_phi * self.update_rate * corr['high_sp_phi'])
        
        if corr['low_prob'] != 0.0:
            self.low_prob = self.low_prob - (self.low_prob * self.update_rate * corr['low_prob']) if corr['low_prob'] < 0 else self.low_prob + (self.low_prob * self.update_rate * corr['low_prob'])
        
        if corr['high_prob'] != 0.0:
            self.high_prob = self.high_prob - (self.high_prob * self.update_rate * corr['high_prob']) if corr['high_prob'] < 0 else self.high_prob + (self.high_prob * self.update_rate * corr['high_prob'])
        
        

        # if len(self.params['fitness']) <= 1:
        #     self._update()
        # else:
        #     reverse = True if self.params['fitness'][-1] < self.params['fitness'][-2] else False
        #     self._update(reverse)

        new_limits = self.get_limits()
        return new_limits

    def _update(self, reverse=False):
        if random.random() > 0.5:
            self.high_decor = self.high_decor - (self.high_decor * self.update_rate) if not reverse else self.high_decor + (self.high_decor * self.update_rate)
            self.high_spb = self.high_spb - (self.high_spb * self.update_rate) if not reverse else self.high_spb + (self.high_spb * self.update_rate)
            self.high_spm = self.high_spm - (self.high_spm * self.update_rate) if not reverse else self.high_spm + (self.high_spm * self.update_rate)
            self.high_sp_phi = self.high_sp_phi - (self.high_sp_phi * self.update_rate) if not reverse else self.high_sp_phi + (self.high_sp_phi * self.update_rate)
            self.high_prob = self.high_prob - (self.high_prob * self.update_rate) if not reverse else self.high_prob + (self.high_prob * self.update_rate)
        else:
            self.low_decor = self.low_decor + (self.low_decor * self.update_rate) if not reverse else self.low_decor - (self.low_decor * self.update_rate)
            self.low_spb = self.low_spb + (self.low_spb * self.update_rate) if not reverse else self.low_spb - (self.low_spb * self.update_rate)
            self.low_spm = self.low_spm + (self.low_spm * self.update_rate) if not reverse else self.low_spm - (self.low_spm * self.update_rate)
            self.low_sp_phi = self.low_sp_phi + (self.low_sp_phi * self.update_rate) if not reverse else self.low_sp_phi - (self.low_sp_phi * self.update_rate)
            self.low_prob = self.low_prob + (self.low_prob * self.update_rate) if not reverse else self.low_prob - (self.low_prob * self.update_rate)
            

        # if random.random() > 0.5:
        # else:

        # if random.random() > 0.5:
        # else:

        # if random.random() > 0.5:
        # else:

        # if random.random() > 0.5:
        # else:

        # if random.random() > 0.5:
        # else:

    def save_limits(self):
        if os.path.exists(PATH_TO_SAVE):
            df = pd.read_csv(PATH_TO_SAVE)

        df_params = pd.DataFrame.from_dict(self.params)

        try:
            df = pd.concat([df, df_params], axis=0)
        except Exception:
            df = df_params

        df.to_csv(PATH_TO_SAVE, index=False)