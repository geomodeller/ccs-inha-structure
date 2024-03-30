## this workflow is to create multiple subsurface models that can be 
## used in CCS simulation

## Version 0.0.1
## Author: Honggeun Jo

## Import packages that will be used in this demo
import os                                                 # to set current working directory 
import numpy as np                                        # arrays and matrix math
import pandas as pd                                       # DataFrames
import matplotlib.pyplot as plt                           # plotting
import geostatspy.geostats as geostats
import geostatspy.GSLIB as GSLIB
import datetime
from visualize_3d_model import *
from scipy.interpolate import interp1d
from utils.check_var_type import check_var_type
from utils.check_param_element import check_param_element
from utils.gslib_wrapper import create_sgs_model, sgs_realizations, create_sis_model, sis_realizations


## 
class GeoModelling():
	version = '0.0.1'

	def __init__(self, data=None, params=None):
		pass

	def addData(self, data):
		assert check_var_type(data, [type(None), pd.DataFrame]),\
				"input data should be either pd.DataFrame or None"
		self.data = data
		return self
	
	def addParams(self, params):	
		assert check_var_type(params, [type(None), dict]),\
				"params should be either dictionary or None"
		if (params == None):
			self.param = params
		else:
			check_param_element(params, ['X','Y','Z','Var'])
			self.data = params
			self.param = params
			
	def __version__(self):
		print(f'{self.version}')

	def normal_score_transform(self,var_names):
		if type(var_names) == str: 
			assert var_names in self.data.columns, \
				f'No {var_names} is found in columns'
			self.data['N' + var_names], _, _ = geostats.nscore(self, var_names)
		if type(var_names) == list:
			for var_name in var_names:
				assert var_name in self.data.columns, \
					f'No {var_name} is found in columns'	
				self.data['N' + var_name], _, _ = geostats.nscore(self, var_name)

if __name__ == '__main__':  
	## Load dataset (well data)

	df = pd.read_csv("1_sample_data.csv")      
	df.head()
	plt.figure()
	plt.show()