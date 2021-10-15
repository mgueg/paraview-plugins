# Plugin Author : Mikael Gueguen 2021
# reference : https://gitlab.kitware.com/paraview/paraview/-/blob/master/Examples/Plugins/PythonAlgorithm/PythonAlgorithmExamples.py
# https://kitware.github.io/paraview-docs/latest/python/paraview.util.html
"""
Plugin ThesholdSystemFilter

This module compute means by grains for FFT result 


:author: Mikael Gueguen 
"""
import vtk,os
import sys
from vtkmodules.vtkCommonDataModel import vtkDataSet
# This is module to import. It provides VTKPythonAlgorithmBase, the base class
# for all python-based vtkAlgorithm subclasses in VTK and decorators used to
# 'register' the algorithm with ParaView along with information about UI.
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase

# new module for ParaView-specific decorators.
# ref https://kitware.github.io/paraview-docs/latest/python/paraview.util.vtkAlgorithm.html#decorator-basics
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain, smhint


THIS_PATH = os.path.abspath(__file__)
THIS_DIR = os.path.dirname(THIS_PATH)
FILTER_NAME = os.path.basename(THIS_PATH)
sys.path.append(THIS_DIR)
import ed_fft_tools as edt



@smproxy.filter(label="Compute Mean Field by Grains Filter")
@smproperty.input(name="GrainsId", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)
@smproperty.input(name="FFTResult", port_index=1)
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)
class MeanByGrainsFilter(VTKPythonAlgorithmBase):
    # the rest of the code here is unchanged.
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=2, nOutputPorts=1,outputType='vtkImageData')
        self._weighted = False
        self._compute_volume = True
        
    # "StringInfo" and "String" demonstrate how one can add a selection widget
    # that lets user choose a string from the list of strings.
    @smproperty.stringvector(name="SetWeightedInfo", information_only="1")
    def GetStrings(self):
        return ["yes", "no"]

    @smproperty.stringvector(name="Set Weighted by Volume", number_of_elements="1")
    @smdomain.xml(\
        """<StringListDomain name="list">
                <RequiredProperties>
                    <Property name="SetWeightedInfo" function="SetWeightedInfo"/>
                </RequiredProperties>
            </StringListDomain>
        """)
    def SetString(self, value):
        #dummy = value
        if value=="yes":
            self._weighted = True
        if value=="no":
            self._weighted = False
        self.Modified()

    # @smproperty.stringvector(name="ComputeVolumeInfo", information_only="1")
    # def GetStrings(self):
    #     return ["yes", "no"]
    #
    # @smproperty.stringvector(name="Compute Volume", number_of_elements="1")
    # @smdomain.xml(\
    #     """<StringListDomain name="list">
    #             <RequiredProperties>
    #                 <Property name="ComputeVolumeInfo" function="ComputeVolumeInfo"/>
    #             </RequiredProperties>
    #         </StringListDomain>
    #     """)
    # def SetString(self, value):
    #     #dummy = value
    #     if value=="yes":
    #         self._compute_volume = True
    #     if value=="no":
    #         self._compute_volume = False
    #     self.Modified()

    # @smproperty.intvector(name="Weighted by volume", default_values=1)
    # def SetWeightedbyVolume(self, x):
    #     """Specify x"""
    #     if x!=1:
    #         self._weighted = True
    #     self.Modified()

    # def compute_volume(self, grain_index_field,vx_size=(1.,1.,1.)):
    #     """
    #     Compute volume grains.
    #
    #     Args:
    #         grain_index_field:      VTK field containing index
    #         vx_size=(1.,1.,1.):     the voxel size
    #
    #     Returns:
    #         volume_grains:  3D numpy array containing volume grains field
    #     """
    #
    #     import numpy as np
    #     from vtk.numpy_interface import algorithms as algs
    #
    #     real_indx_grains = np.unique(grain_index_field)
    #     volume_grains = np.zeros_like(grain_index_field)
    #     vx_vol = vx_size[0]*vx_size[1]*vx_size[2]
    #     for index in real_indx_grains:
    #         mask_grains = np.nonzero(grain_index_field==index)
    #         volume = np.count_nonzero(grain_index_field==index)*vx_vol
    #         volume_grains[mask_grains] = volume
    #
    #     return volume_grains
    #
    # def compute_mean_field(self, grain_index_field,field_data,field_name,vx_size=(1.,1.,1.)):
    #     """
    #     Compute mean field by grains (and standard deviation).
    #
    #     Args:
    #         grain_index_field:      VTK field containing index
    #         field_data:             VTK field containing  field ; should be defined on Point from the dataset
    #         field_name:   the requested name of field
    #         vx_size=(1.,1.,1.):     the voxel size
    #
    #     Returns:
    #         mean_field:     3D numpy array containing mean  field
    #         std_field:     3D numpy array containing std deviation  field
    #     """
    #
    #     import numpy as np
    #     from vtk.numpy_interface import algorithms as algs
    #
    #     real_indx_grains = np.unique(grain_index_field)
    #     field = field_data.PointData[field_name]
    #     mean_field = np.zeros_like(field)
    #     std_field = np.zeros_like(field)
    #     #volume_grains = np.zeros_like(grain_index_field)
    #     vx_vol = vx_size[0]*vx_size[1]*vx_size[2]
    #
    #     volume = 1.
    #     for index in real_indx_grains:
    #         mask_grains = np.nonzero(grain_index_field==index)
    #         if self._weighted:
    #             volume = np.count_nonzero(grain_index_field==index)*vx_vol
    #
    #         mean = algs.mean(field[mask_grains],axis=0)*volume
    #         std_dev = np.std(field[mask_grains],axis=0)*volume
    #         mean_field[mask_grains] = mean
    #         std_field[mask_grains] = std_dev
    #
    #     return mean_field, std_field
       
    def RequestInformation(self, request, inInfoVec, outInfoVec):
        """
        
        """
        
        executive = self.GetExecutive()
        outInfo = outInfoVec.GetInformationObject(0)
        
        inInfo1 = inInfoVec[0].GetInformationObject(0)
        inInfo2 = inInfoVec[1].GetInformationObject(0)
        
        timesteps1 = inInfo1.Get(executive.TIME_STEPS()) \
                               if inInfo1.Has(executive.TIME_STEPS()) else None
        timesteps2 = inInfo2.Get(executive.TIME_STEPS()) \
                               if inInfo2.Has(executive.TIME_STEPS()) else None
        
        outInfo.Remove(executive.TIME_STEPS())
        outInfo.Remove(executive.TIME_RANGE())

        if timesteps2 is not None:
            #print("time steps def to be imported from data",timesteps2)
            outInfo.Set(executive.TIME_STEPS(), timesteps2, len(timesteps2))
            outInfo.Set(executive.TIME_RANGE(), [timesteps2[0],timesteps2[-1]], 2)
        
        return 1


    def RequestData(self, request, inInfoVec, outInfo):
        """
            Entry Point to Filter (eg used with `Apply` button).
            The script gets executed in what's called the RequestData pass of the pipeline execution. 
            This is the pipeline pass in which an algorithm is expected to produce the output dataset.
        
        """
        
        from vtkmodules.numpy_interface import dataset_adapter as dsa
        from vtk.numpy_interface import algorithms as algs
        import vtk.util.numpy_support as ns
        import numpy as np

        grains_data  = dsa.WrapDataObject(vtkDataSet.GetData(inInfoVec[0], 0)) #inputs[0]
        field_data = dsa.WrapDataObject(vtkDataSet.GetData(inInfoVec[1], 0)) #inputs[1]
        output = dsa.WrapDataObject(vtkDataSet.GetData(outInfo))
        output.CopyStructure(vtkDataSet.GetData(inInfoVec[0]))
        
        dim_grains_data = grains_data.GetDimensions()
        vx_size = grains_data.GetSpacing()
        vx_vol = vx_size[0]*vx_size[1]*vx_size[2]

        if 'Index' in grains_data.PointData.keys():
            grains_index = grains_data.PointData["Index"]
        elif 'FeatureIds' in grains_data.PointData.keys():
            grains_index = grains_data.PointData["FeatureIds"]
        else:
            raise RuntimeError("keys 'Index', 'FeatureIds' is not found in PointData.keys()")
            return 1

        if self._compute_volume:
            volume = edt.compute_volume(grains_index,vx_size=vx_size)
            output.PointData.append(volume,"Volume")
        
        output.FieldData.append(field_data.FieldData['TIME'],"TIME")

        for field_name in field_data.PointData.keys():
            field = field_data.PointData[field_name]

            mean_field = np.zeros_like(field)
            std_field = np.zeros_like(field)
    
            _, mean_field, std_field = edt.compute_mean_field(grains_index,field_data,
                                                        field_name=field_name,
                                                        vx_size=vx_size,
                                                        weighted=self._weighted,
                                                        compute_std_dev=True)
            
            output.PointData.append(mean_field,"<{}>".format(field_name))
            output.PointData.append(std_field,"<<{}>>".format(field_name))

        return 1
        