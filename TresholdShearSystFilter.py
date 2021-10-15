# Plugin Author : Mikael Gueguen 2021
# reference : https://gitlab.kitware.com/paraview/paraview/-/blob/master/Examples/Plugins/PythonAlgorithm/PythonAlgorithmExamples.py
# https://kitware.github.io/paraview-docs/latest/python/paraview.util.html
"""
Plugin ThesholdSystemFilter

This module treshold shear system for FFT result 


:author: Mikael Gueguen 
"""

import vtk, os, sys
from vtkmodules.vtkCommonDataModel import vtkDataSet
# This is module to import. It provides VTKPythonAlgorithmBase, the base class
# for all python-based vtkAlgorithm subclasses in VTK and decorators used to
# 'register' the algorithm with ParaView along with information about UI.
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase

# new module for ParaView-specific decorators.
# ref https://kitware.github.io/paraview-docs/latest/python/paraview.util.vtkAlgorithm.html#decorator-basics
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain, smhint

# 
THIS_PATH = os.path.abspath(__file__)
THIS_DIR = os.path.dirname(THIS_PATH)
FILTER_NAME = os.path.basename(THIS_PATH)

sys.path.append(THIS_DIR)
import ed_fft_tools as edt




@smproxy.filter(label="Treshold Shear Systems Filter")
@smproperty.input(name="GrainsId", port_index=0)
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)
@smproperty.input(name="FFTResult", port_index=1)
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)
class TresholdSSFilter(VTKPythonAlgorithmBase):
    # the rest of the code here is unchanged.
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=2, nOutputPorts=1,outputType='vtkImageData')
        self._shear_treshold = None
        self._filename = '/Users/mik/Documents/dpmm/codes/foxtrot/fx_fft/fftbuild/tests/evp.mat'
        self._save = False
        self._compute_volume = True

    @smproperty.doublevector(name="ShearTreshold", default_values=1e-5)
    @smdomain.doublerange(min=1e-6,max=1e-1)
    def SetShearTreshold(self, x):
        """Specify x for shear treshold"""
        self._shear_treshold = x
        self.Modified()

    # "StringInfo" and "String" demonstrate how one can add a selection widget
    # that lets user choose a string from the list of strings.
    @smproperty.stringvector(name="SaveDataInfo", information_only="1")
    def GetStrings(self):
        return ["yes", "no"]

    @smproperty.stringvector(name="Save Data", number_of_elements="1")
    @smdomain.xml(\
        """<StringListDomain name="list">
                <RequiredProperties>
                    <Property name="SaveDataInfo" function="SaveDataInfo"/>
                </RequiredProperties>
            </StringListDomain>
        """)
    def SetString(self, value):
        #dummy = value
        if value=="yes":
            self._save = True
        if value=="no":
            self._save = False
        self.Modified()
        
    # @smproperty.stringvector(name="Compute Volume", number_of_elements="1")
    # @smdomain.xml(\
    #     """<StringListDomain name="list">
    #             <RequiredProperties>
    #                 <Property name="SetWeightedInfo" function="SetWeightedInfo"/>
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
   
    # def compute_mean_field(self, grain_index_field,field_data,field_name='Gamma',vx_size=(1.,1.,1.)):
    #     """
    #     Compute mean shear system by grains.
    #
    #     Args:
    #         grain_index_field:      VTK field containing index
    #         field_data:             VTK field containing shear field
    #         field_name=('Gamma'):   the requested name of field
    #         vx_size=(1.,1.,1.):     the voxel size
    #
    #     Returns:
    #         gamma_by_grain: 2D numpy array with every mean shear for each grains
    #         mean_field:     3D numpy array containing mean shear field
    #         volume_grains:  3D numpy array containing volume grains field
    #     """
    #
    #     import numpy as np
    #     from vtk.numpy_interface import algorithms as algs
    #
    #     real_indx_grains = np.unique(grain_index_field)
    #     gamma_field = field_data.PointData[field_name]
    #     mean_field = np.zeros_like(gamma_field)
    #     volume_grains = np.zeros_like(grain_index_field)
    #     vx_vol = vx_size[0]*vx_size[1]*vx_size[2]
    #     #gamma_by_grain = []
    #     for index in real_indx_grains:
    #         mask_grains = np.nonzero(grain_index_field==index)
    #         volume = np.count_nonzero(grain_index_field==index)*vx_vol
    #         #if index%100==0:
    #         #    print("index -> ",index)
    #         volume_grains[mask_grains] = volume
    #         mean = algs.mean(gamma_field[mask_grains],axis=0)
    #         #gamma_by_grain.append(mean)
    #         mean_field[mask_grains] = mean
    #
    #     #gamma_by_grain = np.row_stack(gamma_by_grain)
    #     gamma_by_grain = np.unique(mean_field,axis=0)
    #     #print(" gamma_by_grain ", gamma_by_grain.shape)
    #     #mean_by_grains = np.column_stack((real_indx_grains,gamma_by_grain))
    #
    #     return gamma_by_grain,mean_field,volume_grains

    @smproperty.stringvector(name="FileName",default_values='/Users/mik/Documents/dpmm/codes/foxtrot/fx_fft/fftbuild/tests/evp.mat')
    @smdomain.filelist()
    @smhint.filechooser(extensions="json,mat", file_description="json mat files")
    def SetFileName(self, name):
        """Specify filename for the file to read."""
        if self._filename != name:
            self._filename = name
            self.Modified()

    def RequestInformation(self, request, inInfoVec, outInfoVec):
        from vtkmodules.numpy_interface import dataset_adapter as dsa
        
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
        import json
        import os
        global LIST_TYPES, LIST_TYPES_INDEXED #, FILTER_NAME

        def get_time(inputfield):
            import vtk.util.numpy_support as ns
    
            time_from_ds = inputfield.FieldData['TIME']
            #time = GetUpdateTimestep(self)
            #t_from_time_evol = inputfield.GetInformation().Get(vtk.vtkDataObject.DATA_TIME_STEP())
            #print("t : ",time_from_ds, " t ",t_from_time_evol )
            time_array = ns.vtk_to_numpy(time_from_ds)
            #print("t : ",time_array)
            return time_array
        
        # with open(self._filename,'r') as fp:
        #      model_mat = json.load(fp)
        # crss0_values = np.array(model_mat["materials"]["alpha"]["crss0"])
        #
        # types = 3*'basal ' + 3*'prism ' + 12*'pyram '
        # list_types = types.split()
        # basal = ['{1}_{0}'.format(i,n) for i,n in enumerate(list_types[:3])]
        # prism = ['{1}_{0}'.format(i,n) for i,n in enumerate(list_types[4:7])]
        # pyram = ['{1}_{0}'.format(i,n) for i,n in enumerate(list_types[8:])]
        # list_types_indexed = basal + prism + pyram

        grains_data  = dsa.WrapDataObject(vtkDataSet.GetData(inInfoVec[0], 0)) #inputs[0]
        field_data = dsa.WrapDataObject(vtkDataSet.GetData(inInfoVec[1], 0)) #inputs[1]
        output = dsa.WrapDataObject(vtkDataSet.GetData(outInfo))
        output.CopyStructure(vtkDataSet.GetData(inInfoVec[0]))
        
        dim_grains_data = grains_data.GetDimensions()
        vx_size = grains_data.GetSpacing()
        vx_vol = vx_size[0]*vx_size[1]*vx_size[2]

        time = get_time(field_data)
        if not isinstance(time, np.ndarray):
            time = np.array([time])

        if 'Index' in grains_data.PointData.keys():
            grains_index = grains_data.PointData["Index"]
        elif 'FeatureIds' in grains_data.PointData.keys():
            grains_index = grains_data.PointData["FeatureIds"]
        else:
            raise RuntimeError("keys 'Index', 'FeatureIds' is not found in keys()")
            return 1

        #grains_index = grains_data.PointData["Index"]
        real_indx_grains = np.unique(grains_index)
        volume = edt.compute_volume(grains_index,vx_size=vx_size)

        gamma_by_grain, mean_field, _ = edt.compute_mean_field(grains_index,field_data,\
                                                                 field_name="Gamma",vx_size=vx_size,
                                                                 weighted=False, compute_std_dev=False)
        datas = edt.treshold_field(self._shear_treshold,gamma_by_grain)
        
        if datas:
            unique_grains, counts_shear_grains, syst_activated = datas
            #print("grains activated for T = {0} :: {1} ".format(self._shear_treshold,real_indx_grains[unique_grains]))
            #print("nb sys activated for T = {0} :: {1} ".format(self._shear_treshold,counts_shear_grains))
            nb_act_allgrains = np.zeros_like(grains_index)

            for index,counts in zip(unique_grains,counts_shear_grains):
                mask_grains = np.nonzero(grains_index==real_indx_grains[index])
                nb_act_allgrains[mask_grains] = counts

            activ_field = np.zeros_like(grains_index)
            for index in unique_grains:
                mask_grains = np.nonzero(grains_index==real_indx_grains[index])
                activ_field[mask_grains] = 1
        
        
            #print("syst_activated ",syst_activated)
            #print("Create data with activated systems : ")
            #print("-------------------------------------")
            for shear_syst in  range(1,syst_activated.shape[1]):
                syst_activated_indx = -1*np.ones_like(grains_index)
                for gr,indx in zip(syst_activated[:,0],syst_activated[:,shear_syst]):
                    mask_grains = np.nonzero(grains_index==real_indx_grains[gr])
                    syst_activated_indx[mask_grains] = indx
                output.PointData.append(syst_activated_indx,"{0}st_syst".format(shear_syst))

            output.PointData.append(nb_act_allgrains,"Nbsyst")
            output.PointData.append(activ_field,"ActiveGrains")
            output.PointData.append(mean_field,"Mean Gamma")
            output.PointData.append(volume,"Volume")
            output.FieldData.append(field_data.FieldData['TIME'],"TIME")
        
            if self._save:
                time = np.round(time,np.ceil(np.log10(time)).astype('int')[0])
                #numdigits = np.ceil(np.log10(time[0])))
                t = str(time[0]).replace('.','p')
                workdir = os.path.dirname(self._filename)
                os.chdir(workdir)
            
                print("Save array with activated systems in dir {}".format(os.getcwd()))
                print("------------------------------------------------")
                comments = " Shear systems : {0} \n".format(np.transpose(edt.LIST_TYPES_INDEXED))
                comments = comments + " treshold = {} ; TIME = {}\n".format(self._shear_treshold, time)
                comments = comments + " number of grains = {} ; size of data = {}\n".format(real_indx_grains.shape,\
                                                                                         syst_activated.shape)
            
                header = comments + " Written by {}\n# grain_number    shear_syst_1st     shear_sys_2nd    ....".format(FILTER_NAME)
                np.savetxt("systems_activated_by_index_t{}.rpt".format(t),syst_activated,  fmt='%i', header = header)
                #np.save("systems_activated_by_index",syst_activated)
                        
        else:
            output.PointData.append(mean_field,"Mean Gamma")
            output.PointData.append(volume,"Volume")
            output.FieldData.append(field_data.FieldData['TIME'],"TIME")
            print(' Do nothing ... : no grains found with treshold set {}'.format(self._shear_treshold))

        return 1
        
