# Plugin Author : Mikael Gueguen 2021
# reference : https://gitlab.kitware.com/paraview/paraview/-/blob/master/Examples/Plugins/PythonAlgorithm/PythonAlgorithmExamples.py
# https://kitware.github.io/paraview-docs/latest/python/paraview.util.html
"""
Plugin SelectMacrozoneFilter

This module compute select a region a vtkImageData for pre-processing FFT
computation


:author: Mikael Gueguen
"""
import vtk, os
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


@smproxy.filter(label="Select MacroZone Filter")
@smproperty.input(name="GrainsId")
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)
class SelectMacrozoneFilter(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=1, nOutputPorts=1, outputType="vtkImageData"
        )
        # TODO from vtkmodules.vtkFiltersSources import vtkSphereSource
        # TODO self._realAlgorithm = vtkSphereSource()
        self._center = [100, 100, 100]
        self._radius = 10

    @smproperty.doublevector(name="Center", default_values=[100, 100, 100])
    @smdomain.doublerange()
    def SetCenter(self, x, y, z):
        self._center = (x, y, z)
        # self._realAlgorithm.SetCenter(x,y,z)
        self.Modified()

    @smproperty.doublevector(name="Radius", default_values=10)
    @smdomain.doublerange()
    def SetRadius(self, x):
        self._radius = x
        # self._realAlgorithm.SetPhiResolution(x)
        self.Modified()

    # def RequestInformation(self, request, inInfoVec, outInfoVec):
    #    """ """

    def RequestData(self, request, inInfoVec, outInfo):
        """
        Entry Point to Filter (eg used with `Apply` button).
        The script gets executed in what's called the RequestData pass
        of the pipeline execution.
        This is the pipeline pass in which an algorithm is expected to
        produce the output dataset.

        """

        from vtkmodules.numpy_interface import dataset_adapter as dsa
        from vtk.numpy_interface import algorithms as algs
        import vtk.util.numpy_support as ns
        import numpy as np

        grains_data = dsa.WrapDataObject(
            vtkDataSet.GetData(inInfoVec[0], 0)
        )  # inputs[0]
        output = dsa.WrapDataObject(vtkDataSet.GetData(outInfo))
        output.CopyStructure(vtkDataSet.GetData(inInfoVec[0]))

        dim_grains_data = grains_data.GetDimensions()
        vx_size = grains_data.GetSpacing()
        # vx_vol = vx_size[0] * vx_size[1] * vx_size[2]
        has_phase = False
        if "Phase" in grains_data.PointData.keys():
            grains_phase = grains_data.PointData["Phase"]
            has_phase = True

        if "Index" in grains_data.PointData.keys():
            grains_index = grains_data.PointData["Index"]
        elif "FeatureIds" in grains_data.PointData.keys():
            grains_index = grains_data.PointData["FeatureIds"]
        else:
            raise RuntimeError(
                "keys 'Index', 'FeatureIds' is not found in PointData.keys()"
            )
            return 1

        masked_object = edt.mask_sphere(
            dim_grains_data, center=self._center, r=self._radius
        )
        masked_object_flat = masked_object.flatten("F")

        grains_in_mask = np.unique(grains_index[masked_object_flat])
        grains_out_mask = np.unique(grains_index[~masked_object_flat])
        mask_boarder = np.intersect1d(grains_out_mask, grains_in_mask)
        for index in mask_boarder:
            m1 = (grains_index == index) & masked_object_flat
            m2 = (grains_index == index) & ~masked_object_flat
            l1 = len(grains_index[m1])
            l2 = len(grains_index[m2])
            if l1 > l2:
                mask = grains_index == index
                masked_object_flat[mask] = True
            else:
                mask = grains_index == index
                masked_object_flat[mask] = False

        masked_object_flat = masked_object_flat.flatten("F")

        for field_name in grains_data.PointData.keys():
            field = edt.load_data(grains_data, field_name)
            print("data shape :", field.shape)
            masked_field = np.zeros_like(field)

            masked_field[masked_object_flat] = field[masked_object_flat]

            output.PointData.append(masked_field, "{}".format(field_name))

        masked_object_flat = np.asarray(masked_object_flat, dtype=np.uint8)
        output.PointData.append(masked_object_flat, "MaskedObject")

        return 1
