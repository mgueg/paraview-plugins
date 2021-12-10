# state file generated using paraview version 5.9.0-RC2

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView("RenderView")
renderView1.ViewSize = [688, 812]
renderView1.AxesGrid = "GridAxes3DActor"
renderView1.CenterOfRotation = [49.5, 49.5, 49.5]
renderView1.StereoType = "Crystal Eyes"
renderView1.CameraPosition = [-86.88206444680105, 330.1357334575657, 160.75199997711323]
renderView1.CameraFocalPoint = [49.5000000000002, 49.50000000000013, 49.50000000000011]
renderView1.CameraViewUp = [
    0.13100032287631286,
    -0.30968255455927896,
    0.941772600370143,
]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 85.73651497465943
renderView1.BackEnd = "OSPRay raycaster"
renderView1.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView2 = CreateView("RenderView")
renderView2.ViewSize = [688, 812]
renderView2.AxesGrid = "GridAxes3DActor"
renderView2.CenterOfRotation = [49.5, 49.5, 49.5]
renderView2.StereoType = "Crystal Eyes"
renderView2.CameraPosition = [-93.08854650397346, 406.969325484716, 184.01328776329044]
renderView2.CameraFocalPoint = [49.50000000000006, 49.50000000000006, 49.49999999999996]
renderView2.CameraViewUp = [0.12500685971703732, -0.3053565923255747, 0.943997159183754]
renderView2.CameraFocalDisk = 1.0
renderView2.CameraParallelScale = 106.79679263558594
renderView2.BackEnd = "OSPRay raycaster"
renderView2.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name="Layout #1")
layout1.SplitHorizontal(0, 0.500000)
layout1.AssignView(1, renderView1)
layout1.AssignView(2, renderView2)
layout1.SetSize(1377, 812)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'Programmable Source'
image_index = ProgrammableSource(registrationName="ImageIndex")
image_index.OutputDataSetType = "vtkImageData"
image_index.Script = """import numpy as np
data=np.zeros((100,100,100))

data[:40,:,:]=1
data[40:,:,:]=3

output.SetExtent(0, 99, 0, 99, 0, 99)
output.PointData.append(data.flatten("C"), "Index")
#output.PointData.append(data, "scalars")
output.PointData.SetActiveScalars("Index")
"""
image_index.ScriptRequestInformation = """# Code for 'RequestInformation Script'.
executive = self.GetExecutive()
outInfo = executive.GetOutputInformation(0)
# we assume the dimensions are (100, 100, 100).
outInfo.Set(executive.WHOLE_EXTENT(), 0, 99, 0, 99, 0, 99)
outInfo.Set(vtk.vtkDataObject.SPACING(), 1, 1, 1)"""
image_index.PythonPath = ""

# create a new 'Programmable Source'
image_scalar_field = ProgrammableSource(registrationName="ImageScalarField")
image_scalar_field.OutputDataSetType = "vtkImageData"
image_scalar_field.Script = """import numpy as np
data=np.zeros((100,100,100))

data[:40,:,:]=0.1*np.random.randn(40,100,100)+1
data[40:,:,:]=0.1*np.random.randn(60,100,100)+3

output.SetExtent(0, 99, 0, 99, 0, 99)
output.PointData.append(data.flatten("C"), "Scalars")
#output.PointData.append(data, "scalars")
output.PointData.SetActiveScalars("Scalars")
"""
image_scalar_field.ScriptRequestInformation = """# Code for 'RequestInformation Script'.
executive = self.GetExecutive()
outInfo = executive.GetOutputInformation(0)
# we assume the dimensions are (100, 100, 100).
outInfo.Set(executive.WHOLE_EXTENT(), 0, 99, 0, 99, 0, 99)
outInfo.Set(vtk.vtkDataObject.SPACING(), 1, 1, 1)"""
image_scalar_field.PythonPath = ""

# create a new 'Compute Mean Field by Grains Filter'
computeMeanFieldbyGrainsFilter1 = ComputeMeanFieldbyGrainsFilter(
    registrationName="ComputeMeanFieldbyGrainsFilter1",
    FFTResult=image_scalar_field,
    GrainsId=image_index,
)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from computeMeanFieldbyGrainsFilter1
computeMeanFieldbyGrainsFilter1Display = Show(
    computeMeanFieldbyGrainsFilter1, renderView1, "UniformGridRepresentation"
)

# get color transfer function/color map for 'Scalars'
scalarsLUT = GetColorTransferFunction("Scalars")
scalarsLUT.RGBPoints = [
    0.4001015775332455,
    0.0,
    0.0,
    0.5625,
    0.7412218959398055,
    0.0,
    0.0,
    1.0,
    1.5209270159120534,
    0.0,
    1.0,
    1.0,
    1.9107788083766932,
    0.5,
    1.0,
    0.5,
    2.300630600841333,
    1.0,
    1.0,
    0.0,
    3.0803357208135806,
    1.0,
    0.0,
    0.0,
    3.4701875132782205,
    0.5,
    0.0,
    0.0,
]
scalarsLUT.ColorSpace = "RGB"
scalarsLUT.NanColor = [0.498039215686, 0.498039215686, 0.498039215686]
scalarsLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'Scalars'
scalarsPWF = GetOpacityTransferFunction("Scalars")
scalarsPWF.Points = [
    0.4001015775332455,
    0.0,
    0.5,
    0.0,
    3.4701875132782205,
    1.0,
    0.5,
    0.0,
]
scalarsPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
computeMeanFieldbyGrainsFilter1Display.Representation = "Surface"
computeMeanFieldbyGrainsFilter1Display.ColorArrayName = ["POINTS", "<Scalars>"]
computeMeanFieldbyGrainsFilter1Display.LookupTable = scalarsLUT
computeMeanFieldbyGrainsFilter1Display.SelectTCoordArray = "None"
computeMeanFieldbyGrainsFilter1Display.SelectNormalArray = "None"
computeMeanFieldbyGrainsFilter1Display.SelectTangentArray = "None"
computeMeanFieldbyGrainsFilter1Display.OSPRayScaleArray = "<<Scalars>>"
computeMeanFieldbyGrainsFilter1Display.OSPRayScaleFunction = "PiecewiseFunction"
computeMeanFieldbyGrainsFilter1Display.SelectOrientationVectors = "None"
computeMeanFieldbyGrainsFilter1Display.ScaleFactor = 9.9
computeMeanFieldbyGrainsFilter1Display.SelectScaleArray = "<<Scalars>>"
computeMeanFieldbyGrainsFilter1Display.GlyphType = "Arrow"
computeMeanFieldbyGrainsFilter1Display.GlyphTableIndexArray = "<<Scalars>>"
computeMeanFieldbyGrainsFilter1Display.GaussianRadius = 0.495
computeMeanFieldbyGrainsFilter1Display.SetScaleArray = ["POINTS", "<<Scalars>>"]
computeMeanFieldbyGrainsFilter1Display.ScaleTransferFunction = "PiecewiseFunction"
computeMeanFieldbyGrainsFilter1Display.OpacityArray = ["POINTS", "<<Scalars>>"]
computeMeanFieldbyGrainsFilter1Display.OpacityTransferFunction = "PiecewiseFunction"
computeMeanFieldbyGrainsFilter1Display.DataAxesGrid = "GridAxesRepresentation"
computeMeanFieldbyGrainsFilter1Display.PolarAxes = "PolarAxesRepresentation"
computeMeanFieldbyGrainsFilter1Display.ScalarOpacityUnitDistance = 1.7320508075688774
computeMeanFieldbyGrainsFilter1Display.ScalarOpacityFunction = scalarsPWF
computeMeanFieldbyGrainsFilter1Display.OpacityArrayName = ["POINTS", "<<Scalars>>"]
computeMeanFieldbyGrainsFilter1Display.SliceFunction = "Plane"
computeMeanFieldbyGrainsFilter1Display.Slice = 49

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
computeMeanFieldbyGrainsFilter1Display.OSPRayScaleFunction.Points = [
    -7361.046423546424,
    0.0,
    0.5,
    0.0,
    9035.612551237551,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
computeMeanFieldbyGrainsFilter1Display.ScaleTransferFunction.Points = [
    0.039929612742203434,
    0.0,
    0.5,
    0.0,
    0.06003531027616263,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
computeMeanFieldbyGrainsFilter1Display.OpacityTransferFunction.Points = [
    0.039929612742203434,
    0.0,
    0.5,
    0.0,
    0.06003531027616263,
    1.0,
    0.5,
    0.0,
]

# init the 'Plane' selected for 'SliceFunction'
computeMeanFieldbyGrainsFilter1Display.SliceFunction.Origin = [49.5, 49.5, 49.5]

# setup the color legend parameters for each legend in this view

# get color legend/bar for scalarsLUT in view renderView1
scalarsLUTColorBar = GetScalarBar(scalarsLUT, renderView1)
scalarsLUTColorBar.Title = "Scalars"
scalarsLUTColorBar.ComponentTitle = ""

# set color bar visibility
scalarsLUTColorBar.Visibility = 1

# show color legend
computeMeanFieldbyGrainsFilter1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView2'
# ----------------------------------------------------------------

# show data from image_index
image_indexDisplay = Show(image_index, renderView2, "UniformGridRepresentation")

# get color transfer function/color map for 'Index'
indexLUT = GetColorTransferFunction("Index")
indexLUT.RGBPoints = [
    0.518956327245927,
    0.0,
    0.0,
    0.5625,
    0.8478127688749793,
    0.0,
    0.0,
    1.0,
    1.5994861153111364,
    0.0,
    1.0,
    1.0,
    1.9753220486014815,
    0.5,
    1.0,
    0.5,
    2.3511579818918267,
    1.0,
    1.0,
    0.0,
    3.1028313283279845,
    1.0,
    0.0,
    0.0,
    3.4786672616183294,
    0.5,
    0.0,
    0.0,
]
indexLUT.ColorSpace = "RGB"
indexLUT.NanColor = [0.498039215686, 0.498039215686, 0.498039215686]
indexLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'Index'
indexPWF = GetOpacityTransferFunction("Index")
indexPWF.Points = [0.518956327245927, 0.0, 0.5, 0.0, 3.4786672616183294, 1.0, 0.5, 0.0]
indexPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
image_indexDisplay.Representation = "Surface"
image_indexDisplay.ColorArrayName = ["POINTS", "Index"]
image_indexDisplay.LookupTable = indexLUT
image_indexDisplay.SelectTCoordArray = "None"
image_indexDisplay.SelectNormalArray = "None"
image_indexDisplay.SelectTangentArray = "None"
image_indexDisplay.OSPRayScaleArray = "Index"
image_indexDisplay.OSPRayScaleFunction = "PiecewiseFunction"
image_indexDisplay.SelectOrientationVectors = "None"
image_indexDisplay.ScaleFactor = 9.9
image_indexDisplay.SelectScaleArray = "Index"
image_indexDisplay.GlyphType = "Arrow"
image_indexDisplay.GlyphTableIndexArray = "Index"
image_indexDisplay.GaussianRadius = 0.495
image_indexDisplay.SetScaleArray = ["POINTS", "Index"]
image_indexDisplay.ScaleTransferFunction = "PiecewiseFunction"
image_indexDisplay.OpacityArray = ["POINTS", "Index"]
image_indexDisplay.OpacityTransferFunction = "PiecewiseFunction"
image_indexDisplay.DataAxesGrid = "GridAxesRepresentation"
image_indexDisplay.PolarAxes = "PolarAxesRepresentation"
image_indexDisplay.ScalarOpacityUnitDistance = 1.7320508075688774
image_indexDisplay.ScalarOpacityFunction = indexPWF
image_indexDisplay.OpacityArrayName = ["POINTS", "Index"]
image_indexDisplay.IsosurfaceValues = [2.0]
image_indexDisplay.SliceFunction = "Plane"
image_indexDisplay.Slice = 49

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
image_indexDisplay.OSPRayScaleFunction.Points = [
    -7361.046423546424,
    0.0,
    0.5,
    0.0,
    9035.612551237551,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
image_indexDisplay.ScaleTransferFunction.Points = [
    1.0,
    0.0,
    0.5,
    0.0,
    3.0,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
image_indexDisplay.OpacityTransferFunction.Points = [
    1.0,
    0.0,
    0.5,
    0.0,
    3.0,
    1.0,
    0.5,
    0.0,
]

# init the 'Plane' selected for 'SliceFunction'
image_indexDisplay.SliceFunction.Origin = [49.5, 49.5, 49.5]

# setup the color legend parameters for each legend in this view

# get color legend/bar for indexLUT in view renderView2
indexLUTColorBar = GetScalarBar(indexLUT, renderView2)
indexLUTColorBar.Title = "Index"
indexLUTColorBar.ComponentTitle = ""

# set color bar visibility
indexLUTColorBar.Visibility = 1

# show color legend
image_indexDisplay.SetScalarBarVisibility(renderView2, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# restore active source
SetActiveSource(computeMeanFieldbyGrainsFilter1)
# ----------------------------------------------------------------


if __name__ == "__main__":
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory="extracts")
