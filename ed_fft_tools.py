"""
ed_fft_tools

This module is a collection of tools to do post-treatment based for FFT result,
more precisely for EVP-FFT result


:author: Mikael Gueguen
"""
import numpy as np
import scipy as sp
from vtk.numpy_interface import algorithms as algs

# Angles d'Euler de la phase beta d'orientations respectives 001 , 101 , 111
beta_orientation = np.array([[0, 0, 0], [0, 0.7854, 0], [0, 0.9599, 0.7854]])

types = 3 * "basal " + 3 * "prism " + 12 * "pyram "
LIST_TYPES = types.split()
basal = ["{1}_{0}".format(i, n) for i, n in enumerate(LIST_TYPES[:3])]
prism = ["{1}_{0}".format(i, n) for i, n in enumerate(LIST_TYPES[4:7])]
pyram = ["{1}_{0}".format(i, n) for i, n in enumerate(LIST_TYPES[8:])]
LIST_TYPES_INDEXED = basal + prism + pyram
VERBOSE = False


def mask_sphere(array_size, center, r):
    """
        Create a mask based of sphere shape in a volumic box

    Args:
        array_size:    tuple with dimension of the box
        center    :    tuple with center of mask_sphere
        r         :    radius of shpere

    Returns:
        numpy mask:    test if voxels are inside or outside the sphere

    """
    coords = np.ogrid[: array_size[0], : array_size[1], : array_size[2]]
    distance = np.sqrt(
        (coords[0] - center[0]) ** 2
        + (coords[1] - center[1]) ** 2
        + (coords[2] - center[2]) ** 2
    )
    return distance <= r


def angles_in_deg(ea_field):
    """
    check if euler angles from ea_field is in degrees or not

    Args:
        ea_field:      VTK field containing euler angles

    Returns:
        True or False
    """
    return (
        np.all(np.abs(ea_field) < 360)
        & np.all(np.abs(ea_field) > 0.0)
        & np.any(np.abs(ea_field) > np.pi)
    )


def load_data(field_data, data_name, dim_in=None):
    """
    Load Dataset (eg `vtkDataSet`) with key name

    Args:
        field_data:      any VTK field (vtkDataSet)
        data_name :      string for key data name

    Returns:
        volume_grains:  3D numpy array containing volume grains field
    """
    # raise NotImplementedError("Todo")

    array = field_data.PointData[data_name]
    print("initial arr.shape : ", array.shape)
    if dim_in:
        print("New arr.shape : ", dim_in)
        return array.reshape(dim_in)
    return array


def compute_theta_direction(angles, load_direction="y"):
    """
    Compute theta function in of direction loading considering euler angles
    in radian ;
    by default it is currently defined as the angle between the c axis of the
     HCP lattice and the Y axis

    """
    assert (
        load_direction == "y"
    ), "You must use 'y' for load direction, theta is only defined for this"
    return 180 * (np.arccos(np.sin(angles[0]) * np.sin(angles[1]))) / np.pi


def compute_theta_direction_R(R, direction=np.array([0, 0, 1])):
    """
    Compute theta function in of direction loading considering euler angles
    in radian

    """
    # On sait que R x (001) = (0 0 cos(thetaz)
    index_direction = np.nonzero(direction)[0][0]
    return np.arccos(np.dot(R, direction)[index_direction]) * 180.0 / np.pi


def compute_young_modulus(Cij, h, k, l):
    """
    return Young Modulus from beta phase function of orientation
    """
    S = np.linalg.inv(Cij) * 10 ** (-9)
    # Formule pour calculer le module d'Young.
    # Ici aussi une erreur dans la these d'Aurelien.
    # Il faut penser a diviser par h**2+k**2+l**2)**2
    E = (
        1
        / (
            S[0, 0]
            - (2 * (S[0, 0] - S[0, 1]) - S[3, 3])
            * (h ** 2 * k ** 2 + k ** 2 * l ** 2 + l ** 2 * h ** 2)
            / (h ** 2 + k ** 2 + l ** 2) ** 2
        )
        / 10 ** 9
    )

    return E


def center_of_mass(grain_index_field):
    """
    Return center of mass of grains

    Args:
        grain_index_field:      VTK field containing index index_voxels


    """
    return np.array(
        [
            [
                np.average(component)
                for component in np.nonzero(grain_index_field == indx)
            ]
            for indx in np.unique(grain_index_field)
        ]
    )


def mat_from_euler_angles(phi):
    """
    Return transformation matrix function of euler angles (assumed in radians)

    Args:
        phi : numpy array (shape : [3]), with euler angles assumed in radians

    Returns:
        P :  3x3 numpy array

    ..note::

        P is computed as
        ts_{i} = P_{ij} tc_{j}

        with:
            - ts_{i} i component of t vector from sample frame
            - tc_{i} i component of t vector from crystalline frame
    """
    # conversion des angles d'Euler en radians
    #     phi1 = np.radians(phi[0])
    #     Phi = np.radians(phi[1])
    #     phi2 = np.radians(phi[2])

    # calcul des termes de la matrice de passage
    phi1 = phi[0]
    Phi = phi[1]
    phi2 = phi[2]

    c1 = np.cos(phi1)
    s1 = np.sin(phi1)
    c2 = np.cos(phi2)
    s2 = np.sin(phi2)
    cG = np.cos(Phi)
    sG = np.sin(Phi)

    # construction de la matrice de passage P
    P = np.array(
        [
            [c1 * c2 - s1 * s2 * cG, -c1 * s2 - s1 * c2 * cG, s1 * sG],
            [s1 * c2 + c1 * s2 * cG, -s1 * s2 + c1 * c2 * cG, -c1 * sG],
            [s2 * sG, c2 * sG, cG],
        ]
    )
    return P.transpose()


def compute_volume(grain_index_field, vx_size=(1.0, 1.0, 1.0)):
    """
    Compute volume grains.

    Args:
        grain_index_field:      VTK field containing index
        vx_size=(1.,1.,1.):     the voxel size

    Returns:
        volume_grains:  3D numpy array containing volume grains field
    """
    real_indx_grains = np.unique(grain_index_field)
    volume_grains = np.zeros_like(grain_index_field)
    vx_vol = vx_size[0] * vx_size[1] * vx_size[2]
    for index in real_indx_grains:
        mask_grains = np.nonzero(grain_index_field == index)
        volume = np.count_nonzero(grain_index_field == index) * vx_vol
        volume_grains[mask_grains] = volume

    return volume_grains


def find_grains_edges(grains_index_field):
    """
    Find grains edges by calculating the gradient of given index number of
    each grains.
    The result is regularized with closing/opening morphology operation with
    ball structure operator.

    Args:
        grain_index_field:      VTK field containing index

    Returns:
        initial_mask : 3D numpy array with 2 phases (as uint8) edges or not
                    given by gradient mask information
        adapted_mask: 3D numpy array with 2 phases (as uint8) edges or with
                    regularized morphology operation

    """
    ball_r1 = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    # dim_data = inputs[0].GetDimensions()
    # vx_size = inputs[0].GetSpacing()
    # vx_vol = vx_size[0]*vx_size[1]*vx_size[2]
    # extent_data = inputs[0].GetExtent()
    #
    # grains_index_field = inputs[0].PointData['FeatureIds']
    grad_index = algs.gradient(grains_index_field)

    # print(grad_index.shape)
    # output.PointData.append(grad_index, "Grad")

    initial_mask = (
        (grad_index[:, 0] != 0.0)
        | (grad_index[:, 1] != 0.0)
        | (grad_index[:, 2] != 0.0)
    )
    initial_mask = np.asarray(initial_mask, dtype=np.uint8)
    adapted_mask = sp.ndimage.binary_closing(
        sp.ndimage.binary_opening(initial_mask, structure=ball_r1), structure=ball_r1
    )
    # output.PointData.append(mask, "labels")

    return initial_mask, adapted_mask


def compute_mean_field(
    grain_index_field,
    field_data,
    field_name,
    vx_size=(1.0, 1.0, 1.0),
    weighted=False,
    compute_std_dev=False,
):
    """
    Compute mean shear system by grains.

    Args:
        grain_index_field :      VTK field containing index
        field_data :             VTK field containing shear field
        field_name :   the requested name of field
        vx_size=(1.,1.,1.) :     the voxel size
        weighted=False :          whether or not the mean and stddev is weighted
                                 by grain volume ratio
        compute_std_dev=False :          whether we compute standard deviation
                                    for `field_name`

    Returns:
        value_by_grain: 2D numpy array with every mean value for each grains
        mean_field:     3D numpy array containing mean shear field
        std_field:      3D numpy array containing standard_dev grains field
                            if compute_std_dev is True
    """

    real_indx_grains = np.unique(grain_index_field)
    field = field_data.PointData[field_name]
    field_dimension = field_data.GetDimensions()
    mean_field = np.zeros_like(field)
    std_field = np.zeros_like(field)
    # volume_grains = np.zeros_like(grain_index_field)
    vx_vol = np.prod(vx_size)  # vx_size[0]*vx_size[1]*vx_size[2]
    # print(np.prod(vx_size))

    # if weighted:
    volume_total = vx_vol * np.prod(field_dimension)
    # else:
    #     volume_total = 1.0

    # print(" volume_total ", volume_total)
    # print(" np.prod(field_dimension) ", np.prod(field_dimension))

    volume = 1.0
    for index in real_indx_grains:
        mask_grains = np.nonzero(grain_index_field == index)
        # if weighted:
        #     volume = np.count_nonzero(grain_index_field == index) * vx_vol

        mean = algs.mean(field[mask_grains], axis=0)  # * volume / volume_total
        if VERBOSE:
            print(
                "- index {} v_i {} v_t {} mean {} mean {}".format(
                    index,
                    volume,
                    volume_total,
                    algs.mean(field[mask_grains], axis=0),
                    mean,
                )
            )

        if compute_std_dev:
            std_dev = np.std(field[mask_grains], axis=0)  # * volume / volume_total
            std_field[mask_grains] = std_dev

        # volume_grains[mask_grains] = volume
        mean_field[mask_grains] = mean

    # gamma_by_grain = np.row_stack(gamma_by_grain)
    value_by_grain = np.unique(mean_field, axis=0)
    # print(" gamma_by_grain ", gamma_by_grain.shape)
    # mean_by_grains = np.column_stack((real_indx_grains,gamma_by_grain))

    return value_by_grain, mean_field, std_field


def treshold_field(shear_treshold, gamma_by_grain):
    """
    Determine all grains with shear max greater than value `shear_treshold`
    The output array correspond to all grains in rows, and number of max
    activated shear systems in columns.

    Args:
        float corresponding to treshold value
        numpy array corresponding to mean shear by grains

    Returns:
        unique_grains       : 1D numpy array containing all grains index
        counts_shear_grains : 1D numpy array containing number of systems
                                activated for each grains
        syst_activated      : 2D numpy array with system index for
                                each activated grains ;
                                all values are initialized to -1 unless
                                a system is activated.

    """

    # global LIST_TYPES

    abs_gamma_by_grain = np.abs(gamma_by_grain)
    if np.any(abs_gamma_by_grain >= shear_treshold):
        shear_activated = abs_gamma_by_grain >= shear_treshold
        nb_shear_sup_tresh = np.count_nonzero(shear_activated, axis=1)
        indx_shear_sup_tresh = np.nonzero(shear_activated)
        # print("indx_shear_sup_tresh[0] : ", indx_shear_sup_tresh[0])
        # real_index = real_indx_grains[indx_shear_sup_tresh[0]]

        # crss0_act = [(g,crss0_values[indx_shear_sup_tresh[1][i]]) for i,g in enumerate(indx_shear_sup_tresh[0])]
        # type_act = [(g,list_types_indexed[indx_shear_sup_tresh[1][i]]) for i,g in enumerate(indx_shear_sup_tresh[0])]

        nb_act = np.array(
            [
                [g, np.count_nonzero(LIST_TYPES[indx_shear_sup_tresh[1][i]])]
                for i, g in enumerate(indx_shear_sup_tresh[0])
            ]
        )

        # print("nb act ", nb_act)
        unique_grains, counts_shear_grains = np.unique(nb_act[:, 0], return_counts=True)
        max_activated = np.max(counts_shear_grains)

        syst_activated = -1 * np.ones(
            (len(unique_grains), max_activated + 1), dtype=int
        )
        for i, gr in enumerate(unique_grains):
            # gammas_sorted = np.sort(abs_gamma_by_grain[gr,:])[::-1]
            index_gammas = np.argsort(abs_gamma_by_grain[gr, :])[::-1]
            syst_activated[i, 0] = gr
            nb_act = counts_shear_grains[i] + 1
            syst_activated[i, 1:nb_act] = index_gammas[: counts_shear_grains[i]]
            # print(">>-- nb act for grain {} = {}".format(gr,counts_shear_grains[i]))
            # print("  -- nb act for grain {} = {}".format(gr,index_gammas[:counts_shear_grains[i]]))

        return unique_grains, counts_shear_grains, syst_activated
    else:
        return None


# def change_orientation_for_beta_phase(field_data, beta_orientation):
#     """
#     :author: HUET Anaïs
#
#     :date: 2021 (stage ingénieur)
#
#     Manage phases corresponding to different orientation
#
#     """
#
#     dim_data = field_data.GetDimensions()
#     # On charge les champs permettant de différencier les phases (=1 si alpha
#     # et 2 si beta)
#     image_data = load_data(field_data, "ImageData")
#
#     # On charge la colonne qui contient les numéros de grains et on regarde
#     # le nombre total de grains
#     if "Index" in field_data.PointData.keys():
#         grains_index = field_data.PointData["Index"]
#     elif "FeatureIds" in field_data.PointData.keys():
#         grains_index = field_data.PointData["FeatureIds"]
#     else:
#         raise RuntimeError(
#             "keys 'Index', 'FeatureIds' is not found in PointData.keys()"
#         )
#
#     nb_grains = len(np.unique(grains_index))
#     print(nb_grains)
#
#     # On crée le champ Phases nécessaire au calculateur (=1 si phase alpha
#     # et 2 si phase beta)
#     phases = np.zeros(grains_index.shape[0])
#     phases = image_data / 255 + 1
#
#     if "MaskedObject" in field_data.PointData.keys():
#         # On charge les champs permettant de différencier les zones
#         # (matrice ou grain central)
#         in_zone = load_data(field_data, "ImageDatacercle", dim_data)
#         in_data_zone = np.unique(grains_index[in_zone == 255])
#         out_data_zone = np.unique(grains_index[in_zone != 255])
#         # On cherche les nodules qui sont coupés par la frontière
#         # entre les deux zones et donc qui sont à la fois dans la
#         # matrice et dans le grain
#         intersected_zone = np.intersect1d(out_data_zone, in_data_zone)
#         print(intersected_zone)
#     else:
#         in_zone = np.zeros(dim_data)
#
#     # On charge les angles d'Euler de la phase alpha
#     ea_alpha = load_data(image_data, "EulerAnglesalpha", dim_data)
#     # Lorsque la matrice est aléatoire, on charge également
#     # les angles d'euler de la phase beta qui n'a donc plus une
#     # orientation unique
#     ea_beta = load_data(image_data, "EulerAnglesbeta", dim_data)
#
#     # On crée un champ qui contiendra les angles d'euler de tout l'agrégat
#     ea_field = np.zeros((grains_index.shape[0], 3))
#     ea_field = ea_alpha
#     mbeta1 = (phases == 2) & (in_zone != 255)
#     mbeta2 = (phases == 2) & (in_zone == 255)
#     ea_field[mbeta1] = ea_beta[mbeta1]  # Pour matrice aleatoire
#     # Ligne à changer lorsque l'orientation de la phase beta
#     # du grain central est modifiée
#     ea_field[mbeta2] = beta_orientation[0]
#
#     # On fait la même démarche avec les IPF
#     # ipf_alpha=load_data(data,"IPFColorz",dim_data)
#     # ipf_beta1=load_data(data,"IPFColorbetaz",dim_data) #Pour aleatoire
#     # Ligne à changer lorsque l'orientation de la phase beta 'du grain central
#     #  est modifiée. Doit concorder avec la ligne 53
#     # ipf_beta2=load_data(data,"IPFColorbeta001z",dim_data)
#     #
#     # ipf=np.zeros((grains.shape[0],3))
#     #
#     # m_ipf_beta_1=(phases==2)&(inzone!=255)
#     # m_ipf_beta_2=(phases==2)&(inzone==255)
#     #
#     # ipf[phases==1]=ipf_alpha[phases==1]
#     # ipf[m_ipf_beta_1]=ipf_beta1[m_ipf_beta_1]
#     # ipf[m_ipf_beta_2]=ipf_beta2[m_ipf_beta_2]
#
#     # On associe les nodules aux frontières à la zone à laquelle
#     # ils appartiennent
#     # le plus en comptant le nombre de voxels de chacun de ces
#     # grains dans chaque zone.
#     for grain in intersected_zone:
#         m1 = (grains_index == grain) & (in_zone == 255) & (phases == 1)
#         m2 = (grains_index == grain) & (in_zone != 255) & (phases == 1)
#         l1 = len(grains_index[m1])
#         l2 = len(grains_index[m2])
#         if l1 > l2:  # Si il y a plus de voxels dans le grain central
#             m1i = (grains_index == grain) & (phases == 1)
#             in_zone[m1i] = 255  # La zone occupée par ce nodule est integrée
#             # au grain central
#         else:  # Si il y a plus de voels dans la matrice
#             m1e = (grains_index == grain) & (phases == 1)
#             in_zone[m1e] = 0  # La zone occupée par ce nodule est integrée à la matrice
#
#     return in_zone, ea_field  # , ipf
