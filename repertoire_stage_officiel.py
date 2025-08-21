# This directory is dedicated to the treatment of the patellar tendon. Il y a plusieurs fonctions dans ce répertoire, elles peuvent toutes être utilisé 
# pour mieux les comprendre, mais certaines d'entres elles sont suelement des fonctions tremplins pour les fonctions les plus importantes et utilisables. il n'est donc pas nécessaire de toutes les comprete sonndre
# Les fonctions que je note sont celles qui doivent être comprise pour être utiliser. It consists of the following 4 parts:
"""
1. Functions that calculate second order statistics analyze on segmentation and extraction of Haralick features.
    - GLCM_treatement_tendon_3D

2. Functions to quantify and visualize inter-patient and intra-patient variance and the inter/intra-patient ratio.
    - GLCM_acq1_2_VF_mean_tendon
    - GLCM_acq1_seriea_b_VF_mean_tendon
    - plot_variance_intraindividuelle
    - ratio
    - moyenne_angles
    - hist_ratio
    - mean_ratio
    - display_all_six_plot

3. Functions that calculate the reproducibility of methods using RMSCV and CV, considering the median instead of mean.
    - find_file
    - GLCM_acq1_2_VF_tendon
    - GLCM_acq1_seriea_b_VF_tendon
    - GLCM_acq1_2_LG_tendon

Pour le calcul de repro, plusieurs cas sont possibles: 
    - Cas 1, CAS UTILISABLE LORSQUE L'ON TRAITE TOUS LE SUJETS D'UN COUP: Il y a une seule pair d'hyper-paramètre : (distance1, angle1) alors on utilise print_repro(features_list1, features_list1) où 
             features_list1 est le résultat des fonctions soit f1, f2 ou f3 pr acq1 et pareil pr features_list2 pr acq2. 
    - Cas 2, CAS UTILISABLE LORSQUE L'ON TRAITE TOUS LE SUJETS D'UN COUP: Il y a plusieurs combinaisons d'hyper-paramètres possible soit (distance_liste, angle1) soit (distance1, angle_liste) alors on 
             utilise print_repro_mean_parameter(features_list1, features_list1) où features_list1 est le résultat des fonctions soit f1, f2 ou f3 pr acq1 et pareil pr features_list2 pr acq2. 
    - Cas 3, CAS UTILISABLE LORSQUE L'ON TRAITE UN SEUL PATIENT: extract_feature_per_patient puis cv_robuste_mad pour une seule pair d'hyper-paramètre (distance1, angle1)

Dans tous les cas, un exemple est donné dans le Jupyter Notebook. 

4. GLCM for muscle: GLCM_treatement_muscle_3D and GLCM_acq1_2_VF_muscle

5. LBP functions for tendon and muscle

All thoses function can be tested and used to understand what they do but the real function to used for analyzed some datas are denoted by "FONCTION A UTILISER". 
Functions can be tested and used on a Jupyter Notebook. 

"""
# Import des librairies 

import numpy as np
import nrrd
import matplotlib.pyplot as plt
from typing import List, Tuple
from skimage.morphology import erosion, dilation, disk
from skimage.feature import graycomatrix,graycoprops
from statistics import mean, pvariance
import pandas as pd
from scipy.stats import shapiro 
from scipy.stats import spearmanr
import imageio.v2 as imageio
import seaborn as sns
import os
from typing import List, Tuple
from datetime import datetime

# Sujets traités
patient_label = ["IH1", "CB2", "HT1", "CA1", "ET1", "SLT", "JH1", "MK1", "SR1", 
                 "AV1", "AB1", "AB2", "SB1", "FB1", "CS1", "CB1", "PG", "YG", "IR"]


# 1. Functions that calculate second order statistics analyze on segmentation and extraction of Haralick features.

def convert_intensities(img : np.ndarray) -> np.ndarray: ## NE PAS UTILISER CETTE FONCTION
    """
    This functions returns the ultrasound image with pixel's intensity between 0 and 255 instead of 0 and 1; 

    Arg:
        roi_img (np.ndarray): A matrix representing the region of interest of a gray-level ultrasound image with pixels 
                              that have an intensity between 0 and 1.

    Returns:
        nd.array: A matrix representing the region of interest of a gray-level ultrasound image with pixels that have 
                  an intensity between 0 and 255.
    """    
    img = img/img.max()
    img = 255*img
    img = img.astype(np.uint8)

    return img


def crooped_tendon(seg: np.ndarray) -> None:    ## UTILISER CETTE FONCTION
    """ 
    This function returns the beginning and the end of the segmentation : the first slide of the segmentation, and the last slide of the segmentation. 

    Args:
        seg (np.ndarray): A 3D image of tendon's segmentation : the first dimension is the number of line, the second one is the number of column and the last one is about the slide of the volume. 

    Returns : 
        Tuple : a tuple containing two Nonetype : the first slide, where the segmentation is visible and the last slide where the segmentation is visible.
    """
    for i in range(seg.shape[2]): 
        if np.max(seg[:,:,i])!=0:
            break 
    print("The first slide, where the segmentation is visible is", str(i))

    for j in range(seg.shape[2]//2, seg.shape[2]):
        if np.max(seg[:,:,j])==0:
            break 
    print(j-1)


def display_six_plot_two_acq(features_list1: list, features_list2: list, suptitle : str) -> None:   ## NE PAS UTILISER CETTE FONCTION
    """
    This functions displays 6 graphs in 2 rows, 3 columns with two curves (acquisitions) in each plot.

    Args : 
        features_list (list) : list of 6 features (contrast, correlation, dissimilarity, homogeneity, ASM, entropy) 
                               for a patient. Each feature is a numpy.ndarray type;   
        suptitle (str) : suptitle for the 6 graphs.
    
    Returns : 
        NoneType : 6 graphs showing the evolution of each feature as a function of acquisitions.    
    """
    plt.figure(figsize = (18,8))

    title_list = ["Contrast","Correlation","Dissimilarity","Homogeneity","ASM", "Entropy"]
    for i in range(6):

        plt.subplot(2,3, i+1)
        plt.plot(features_list1[i],marker = 'o', color = "#69B3A2", label = "Acquisition 1")
        plt.plot(features_list2[i],marker = 'o', color = "indianred", label = "Acquisition 2")
        plt.title(title_list[i],fontsize=18)
        plt.grid(True)
        plt.xlabel("Slide of the 3D volume", fontsize = 18)
        plt.ylabel("Feature", fontsize = 18)
        plt.legend()

    plt.suptitle(suptitle, fontsize=20)
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    plt.show()


def mean_haralick_feature(harlick_features_acq1: list, harlick_features_acq2: list) -> list:    ## NE PAS UTILISER CETTE FONCTION
    """
    This function returns the mean between the first acquisition and the second one.

    Args:
        harlick_features_acq1 (list): list containing 6 Haralick feature for one patient, each feature has a value per slide of the volume, for the first acquisition e.g contrast has 100 values (one value per slide)
        harlick_features_acq2 (list): list containing 6 Haralick feature for one patient, each feature has a value per slide of the volume, for the second acquisition e.g contrast has 100 values (one value per slide)

    Returns:
        list: containing mean between the first acquisition and the second one : ## a voir a tester  
    """
    harlick_features_acq1 = np.array(harlick_features_acq1)
    harlick_features_acq2 = np.array(harlick_features_acq2)

    contraste_mean = np.mean((np.array(harlick_features_acq1).reshape(6,len(harlick_features_acq1[0])), np.array(harlick_features_acq2).reshape(6,len(harlick_features_acq2[0]))), axis = 0)[0,:]
    correlation_mean = np.mean((np.array(harlick_features_acq1).reshape(6,len(harlick_features_acq1[0])), np.array(harlick_features_acq2).reshape(6,len(harlick_features_acq2[0]))), axis = 0)[1,:]
    dissimilarity_mean = np.mean((np.array(harlick_features_acq1).reshape(6,len(harlick_features_acq1[0])), np.array(harlick_features_acq2).reshape(6,len(harlick_features_acq2[0]))), axis = 0)[2,:]
    homogeneity_mean = np.mean((np.array(harlick_features_acq1).reshape(6,len(harlick_features_acq1[0])), np.array(harlick_features_acq2).reshape(6,len(harlick_features_acq2[0]))), axis = 0)[3,:]
    ASM_mean = np.mean((np.array(harlick_features_acq1).reshape(6,len(harlick_features_acq1[0])), np.array(harlick_features_acq2).reshape(6,len(harlick_features_acq2[0]))), axis = 0)[4,:]
    entropy_mean = np.mean((np.array(harlick_features_acq1).reshape(6,len(harlick_features_acq1[0])), np.array(harlick_features_acq2).reshape(6,len(harlick_features_acq2[0]))), axis = 0)[5,:]

    return [contraste_mean, correlation_mean, dissimilarity_mean, homogeneity_mean, ASM_mean, entropy_mean]
    
def save_glcm_features_to_excel_onepatient(acq1: list, acq2: list,
                                filename: str,
                                distance: int, angle: float,
                                patient_label: str) -> None:
    """
    Sauvegarde les features Haralick (acq1 et acq2) dans un fichier Excel.
    """

    features = ["Contrast", "Correlation", "Dissimilarity", "Homogeneity", "ASM", "Entropy"]

    df_acq1 = pd.DataFrame({feat: acq1[i] for i, feat in enumerate(features)})
    df_acq1["Acquisition"] = "Acq1"
    df_acq1["Slice"] = range(1, len(acq1[0]) + 1)

    df_acq2 = pd.DataFrame({feat: acq2[i] for i, feat in enumerate(features)})
    df_acq2["Acquisition"] = "Acq2"
    df_acq2["Slice"] = range(1, len(acq2[0]) + 1)

    df = pd.concat([df_acq1, df_acq2], ignore_index=True)
    df["Distance"] = distance
    df["Angle (rad)"] = angle
    df["Patient"] = patient_label

    # Ajout distance + angle au nom du fichier
    filename = filename.replace(".xlsx", f"_dist{distance}_angle{round(angle, 2)}.xlsx")

    df.to_excel(filename, index=False)
    print(f"✅ Fichier Excel sauvegardé : {filename}")

def GLCM_treatement_tendon_3D(
    echo_acq1: np.ndarray, 
    seg_acq1: np.ndarray, 
    frame_start1: int, 
    frame_end1: int, 
    echo_acq2: np.ndarray, 
    seg_acq2: np.ndarray, 
    frame_start2: int, 
    frame_end2: int, 
    distance: int, 
    angle: int, 
    patient_label: str) -> tuple[list, list, list]: ## UTILISER CETTE FONCTION
    
    """
    This functions applies second order statistics on tendon segmentation and extract Haralick features for both acquisitions. 

    Args : 
    echo_acq1 (np.ndarray): Tendon 3D ultrasound image of the first acquisition. Dimension of the image is (512,512,512). The first dimension is sagital plane of the tendon, 
                            the second one is the coronal plane, the last one is the transversal plane. 
    seg_acq1 (np.ndarray) : Segmentation of the tendon 3D ultrasound image of the first acquisition. Dimension of the image is (512,512,512). The first dimension is sagital plane of the tendon, 
                            the second one is the coronal plane, the last one is the transversal plane.
    frame_start1 (int)    : The first slide where the segmentation of the tendon is visible. frame_start1 ∈ [0,412]
    frame_end1 (int)      : The last slide of the tendon segmentation volume; Usually frame_end1 = frame_start1 + nx where nx can be equal to 90 or 100, depends on the patient's height cf Christine for more details.  
                            The volume studied is of size (512 row, 512 columns, frame_end1-frame_start1)
    Same for echo_acq2, seg_acq2, frame_start2, frame_end2, but for the second acquisition. 
    distance (int)        : Parameter of the GLCM, distance between two pixels that we are studying. 
    angle (int)           : Parameter of the GLCM, angle between two pixels that we are studying; BE CAREFUL !! Angle's unity is not degree but radians, then must write e.g np.radians(0), for angle = 0 radians. 
    patient_label (str)   : Name of the patient, using for the title of plot.
    
    Returns:
    Tuple [list, list, list] : the first list (respectively the second list) is about 6 features (contrast, correlation, dissimilarity, homogeneity, ASM, entropy) extracted for the first acquisition (respectively 
                                for the second acquisition) and displays into 6 graphs, one graph per feature. A graph is the evolution of the feature in the volume (as a function of slices). 
                                The last list is the mean between feature extracted from the first acquisition and from the second one. 
    """
    # Ouvrir la segmentation de la première acquisition 
    pre_tendon1 = echo_acq1*(seg_acq1>0)
    tendon1 = np.rot90(pre_tendon1, k = 3)

    # Ouvrir la segmentation de la deuxième acquisition
    pre_tendon2 = echo_acq2*(seg_acq2>0)
    tendon2 = np.rot90(pre_tendon2, k = 3)

    # Matrice vide qui contiendra les tendons érodés. 
    tendon_erode_acq1 = np.zeros_like(tendon1)
    tendon_erode_acq2 = np.zeros_like(tendon2)

    # Conversion des intensités des pixels de [0,1] à [0,255]
    tendon_convert1 = convert_intensities(tendon1)
    tendon_convert2 = convert_intensities(tendon2)

    # Pour stocker les features de la première acquisition 
    contrast1 = []
    correlation1 = []
    dissimilarity1 = []
    homogeneity1 = []
    ASM1 = []
    entropy1 = []

    # Pour stocker les features de la deuxième acquisition 
    contrast2 = []
    correlation2 = []
    dissimilarity2 = []
    homogeneity2 = []
    ASM2 = []
    entropy2 = []

    for i in range(tendon1.shape[2]): # boucle qui parcourt tout le volume de la première acquisition (taille du volume acq1 = taille du volume acq2)

        # Ouverture morphologiques : Erosion par un disque de rayon 1 pixel puis dilatation par un disque de rayon 1 des tendons pour enlever l'épimysium (la gaine) 
        tendon_erode_acq1[:,:,i] = dilation(erosion(tendon_convert1[:,:,i], disk(1)), disk(1))       
        tendon_erode_acq2[:,:,i] = dilation(erosion(tendon_convert2[:,:,i], disk(1)), disk(1))       

    # Calcul de la glcm pour la première acquisition
    for i in range(frame_start1, frame_end1):
        
        glcm_acq1 = graycomatrix(tendon_erode_acq1[:,:,i], distances = [distance], angles = [angle], levels = 256, 
                                        symmetric = False, normed = True)
        
        # La glcm est calculé pour un rectangle, mais la segmentation n'a pas la forme d'un rectangle, donc lors du calcul de la glcm on prend le fond noir en compte :
        # pour ignorer ces valeurs, on force les relations entre un pixel non noir et un pixel noir à être ignoré. Donc on met la première ligne et première colonne à 0
        
        glcm_acq1[0,:,:,:] = 0
        glcm_acq1[:,0,:,:] = 0

        # Extraction des Haralick features : on peut changer le type de Haralick feature (à regarder les autres Haralick features disponible sur la doc en ligne)
        contrast1.append(graycoprops(glcm_acq1[:,:,:,:],'contrast'))
        correlation1.append(graycoprops(glcm_acq1[:,:,:,:],'correlation'))
        dissimilarity1.append(graycoprops(glcm_acq1[:,:,:,:],'dissimilarity'))
        homogeneity1.append(graycoprops(glcm_acq1[:,:,:,:],'homogeneity'))
        ASM1.append(graycoprops(glcm_acq1[:,:,:,:],'ASM'))
        entropy1.append(graycoprops(glcm_acq1[:,:,:,:],'entropy'))

    # Analyse de la glcm pour la deuxième acquisition : même processus que pour la première acquisition
    for i in range(frame_start2, frame_end2):

        glcm_acq2 = graycomatrix(tendon_erode_acq2[:,:,i], distances = [distance], angles = [angle], levels = 256, 
                                        symmetric = False, normed = True)
        
        glcm_acq2[0,:,:,:] = 0
        glcm_acq2[:,0,:,:] = 0

        contrast2.append(graycoprops(glcm_acq2[:,:,:,:],'contrast'))
        correlation2.append(graycoprops(glcm_acq2[:,:,:,:],'correlation'))
        dissimilarity2.append(graycoprops(glcm_acq2[:,:,:,:],'dissimilarity'))
        homogeneity2.append(graycoprops(glcm_acq2[:,:,:,:],'homogeneity'))
        ASM2.append(graycoprops(glcm_acq2[:,:,:,:],'ASM'))
        entropy2.append(graycoprops(glcm_acq2[:,:,:,:],'entropy'))

    # Affichage des 6 Haralick features pour les 2 acquisitions : évolution des features en fonction dans le volume
    display_six_plot_two_acq([np.array(contrast1).reshape(len(contrast1)), np.array(correlation1).reshape(len(contrast1)), np.array(dissimilarity1).reshape(len(contrast1)), np.array(homogeneity1).reshape(len(contrast1)), np.array(ASM1).reshape(len(contrast1)), np.array(entropy1).reshape(len(contrast1))], 
                            [np.array(contrast2).reshape(len(contrast1)), np.array(correlation2).reshape(len(contrast1)), np.array(dissimilarity2).reshape(len(contrast1)), np.array(homogeneity2).reshape(len(contrast1)), np.array(ASM2).reshape(len(contrast1)), np.array(entropy2).reshape(len(contrast1))],
    f"Haralick features for patient "+ str(patient_label)+ f" distance = {distance}, angle = {angle}")

    save_glcm_features_to_excel_onepatient(
    acq1=[np.array(contrast1).reshape(len(contrast1)),
          np.array(correlation1).reshape(len(contrast1)),
          np.array(dissimilarity1).reshape(len(contrast1)),
          np.array(homogeneity1).reshape(len(contrast1)),
          np.array(ASM1).reshape(len(contrast1)),
          np.array(entropy1).reshape(len(contrast1))],

    acq2=[np.array(contrast2).reshape(len(contrast2)),
          np.array(correlation2).reshape(len(contrast2)),
          np.array(dissimilarity2).reshape(len(contrast2)),
          np.array(homogeneity2).reshape(len(contrast2)),
          np.array(ASM2).reshape(len(contrast2)),
          np.array(entropy2).reshape(len(contrast2))],

    filename=f"GLCM_features_{patient_label}.xlsx",  # base name
    distance=distance,
    angle=angle,
    patient_label=patient_label)
    
    return [contrast1, correlation1, dissimilarity1, homogeneity1, ASM1, entropy1], [contrast2, correlation2, dissimilarity2, homogeneity2, ASM2, entropy2], mean_haralick_feature([contrast1, correlation1, dissimilarity1, homogeneity1, ASM1, entropy1], [contrast2, correlation2, dissimilarity2, homogeneity2, ASM2, entropy2])


# 2. Functions to quantify and visualize inter-patient and intra-patient variance and the inter/intra-patient ratio 
def moyenne_angles(feature_patient_list: list, n_angles: int) -> list: # permet de moyenner les angles pour chaque patient, renvoie L[i][j][k] où i va de 1 à 19, j va de 1 à 6, k va de 1 au nbr de coupe dans le volume 
    """Fonction qui permet de moyenner les hyper-paramètres pr chaque patient

    Args:
        feature_patient_list (list): résultat de la fonction GLCM_acq1_2_VF_mean ou GLCM_acq1_seriea_b_VF_mean; liste qui contient tous les haralick features pour tous les patients, pour toutes les coupes du volumes et pour tous les hyper-paramètres.
        n_angles (int): nombre de hyper-paramètre contenu soit dans la liste distance_liste soit dans la liste angle_liste, par exemple si distance_list = [1,5,8] alors n_angles = 3

    Returns:
        feature_patient_list_avg (list): Renvoie une liste L[i][j][k]
            - i ∈ [1,nbr de sujets traités=19], moyenne des hyper-paramètres pr chaque patient. Par exemple, au lieu d'avoir une liste d'Haralick feature pour distance = 1 pixel, pour patient 1,...,patient 19 et 
            ensuite dans la même liste, il y aura les Haralick features pour distance = 8 pixels, pour patient 1, ..., patient 19 on aura alors plus qu'une liste d'Haralick feature pour patient 1 ... patient 19 en 
            ayant donc fait une moyenne entre les Haralick features issus des distance = 1 et = 8 pixels.  
            - j ∈ [1,6], où j correspond aux Haralick features dans l'ordre : Contraste, corrélation, dissimilarité, homogéneité, ASM, entropy. 
            - k ∈ [1, nbr de coupe du volume] : correspond concrètement à la valeur du feature j, pour le sujet i, de la coupe k du volume 3D. 
    """
    n_patients = len(feature_patient_list) // n_angles
    feature_patient_list_avg = []

    for p in range(n_patients):  

        moyenne = [[0 for _ in range(len(feature_patient_list[0][0]))] for _ in range(len(feature_patient_list[0]))]

        for a in range(n_angles): 
            idx = p + a * n_patients
            for j in range(len(feature_patient_list[0])):
                for k in range(len(feature_patient_list[0][0])):
                    moyenne[j][k] += feature_patient_list[idx][j][k] / n_angles

        feature_patient_list_avg.append(moyenne)
    return feature_patient_list_avg

def GLCM_acq1_2_VF_mean_tendon(distance_liste: list, angle_liste: list, parameter_based: str) -> list:
    """ 
    Fonction qui renvoie les Haralick features moyenne des Haralick features issus de la première acquisition et de la deuxième.
    Args : 
        distance_liste (list): liste qui contient soit un réel, sous la forme : 1, soit une liste de réel (plusieurs distances), sous la forme :[1,2,3] en pixel 
        angle_liste (list): liste qui contient soit un réel, sous la forme : np.radians (0), soit une liste de réel (plusieurs angles), sous la forme np.radians ((0,45,90)) en radians; METTRE NP.RADIANS((angle0, angle1)): en degré le code ne fonctionne pas. 
        parameter_based (str): permet de savoir quel paramètre varie; si la distance est une liste de distance et donc angle_liste un réel, alors parameter_based = "distance", si l'angle est une liste d'angle et donc distance_liste un réel, alors parameter_based = "angle"
    
    Returns: 
        haralick_feature (list): Renvoie une liste L[i][j][k]
            - i ∈ [1,(nbr dans la liste de paramètre)*nbr de sujets traités = (nbr dans la liste de paramètre)*19], donc si distance_liste = [1,2], i ∈ [1,2*19=38], L[i] aura les caractéristiques pour les 19 sujets 
              pour la distance = 1, puis les caractéristiques pour les 19 sujets pour la distance = 2. 
            - j ∈ [1,6], où j correspond aux Haralick features dans l'ordre : Contraste, corrélation, dissimilarité, homogéneité, ASM, entropy. 
            - k ∈ [1, nbr de coupe du volume] : correspond concrètement à la valeur du feature j, pour le sujet i, de la coupe k du volume 3D. 
    """

    # Dictionnaire contenant les informations nécessaires pour chaque patient
    patients = {
        "IH1": {"acq1": (echo_IH1_acq1_seriea_vf, seg_IH1_acq1_seriea_vf, 259, 339),
                "acq2": (echo_IH1_acq2_seriea_vf, seg_IH1_acq2_seriea_vf, 239, 319)},
        "CB2": {"acq1": (echo_CB2_acq1_seriea_vf, seg_CB2_acq1_seriea_vf, 204, 304),
                "acq2": (echo_CB2_acq2_seriea_vf, seg_CB2_acq2_seriea_vf, 224, 324)},
        "HT1": {"acq1": (echo_HT1_acq1_seriea_vf, seg_HT1_acq1_seriea_vf, 258, 348),
                "acq2": (echo_HT1_acq2_seriea_vf, seg_HT1_acq2_seriea_vf, 253, 343)},
        "CA1": {"acq1": (echo_CA1_acq1_seriea_vf, seg_CA1_acq1_seriea_vf, 240, 340),
                "acq2": (echo_CA1_acq2_seriea_vf, seg_CA1_acq2_seriea_vf, 210, 310)},
        "ET1": {"acq1": (echo_ET1_acq1_seriea_vf, seg_ET1_acq1_seriea_vf, 230, 320),
                "acq2": (echo_ET1_acq2_seriea_vf, seg_ET1_acq2_seriea_vf, 227, 317)},
        "SLT": {"acq1": (echo_SLT_acq1_seriea_vf, seg_SLT_acq1_seriea_vf, 230, 330),
                "acq2": (echo_SLT_acq2_seriea_vf, seg_SLT_acq2_seriea_vf, 200, 300)},
        "JH1": {"acq1": (echo_JH1_acq1_seriea_vf, seg_JH1_acq1_seriea_vf, 232, 332),
                "acq2": (echo_JH1_acq2_seriea_vf, seg_JH1_acq2_seriea_vf, 233, 333)},
        "MK1": {"acq1": (echo_MK1_acq1_seriea_vf, seg_MK1_acq1_seriea_vf, 172, 262),
                "acq2": (echo_MK1_acq2_seriea_vf, seg_MK1_acq2_seriea_vf, 198, 288)},        
        "SR1": {"acq1": (echo_SR1_acq1_seriea_vf, seg_SR1_acq1_seriea_vf, 215, 305),
                "acq2": (echo_SR1_acq2_seriea_vf, seg_SR1_acq2_seriea_vf, 230, 320)}, 
        "AV1": {"acq1": (echo_AV1_acq1_seriea_vf, seg_AV1_acq1_seriea_vf, 220, 310),
                "acq2": (echo_AV1_acq2_seriea_vf, seg_AV1_acq2_seriea_vf, 245, 335)},
        "AB1": {"acq1": (echo_AB1_acq1_seriea_vf, seg_AB1_acq1_seriea_vf, 200, 300),
                "acq2": (echo_AB1_acq2_seriea_vf, seg_AB1_acq2_seriea_vf, 189, 289)},
        "AB2": {"acq1": (echo_AB2_acq1_seriea_vf, seg_AB2_acq1_seriea_vf, 210, 310),
                "acq2": (echo_AB2_acq2_seriea_vf, seg_AB2_acq2_seriea_vf, 225, 325)},
        "SB1": {"acq1": (echo_SB1_acq1_seriea_vf, seg_SB1_acq1_seriea_vf, 225, 315),
                "acq2": (echo_SB1_acq2_seriea_vf, seg_SB1_acq2_seriea_vf, 230, 320)},
        "FB1": {"acq1": (echo_FB1_acq1_seriea_vf, seg_FB1_acq1_seriea_vf, 260, 360),
                "acq2": (echo_FB1_acq2_seriea_vf, seg_FB1_acq2_seriea_vf, 240, 340)},
        "CS1": {"acq1": (echo_CS1_acq1_seriea_vf, seg_CS1_acq1_seriea_vf, 258, 358),
                "acq2": (echo_CS1_acq2_seriea_vf, seg_CS1_acq2_seriea_vf, 263, 363)},
        "CB1": {"acq1": (echo_CB1_acq1_seriea_vf, seg_CB1_acq1_seriea_vf, 240, 330),
                "acq2": (echo_CB1_acq2_seriea_vf, seg_CB1_acq2_seriea_vf, 265, 355)},
        "PG": {"acq1": (echo_PG_acq1_seriea_vf,  seg_PG_acq1_seriea_vf,  190, 290),
                "acq2": (echo_PG_acq2_seriea_vf,  seg_PG_acq2_seriea_vf,  213, 313)},
        "YG":  {"acq1": (echo_YG_acq1_seriea_vf, seg_YG_acq1_seriea_vf, 245, 345),
                "acq2": (echo_YG_acq2_seriea_vf, seg_YG_acq2_seriea_vf, 225, 325)},
        "IR":  {"acq1": (echo_IR_acq1_seriea_vf, seg_IR_acq1_seriea_vf, 255, 355),
                "acq2": (echo_IR_acq2_seriea_vf, seg_IR_acq2_seriea_vf, 255, 355)}}

    haralick_feature = []

    if parameter_based == "distance":
        for distance in distance_liste:
            for patient, data in patients.items():
                acq1 = data["acq1"]
                acq2 = data["acq2"]
                # Appel unique avec décompression des tuples
                hf = GLCM_treatement_tendon_3D(*acq1, *acq2, distance, angle_liste, patient)[2]
                haralick_feature.append(hf)

    elif parameter_based == "angle":
        for angle in angle_liste:
            for patient, data in patients.items():
                acq1 = data["acq1"]
                acq2 = data["acq2"]
                hf = GLCM_treatement_tendon_3D(*acq1, *acq2, distance_liste, angle, patient)[2]
                haralick_feature.append(hf)

    return haralick_feature



def GLCM_acq1_seriea_b_VF_mean_tendon(distance_liste: list, angle_liste: list, parameter_based: str) -> list:
    """ 
    Fonction qui renvoie les Haralick features moyenne des Haralick features issus de la première acquisition série a et b.
    Args : 
        distance_liste (list): liste qui contient soit un réel, sous la forme : 1, soit une liste de réel (plusieurs distances), sous la forme :[1,2,3] en pixel 
        angle_liste (list): liste qui contient soit un réel, sous la forme : np.radians (0), soit une liste de réel (plusieurs angles), sous la forme np.radians ((0,45,90)) en radians; METTRE NP.RADIANS((angle0, angle1)): en degré le code ne fonctionne pas. 
        parameter_based (str): permet de savoir quel paramètre varie; si la distance est une liste de distance et donc angle_liste un réel, alors parameter_based = "distance", si l'angle est une liste d'angle et donc distance_liste un réel, alors parameter_based = "angle"
    
    Returns: 
        haralick_feature (list): Renvoie une liste L[i][j][k]
            - i ∈ [1,(nbr dans la liste de paramètre)*nbr de sujets traités = (nbr dans la liste de paramètre)*19], donc si distance_liste = [1,2], i ∈ [1,2*19=38], L[i] aura les caractéristiques pour les 19 sujets 
              pour la distance = 1, puis les caractéristiques pour les 19 sujets pour la distance = 2. 
            - j ∈ [1,6], où j correspond aux Haralick features dans l'ordre : Contraste, corrélation, dissimilarité, homogéneité, ASM, entropy. 
            - k ∈ [1, nbr de coupe du volume] : correspond concrètement à la valeur du feature j, pour le sujet i, de la coupe k du volume 3D. 
    """

    # Dictionnaire contenant les informations nécessaires pour chaque patient
    patients = {
        "IH1": {"acq1": (echo_IH1_acq1_seriea_vf, seg_IH1_acq1_seriea_vf, 259, 339),
                "acq2": (echo_IH1_acq2_seriea_vf, seg_IH1_acq2_seriea_vf, 259, 339)},
        "CB2": {"acq1": (echo_CB2_acq1_seriea_vf, seg_CB2_acq1_seriea_vf, 204, 304),
                "acq2": (echo_CB2_acq2_seriea_vf, seg_CB2_acq2_seriea_vf, 204, 304)},
        "HT1": {"acq1": (echo_HT1_acq1_seriea_vf, seg_HT1_acq1_seriea_vf, 258, 348),
                "acq2": (echo_HT1_acq2_seriea_vf, seg_HT1_acq2_seriea_vf, 258, 348)},
        "CA1": {"acq1": (echo_CA1_acq1_seriea_vf, seg_CA1_acq1_seriea_vf, 240, 340),
                "acq2": (echo_CA1_acq2_seriea_vf, seg_CA1_acq2_seriea_vf, 240, 340)},
        "ET1": {"acq1": (echo_ET1_acq1_seriea_vf, seg_ET1_acq1_seriea_vf, 230, 320),
                "acq2": (echo_ET1_acq2_seriea_vf, seg_ET1_acq2_seriea_vf, 230, 320)},
        "SLT": {"acq1": (echo_SLT_acq1_seriea_vf, seg_SLT_acq1_seriea_vf, 230, 330),
                "acq2": (echo_SLT_acq2_seriea_vf, seg_SLT_acq2_seriea_vf, 230, 330)},
        "JH1": {"acq1": (echo_JH1_acq1_seriea_vf, seg_JH1_acq1_seriea_vf, 232, 332),
                "acq2": (echo_JH1_acq2_seriea_vf, seg_JH1_acq2_seriea_vf, 232, 332)},
        "MK1": {"acq1": (echo_MK1_acq1_seriea_vf, seg_MK1_acq1_seriea_vf, 172, 262),
                "acq2": (echo_MK1_acq2_seriea_vf, seg_MK1_acq2_seriea_vf, 172, 262)},        
        "SR1": {"acq1": (echo_SR1_acq1_seriea_vf, seg_SR1_acq1_seriea_vf, 215, 305),
                "acq2": (echo_SR1_acq2_seriea_vf, seg_SR1_acq2_seriea_vf, 215, 305)}, 
        "AV1": {"acq1": (echo_AV1_acq1_seriea_vf, seg_AV1_acq1_seriea_vf, 220, 310),
                "acq2": (echo_AV1_acq2_seriea_vf, seg_AV1_acq2_seriea_vf, 220, 310)},
        "AB1": {"acq1": (echo_AB1_acq1_seriea_vf, seg_AB1_acq1_seriea_vf, 200, 300),
                "acq2": (echo_AB1_acq2_seriea_vf, seg_AB1_acq2_seriea_vf, 200, 300)},
        "AB2": {"acq1": (echo_AB2_acq1_seriea_vf, seg_AB2_acq1_seriea_vf, 210, 310),
                "acq2": (echo_AB2_acq2_seriea_vf, seg_AB2_acq2_seriea_vf, 210, 310)},
        "SB1": {"acq1": (echo_SB1_acq1_seriea_vf, seg_SB1_acq1_seriea_vf, 225, 315),
                "acq2": (echo_SB1_acq2_seriea_vf, seg_SB1_acq2_seriea_vf, 225, 315)},
        "FB1": {"acq1": (echo_FB1_acq1_seriea_vf, seg_FB1_acq1_seriea_vf, 260, 360),
                "acq2": (echo_FB1_acq2_seriea_vf, seg_FB1_acq2_seriea_vf, 240, 340)},
        "CS1": {"acq1": (echo_CS1_acq1_seriea_vf, seg_CS1_acq1_seriea_vf, 258, 358),
                "acq2": (echo_CS1_acq2_seriea_vf, seg_CS1_acq2_seriea_vf, 258, 358)},
        "CB1": {"acq1": (echo_CB1_acq1_seriea_vf, seg_CB1_acq1_seriea_vf, 240, 330),
                "acq2": (echo_CB1_acq2_seriea_vf, seg_CB1_acq2_seriea_vf, 240, 330)},
        "PG": {"acq1": (echo_PG_acq1_seriea_vf,  seg_PG_acq1_seriea_vf,  190, 290),
                "acq2": (echo_PG_acq2_seriea_vf,  seg_PG_acq2_seriea_vf,  190, 290)},
        "YG":  {"acq1": (echo_YG_acq1_seriea_vf, seg_YG_acq1_seriea_vf, 245, 345),
                "acq2": (echo_YG_acq2_seriea_vf,  seg_YG_acq2_seriea_vf,  245, 345)},
        "IR":  {"acq1": (echo_IR_acq1_seriea_vf, seg_IR_acq1_seriea_vf, 255, 355),
                "acq2": (echo_IR_acq2_seriea_vf,  seg_IR_acq2_seriea_vf,  255, 355)}}


    haralick_feature = []

    if parameter_based == "distance":
        for distance in distance_liste:
            for patient, data in patients.items():
                acq1 = data["acq1"]
                acq2 = data["acq2"]
                # Appel unique avec décompression des tuples
                hf = GLCM_treatement_tendon_3D(*acq1, *acq2, distance, angle_liste, patient)[2]
                haralick_feature.append(hf)

    elif parameter_based == "angle":
        for angle in angle_liste:
            for patient, data in patients.items():
                acq1 = data["acq1"]
                acq2 = data["acq2"]
                hf = GLCM_treatement_tendon_3D(*acq1, *acq2, distance_liste, angle, patient)[2]
                haralick_feature.append(hf)

    return haralick_feature

def variance_intra_individuelle(feature_patient_list:list) ->Tuple[list, list]:      ## NE PAS UTILISER CETTE FONCTION
    """Fonction qui calcule la moyenne pr chaque patient et la variance intra individuelle c'est à dire la variance au sein d'un même patient, et ceux pour tous les patients

    Args:
        feature_patient_list (list): résultat de la fonction GLCM_acq1_2_VF_mean ou GLCM_acq1_seriea_b_VF_mean; liste qui contient tous les haralick features pour tous les patients, pour toutes les coupes du volumes et pour tous les hyper-paramètres.

    Returns:
        Tuple[list, list]:
        mean_per_patient (list): liste qui contient les moyennes pour chaque patient et chaque feature
        variance_intra (list); liste qui contient la variance intra sujet donc variance au sein d'un même patient et ceux pour chaque patient.
    """
    mean_per_patient = []
    variance_intra = []

    for i in range(len(feature_patient_list)):  # pour chaque patient
        for j in range(6):  # pour chaque feature
            mean_per_patient.append(mean(feature_patient_list[i][j]))
            variance_intra.append(pvariance(feature_patient_list[i][j]))

    return mean_per_patient, variance_intra


def plot_variance_intraindividuelle(feature_patient_list: list, suptitle: str) -> None:        ## UTILISER CETTE FONCTION
    """Fonction qui affiche la variance intra-individuelle pour tous les patients pr chaque Haralick features.

    Args:
        feature_patient_list (list): résultat de la fonction GLCM_acq1_2_VF_mean ou GLCM_acq1_seriea_b_VF_mean; liste qui contient tous les haralick features pour tous les patients, pour toutes les coupes du volumes et pour tous les hyper-paramètres.
        suptitle (str): titre de l'affichage, à mettre sous la forme : "Illustration de la variance intra_individuelle pour tous les sujets" par exemple. 

    Returns: 
        None: Affichage de la variance intra-individuelle pour tous les patients selon chaque Haralick features.  
    """

    mean_vals, var_vals = variance_intra_individuelle(feature_patient_list)

    patients = np.array(patient_label)
    feature_names = ["Contrast", "Correlation", "Energy", "Homogeneity", "ASM", "Entropy"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    handles = None  
    labels = None

    for i in range(6):
        ax = axes[i // 3, i % 3]
        means = mean_vals[i::6]
        variances = var_vals[i::6]
        half_std = [np.sqrt(v)/2 for v in variances]
        lower = [m - h for m, h in zip(means, half_std)]
        upper = [m + h for m, h in zip(means, half_std)]

        _ = ax.vlines(patients, ymin=lower, ymax=upper, colors='b', lw=2, label="1/2 écart-type")
        _ = ax.scatter(patients, lower, color='r', label="Borne min")
        _ = ax.scatter(patients, upper, color='g', label="Borne max")
        #l4, = ax.plot(patients, means, color='orange', label="Moyenne", linestyle='--', marker='o')

        ax.set_xticks(patients)
        ax.set_xticklabels([f'Subject {i}' for i in patients], rotation=45, ha='right')
        ax.set_title(feature_names[i], fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.85, 0.5), fontsize=12)
    fig.suptitle(suptitle, fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 0.85, 0.95]) 
    
    plt.show()


def variance_inter_individuelle(feature_patient_list: list) -> list:      ## NE PAS UTILISER CETTE FONCTION
    """
    Cette fonction renvoie la variance inter-individuelle c'est à dire la variance entre les différents sujets. 

    Arg: 
        feature_patient_list (list): résultat de la fonction GLCM_acq1_2_VF_mean ou GLCM_acq1_seriea_b_VF_mean; liste qui contient tous les haralick features pour tous les patients, pour toutes les coupes du volumes et pour tous les hyper-paramètres.
    Return:
        variance_inter (list): liste de variance inter-patient selon chaque Haralick feature. 
    """
    mean_vals, _ = variance_intra_individuelle(feature_patient_list)

    features_grouped = [[] for _ in range(6)]
    for i, m in enumerate(mean_vals):
        features_grouped[i % 6].append(m)

    variance_inter = [pvariance(f) for f in features_grouped]

    return variance_inter  


def ratio(feature_patient_list: list) -> list:            ## UTILISER CETTE FONCTION
    """Fonction qui calcule le ratio inter/intra individuel selon chaque Haralick feature

    Args:
        feature_patient_list (list): résultat de la fonction GLCM_acq1_2_VF_mean ou GLCM_acq1_seriea_b_VF_mean; liste qui contient tous les haralick features pour tous les patients, pour toutes les coupes du volumes et pour tous les hyper-paramètres.

    Returns:
        list: liste qui contient le ratio inter/intra individuel selon chaque Haralick feature
    """
    variance_inter = variance_inter_individuelle(feature_patient_list)
    _, variance_intra = variance_intra_individuelle(feature_patient_list)

    # Trier les variances 
    contrast = variance_intra[::6]
    correlation = variance_intra[1::6]
    dissimilarity = variance_intra[2::6]
    
    homogeneity = variance_intra[3::6]
    ASM = variance_intra[4::6]
    entropy = variance_intra[5::6]

    ratio_contrast = contrast/variance_inter[0]
    ratio_correlation = correlation/variance_inter[1]
    ratio_dissimilarity = dissimilarity/variance_inter[2]
    ratio_homogeneity = homogeneity/variance_inter[3]
    ratio_ASM = ASM/variance_inter[4]
    ratio_entropy = entropy/variance_inter[5]

    return [ratio_contrast, ratio_correlation, ratio_dissimilarity, ratio_homogeneity, ratio_ASM, ratio_entropy]



def hist_ratio(feature_list_patient: list, suptitle: str) -> None:      ## UTILISER CETTE FONCTION, résultat de moyenne_angle à mettre dans hist_ratio
    """Fonction qui affiche l'histogramme des ratio inter/intra individuel selon chaque Haralick feature.

    Args:
        feature_list_patient (list): résultat de la fonction moyenne_angles
        suptitle (str): titre pour l'histogramme à mettre sous la forme: "Histogramme des ratios inter/intra sujet" par exemple. 

    Returns: 
        None: Histogramme contenant les ratios inter/intra individuel selon chaque Haralick feature, pour chaque sujet.
    """
    patients = patient_label

    ratio_contrast = ratio(feature_list_patient)[0]
    ratio_correlation = ratio(feature_list_patient)[1]
    ratio_energy = ratio(feature_list_patient)[2]
    ratio_homogeneity = ratio(feature_list_patient)[3]
    ratio_ASM = ratio(feature_list_patient)[4]
    ratio_entropy =ratio(feature_list_patient)[5]

    colormap = sns.color_palette("viridis", len(feature_list_patient))
    fig, _ = plt.subplots(2,3,figsize=(18,8))
    
    plt.subplot(2,3,1)
    for i in range(len(feature_list_patient)):
        plt.grid(True, linestyle="--",alpha=0.7, zorder=0)
        plt.bar(patients, ratio_contrast, color=colormap, alpha=0.5, edgecolor='black', linewidth=1.)

    plt.ylabel("Ratio", fontsize=12)
    plt.title("Ratio inter/intra du contraste",fontsize=16)
    plt.xticks(patients, [f"Subject " + patient_label[i] for i in range(len(patient_label))], rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplot(2,3,2)
    for i in range(len(feature_list_patient)):
        plt.grid(True, linestyle="--",alpha=0.7, zorder=0)
        plt.bar(patients, ratio_correlation, color=colormap, alpha=0.5, edgecolor='black', linewidth=1.)

    plt.ylabel("Ratio", fontsize=12)
    plt.title("Ratio inter/intra de corrélation",fontsize=16)
    plt.xticks(patients, [f"Subject " + patient_label[i] for i in range(len(patient_label))], rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplot(2,3,3)
    for i in range(len(feature_list_patient)):
        plt.grid(True, linestyle="--",alpha=0.7, zorder=0)
        plt.bar(patients, ratio_energy, color=colormap, alpha=0.5, edgecolor='black', linewidth=1.)

    plt.ylabel("Ratio", fontsize=12)
    plt.title("Ratio inter/intra de l'énergie",fontsize=16)
    plt.xticks(patients,[f"Subject " + patient_label[i] for i in range(len(patient_label))], rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.subplot(2,3,4)
    for i in range(len(feature_list_patient)):
        plt.grid(True, linestyle="--",alpha=0.7, zorder=0)
        plt.bar(patients, ratio_homogeneity, color=colormap, alpha=0.5, edgecolor='black', linewidth=1.)

    plt.ylabel("Ratio", fontsize=12)
    plt.title("Ratio inter/intra de l'homogenéité",fontsize=16)
    plt.xticks(patients,[f"Subject " + patient_label[i] for i in range(len(patient_label))], rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.subplot(2,3,5)
    for i in range(len(feature_list_patient)):
        
        plt.grid(True, linestyle="--",alpha=0.7, zorder=0)
        plt.bar(patients, ratio_ASM, color=colormap, alpha=0.5, edgecolor='black', linewidth=1.)

    plt.ylabel("Ratio", fontsize=12)
    plt.title("Ratio inter/intra du ASM",fontsize=16)
    plt.xticks(patients,[f"Subject " + patient_label[i] for i in range(len(patient_label))], rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplot(2,3,6)
    for i in range(len(feature_list_patient)):
        plt.grid(True, linestyle="--",alpha=0.7, zorder=0)
        plt.bar(patients, ratio_entropy, color=colormap, alpha=0.5, edgecolor='black', linewidth=1.)

    plt.xticks(patients, rotation=45, ha='right')
    plt.ylabel("Ratio", fontsize=12)
    plt.xticks(patients,[f"Subject " + patient_label[i] for i in range(len(patient_label))], rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

    handles = [plt.Rectangle((0,0),1,1,color = colormap[i]) for i in range(len(feature_list_patient))]
    fig.legend(handles, patients, title='Patients', loc = "upper right",  bbox_to_anchor=(1.12,0.8))

    plt.title("Ratio inter/intra de l'entropie",fontsize=16)
    plt.suptitle(suptitle, fontsize=20)
    plt.tight_layout()


def mean_ratio(feature_list_patient: list) -> Tuple[int, int, int, int, int, int]:       ## UTILISER CETTE FONCTION : A VERIFIER !!!
    """ Fonction qui renvoie le ratio inter/intra moyen (moyenne des ratios entre tous les sujets) selon chaque Haralick feature. 

    Args:
        feature_list_patient (list): résultat de la fonction moyenne_angles

    Returns:
        Tuple: contenant le ratio inter/intra moyen (moyenne des ratios entre tous les sujets) selon chaque Haralick feature.
    """
    ratio_nbr = ratio(feature_list_patient)
    
    ratio_contrast = np.mean(ratio_nbr[0])
    ratio_correlation = np.mean(ratio_nbr[1])
    ratio_dissimilarity = np.mean(ratio_nbr[2])
    ratio_homogeneity = np.mean(ratio_nbr[3])
    ratio_ASM = np.mean(ratio_nbr[4])
    ratio_entropy = np.mean(ratio_nbr[5])

    return ratio_contrast, ratio_correlation, ratio_dissimilarity, ratio_homogeneity, ratio_ASM, ratio_entropy

def display_all_six_plot(feature_list : list, patient_label : list, title) -> None:
    """
    This functions displays 6 graphs in 2 rows, 3 columns with curves for all patients in each plot.

    Args : 
        feature_list (list) : A list of lists where each sublist contains the features (contrast, correlation,
                              dissimilarity, homogeneity, ASM, entropy) of a patient; résultat de la fonction moyenne_angles.
        patient_label (list) : A list of strings used to identify each feature across all patients.

    Returns : 
        NoneType : 6 graphs showing the evolution of each feature (contrast, correlation, energy, homogeneity, 
                   ASM, entropy) for all patients as a function of acquisitions.
    """
    plt.figure(figsize=(20,12))

    colormap = ["#69B3A2","indianred","darkgrey","goldenrod","slateblue","olive","steelblue","palevioletred",
                "peru", "darkseagreen", "violet","navy","tan","brown", "salmon", "yellowgreen", "aqua",
                "deepskyblue"]
    title_list = ["Contrast","Correlation","dissimilarity","Homogeneity","ASM", "Entropy"]

    for i in range(6):
        plt.subplot(2,3,i+1)

        for j, patient_feature in enumerate(feature_list):
            plt.plot(patient_feature[i], marker = 'o', color = colormap[j%len(colormap)], label = patient_label[j])
        
        plt.title(title_list[i], fontsize = 18)
        plt.grid(True)
        plt.xlabel("Acquisition", fontsize = 14)
        plt.ylabel("Feature", fontsize = 14)
        
    plt.legend(fontsize=12, bbox_to_anchor=(1.12,0.5), loc='lower left')
    plt.suptitle(title, fontsize = 20)
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    plt.show()


# 3. Functions that calculate the reproducibility of methods using RMSCV and CV, considering the median instead of mean. 

def find_file(path: str, name_file:str) -> np.ndarray :     ## UTILISER CETTE FONCTION
    """ 
    This function allows you to find a file using a path that allows you to access it and the name of the file;  

    Args:
        path (str): From the file explorer, find the file to open, right-click on it, go to "properties", copy what is written in "location" and paste it into path. IMPORTANT : Put r before the the path e.g r"C:\..." 
        name_file (str): name of the file; don't forget to write the entire name of the file, with the nrrd include, e.g "IH1 VF T  TP2.nrrd" and not only "IH1 VF T  TP2". 
        CAREFUL !! Don't forget to put "" around the path and around name_file parameter e.g below. 
    
    Returns:
        np.ndarray: the file that you want to open is an ultrasound image or a segmentation. 
    """
    full_path = os.path.join(path, name_file)
    echo, _ = nrrd.read(full_path)

    return echo 


# Import des échographies et segmentation de l'opératrice experte VF : acquisition 1 série a et b, acquisition 2 : CHANGE FIND_FILE PARAMETERS TO FIND THE FILE FROM YOUR PC

echo_IH1_acq1_seriea_vf = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\IH1c\IH1 VF TP1","IH1 VF T  TP2.nrrd") # Echographie de l'acquisition 1, série a 
seg_IH1_acq1_seriea_vf  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\IH1c\IH1 VF TP1","Segmentation.seg.nrrd") # Segmentation associée à l'acquisition 1, série a 
seg_IH1_acq1_serieb_vf  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\IH1c\IH1 VF TP1b","Segmentation.IH1vfTP1b.nrrd")   # Segmentation associée à l'acquisition 1, série b 

echo_IH1_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/IH1c/IH1 VF TP2","IH1 VF T  TP.nrrd")  # Echographie de l'acquisition 2, série a (une seule série disponible => pas de série b)
seg_IH1_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/IH1c/IH1 VF TP2","Segmentation.seg.nrrd")  # Segmentation associée à l'acquisition 2, série a (une seule série disponible => pas de série b)


echo_CB2_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CB2c/CB2 VF TP1","CB2 VF T TP.nrrd")
seg_CB2_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CB2c/CB2 VF TP1","Segmentation.seg.nrrd")
seg_CB2_acq1_serieb_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CB2c/CB2 VF TP1","Segmentation.CB2vfTp1b.nrrd")

echo_CB2_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CB2c/CB2 VF TP2","CB2 VF T TP2.nrrd")
seg_CB2_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CB2c/CB2 VF TP2","Segmentation.seg.nrrd")


echo_HT1_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/HT1c/HT1 VF TP1","HT1 VF TTP.nrrd")
seg_HT1_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/HT1c/HT1 VF TP1","Segmentation.seg.nrrd")
seg_HT1_acq1_serieb_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/HT1c/HT1 VF TP1","Segmentation.HT1vfTP1b.nrrd")

echo_HT1_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/HT1c/HT1 VF TP2","HT1 VF TTP2.nrrd")
seg_HT1_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/HT1c/HT1 VF TP2","Segmentation.seg.nrrd")


echo_CA1_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CA1c/CA1 VF TP1","CA1 VF T TP.nrrd")
seg_CA1_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CA1c/CA1 VF TP1","Segmentation.seg.nrrd")
seg_CA1_acq1_serieb_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CA1c/CA1 VF TP1","Segmentation.CA1vfTP2b.nrrd")

echo_CA1_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CA1c/CA1 VF TP2","CA1 VF T TP2.nrrd")
seg_CA1_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CA1c/CA1 VF TP2","Segmentation.seg.nrrd")


echo_ET1_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/ET1c/ET1 VF TP1","ET1 VF T TP.nrrd")
seg_ET1_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/ET1c/ET1 VF TP1","Segmentation.seg.nrrd")
seg_ET1_acq1_serieb_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/ET1c/ET1 VF TP1","Segmentation.ET1vfTP1b.nrrd")

echo_ET1_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/ET1c/ET1 VF TP2","ET1 VF TTP2.nrrd")
seg_ET1_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/ET1c/ET1 VF TP2","Segmentation_1.seg.nrrd")


echo_SLT_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SLTc/SLT VF TP1","SLT VF T TP.nrrd")
seg_SLT_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SLTc/SLT VF TP1","Segmentation.seg.nrrd")
seg_SLT_acq1_serieb_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SLTc/SLT VF TP1","Segmentation.SLTvfTP1b.nrrd")

echo_SLT_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SLTc/SLT VF TP2","SLT VF T TP2.nrrd")
seg_SLT_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SLTc/SLT VF TP2","Segmentation.seg.nrrd")


echo_JH1_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/JH1c/JH1 VF TP1","JH1 VF T TP.nrrd")
seg_JH1_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/JH1c/JH1 VF TP1","Segmentation.seg.nrrd")
seg_JH1_acq1_serieb_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/JH1c/JH1 VF TP1","Segmentation.JH1vfTP1b.nrrd")

echo_JH1_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/JH1c/JH1 VF TP2","JH1 VF T TP2.nrrd")
seg_JH1_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/JH1c/JH1 VF TP2","Segmentation.seg.nrrd")


echo_MK1_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/MK1c/MK1 VF TP1","MK1 VF T TP.nrrd")
seg_MK1_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/MK1c/MK1 VF TP1","Segmentation.seg.nrrd")
seg_MK1_acq1_serieb_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/MK1c/MK1 VF TP1","Segmentation.MK1vfTP1b.nrrd")

echo_MK1_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/MK1c/MK1 VF TP2","MK1 VF T TP2.nrrd")
seg_MK1_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/MK1c/MK1 VF TP2","Segmentation.seg.nrrd")


echo_SR1_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SR1c/SR1 VF TP1","SR1 VF T TP.nrrd")
seg_SR1_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SR1c/SR1 VF TP1","Segmentation.seg.nrrd")
seg_SR1_acq1_serieb_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SR1c/SR1 VF TP1","Segmentation.SR1vfTP1b.nrrd")

echo_SR1_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SR1c/SR1 VF TP2","SR1 VF T TP2.nrrd")
seg_SR1_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SR1c/SR1 VF TP2","Segmentation.seg.nrrd")


echo_AV1_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AV1c/AV1 VF TP1","AV1 VF T TP.nrrd")
seg_AV1_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AV1c/AV1 VF TP1","Segmentation.seg.nrrd")
seg_AV1_acq1_serieb_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AV1c/AV1 VF TP1","Segmentation.AV1vfTP1b.nrrd")

echo_AV1_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AV1c/AV1 VF TP2","AV1 VF T TP2.nrrd")
seg_AV1_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AV1c/AV1 VF TP2","Segmentation.seg.nrrd")


echo_AB1_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AB1/AB1 VF TPpost","AB1VFTposTP.nrrd")
seg_AB1_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AB1/AB1 VF TPpost","Segmentation.seg.nrrd")
seg_AB1_acq1_serieb_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AB1/AB1 VF TPpost","Segmentation_AB1vfTP1b.seg.nrrd")

echo_AB1_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AB1/AB1 VF TP2","AB1 VF T TP_2.nrrd")
seg_AB1_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AB1/AB1 VF TP2","Segmentation.seg.nrrd")


echo_AB2_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AB2c/AB2 VF TP1","AB2 VF T TP.nrrd")
seg_AB2_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AB2c/AB2 VF TP1","Segmentation.seg.nrrd")
seg_AB2_acq1_serieb_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AB2c/AB2 VF TP1","Segmentation.sAB2vfTP1b.nrrd")

echo_AB2_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AB2c/AB2 VF TP2","AB2 VF T TP2.nrrd")
seg_AB2_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AB2c/AB2 VF TP2","Segmentation.seg.nrrd")


echo_SB1_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SB1c/SB1 VF TP1","SB1 VF TTP.nrrd")
seg_SB1_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SB1c/SB1 VF TP1","Segmentation.seg.nrrd")
seg_SB1_acq1_serieb_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SB1c/SB1 VF TP1","Segmentation.SB1vfTP1b.nrrd")

echo_SB1_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SB1c/SB1 VF TP2","SB1 VF TTP2.nrrd")
seg_SB1_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SB1c/SB1 VF TP2","Segmentation.seg.nrrd")


echo_FB1_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/FB1c/FB1 VF TP1","FB1 VF T- TP.nrrd")
seg_FB1_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/FB1c/FB1 VF TP1","Segmentation.seg.nrrd")
seg_FB1_acq1_serieb_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/FB1c/FB1 VF TP1","Segmentation.FB1vfTP1b.nrrd")

echo_FB1_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/FB1c/FB1 VF TP2","FB1 VF T- TP2.nrrd")
seg_FB1_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/FB1c/FB1 VF TP2","Segmentation.seg.nrrd")


echo_CS1_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CS1c/CS1 VF TP1","CS1 VF T TP.nrrd")
seg_CS1_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CS1c/CS1 VF TP1","Segmentation.seg.nrrd")
seg_CS1_acq1_serieb_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CS1c/CS1 VF TP1","Segmentation.CS1vfTP1b.nrrd")

echo_CS1_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CS1c/CS1 VF TP2","CS1 VF T TP2.nrrd")
seg_CS1_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CS1c/CS1 VF TP2","Segmentation.seg.nrrd")


echo_CB1_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CB1c/CB1 VF TP1","CB1 VF T TP.nrrd")
seg_CB1_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CB1c/CB1 VF TP1","Segmentation.seg.nrrd")
seg_CB1_acq1_serieb_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CB1c/CB1 VF TP1","SegmentationCB1vfTP1b.nrrd")

echo_CB1_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CB1c/CB1 VF TP2","CB1 VF T TP_2.nrrd")
seg_CB1_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CB1c/CB1 VF TP2","Segmentation.seg.nrrd")


echo_PG_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/PGc/PG VF TP1","PG VF T TP.nrrd")
seg_PG_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/PGc/PG VF TP1","Segmentation.seg.nrrd")
seg_PG_acq1_serieb_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/PGc/PG VF TP1","Segmentation.PGvfTp1b.nrrd")

echo_PG_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/PGc/PG VF TP2","PG VF T TP2.nrrd")
seg_PG_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/PGc/PG VF TP2","Segmentation.seg.nrrd")


echo_YG_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/YGc/YG VF TP1","YG VF TTP.nrrd")
seg_YG_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/YGc/YG VF TP1","Segmentation.seg.nrrd")
seg_YG_acq1_serieb_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/YGc/YG VF TP1","Segmentation.YGvfTP1b.nrrd")

echo_YG_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/YGc/YG VF TP2","YG VF TTP2.nrrd")
seg_YG_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/YGc/YG VF TP2","Segmentation.seg.nrrd")


echo_IR_acq1_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/IRc/IR VF TP1","IR VF T TP.nrrd")
seg_IR_acq1_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/IRc/IR VF TP1","Segmentation.seg.nrrd")
seg_IR_acq1_serieb_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/IRc/IR VF TP1","Segmentation.IRvfTP1b.nrrd")

echo_IR_acq2_seriea_vf = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/IRc/IR VF TP2","IR VF T TP2.nrrd")
seg_IR_acq2_seriea_vf  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/IRc/IR VF TP2","Segmentation.seg.nrrd")

# Fonctions de traitement pour tous les sujets : statistiques d'ordre 2 
def GLCM_acq1_2_VF_tendon(distance_liste, angle_liste, parameter_based):
    """ 
    Fonction qui prend en compte les acquisitions 1, série a et acquisition 2, série a et qui renvoie une liste L tel que 
    - L[i] i ∈ [1,(nbr dans la liste de paramètre)*nbr de sujets traités = (nbr dans la liste de paramètre)*19], donc si distance_liste = [1,2], i ∈ [1,2*19=38], L[i] aura les caractéristiques pour les 19 sujets 
    pour la distance = 1, puis les caractéristiques pour les 19 sujets pour la distance = 2. 
    - L[i][j] j ∈ [1,6], où j correspond aux Haralick features dans l'ordre : Contraste, corrélation, dissimilarité, homogéneité, ASM, entropy. 
    - L[i][j][k], k ∈ [1, nbr de coupe du volume] : correspond concrètement à la valeur du feature j, pour le sujet i, de la coupe k du volume 3D. 
    """
    subjects = [
        ("IH1", echo_IH1_acq1_seriea_vf, seg_IH1_acq1_seriea_vf, 259, 339, echo_IH1_acq2_seriea_vf, seg_IH1_acq2_seriea_vf, 239, 319),
        ("CB2", echo_CB2_acq1_seriea_vf, seg_CB2_acq1_seriea_vf, 204, 304, echo_CB2_acq2_seriea_vf, seg_CB2_acq2_seriea_vf, 224, 324),
        ("HT1", echo_HT1_acq1_seriea_vf, seg_HT1_acq1_seriea_vf, 258, 348, echo_HT1_acq2_seriea_vf, seg_HT1_acq2_seriea_vf, 253, 343),
        ("CA1", echo_CA1_acq1_seriea_vf, seg_CA1_acq1_seriea_vf, 240, 340, echo_CA1_acq2_seriea_vf, seg_CA1_acq2_seriea_vf, 210, 310),
        ("ET1", echo_ET1_acq1_seriea_vf, seg_ET1_acq1_seriea_vf, 230, 320, echo_ET1_acq2_seriea_vf, seg_ET1_acq2_seriea_vf, 227, 317),
        ("SLT", echo_SLT_acq1_seriea_vf, seg_SLT_acq1_seriea_vf, 230, 330, echo_SLT_acq2_seriea_vf, seg_SLT_acq2_seriea_vf, 200, 300),
        ("JH1", echo_JH1_acq1_seriea_vf, seg_JH1_acq1_seriea_vf, 232, 332, echo_JH1_acq2_seriea_vf, seg_JH1_acq2_seriea_vf, 233, 333),
        ("MK1", echo_MK1_acq1_seriea_vf, seg_MK1_acq1_seriea_vf, 172, 262, echo_MK1_acq2_seriea_vf, seg_MK1_acq2_seriea_vf, 198, 288),
        ("SR1", echo_SR1_acq1_seriea_vf, seg_SR1_acq1_seriea_vf, 215, 305, echo_SR1_acq2_seriea_vf, seg_SR1_acq2_seriea_vf, 230, 320),
        ("AV1", echo_AV1_acq1_seriea_vf, seg_AV1_acq1_seriea_vf, 220, 310, echo_AV1_acq2_seriea_vf, seg_AV1_acq2_seriea_vf, 245, 335),
        ("AB1", echo_AB1_acq1_seriea_vf, seg_AB1_acq1_seriea_vf, 200, 300, echo_AB1_acq2_seriea_vf, seg_AB1_acq2_seriea_vf, 189, 289),
        ("AB2", echo_AB2_acq1_seriea_vf, seg_AB2_acq1_seriea_vf, 210, 310, echo_AB2_acq2_seriea_vf, seg_AB2_acq2_seriea_vf, 225, 325),
        ("SB1", echo_SB1_acq1_seriea_vf, seg_SB1_acq1_seriea_vf, 225, 315, echo_SB1_acq2_seriea_vf, seg_SB1_acq2_seriea_vf, 230, 320),
        ("FB1", echo_FB1_acq1_seriea_vf, seg_FB1_acq1_seriea_vf, 260, 360, echo_FB1_acq2_seriea_vf, seg_FB1_acq2_seriea_vf, 240, 340),
        ("CS1", echo_CS1_acq1_seriea_vf, seg_CS1_acq1_seriea_vf, 258, 358, echo_CS1_acq2_seriea_vf, seg_CS1_acq2_seriea_vf, 263, 363),
        ("CB1", echo_CB1_acq1_seriea_vf, seg_CB1_acq1_seriea_vf, 240, 330, echo_CB1_acq2_seriea_vf, seg_CB1_acq2_seriea_vf, 265, 355),
        ("PG",  echo_PG_acq1_seriea_vf,  seg_PG_acq1_seriea_vf,  190, 290, echo_PG_acq2_seriea_vf,  seg_PG_acq2_seriea_vf,  213, 313),
        ("YG",  echo_YG_acq1_seriea_vf,  seg_YG_acq1_seriea_vf,  245, 345, echo_YG_acq2_seriea_vf,  seg_YG_acq2_seriea_vf,  225, 325),
        ("IR",  echo_IR_acq1_seriea_vf,  seg_IR_acq1_seriea_vf,  255, 355, echo_IR_acq2_seriea_vf,  seg_IR_acq2_seriea_vf,  255, 355),
    ]

    haralick_feature_acq1 = []
    haralick_feature_acq2 = []

    # Choix de la liste d’itération
    if parameter_based == "distance":
        iterable = distance_liste
        is_angle = False
    elif parameter_based == "angle":
        iterable = angle_liste
        is_angle = True
    else:
        raise ValueError("parameter_based must be 'distance' or 'angle'")

    for param in iterable:
        for subj in subjects:
            name, e1, s1, x1, y1, e2, s2, x2, y2 = subj
            if is_angle:
                res_acq1, res_acq2, _ = GLCM_treatement_tendon_3D(
                    e1, s1, x1, y1, e2, s2, x2, y2,
                    distance_liste, param, name
                )
            else:
                res_acq1, res_acq2, _ = GLCM_treatement_tendon_3D(
                    e1, s1, x1, y1, e2, s2, x2, y2,
                    param, angle_liste, name
                )
            haralick_feature_acq1.append(res_acq1)
            haralick_feature_acq2.append(res_acq2)

    return haralick_feature_acq1, haralick_feature_acq2



def GLCM_acq1_seriea_b_VF_tendon(distance_liste, angle_liste, parameter_based, acq):      ## UTILISER CETTE FONCTION
    """ 
    Fonction qui prend en compte les acquisitions 1, série a et acquisition 1, série b et qui renvoie une liste L tel que 
    - L[i] i ∈ [1,(nbr dans la liste de paramètre)*nbr de sujets traités = (nbr dans la liste de paramètre)*19], donc si distance_liste = [1,2], i ∈ [1,2*19=38], L[i] aura les caractéristiques pour les 19 sujets 
    pour la distance = 1, puis les caractéristiques pour les 19 sujets pour la distance = 2. 
    - L[i][j] j ∈ [1,6], où j correspond aux Haralick features dans l'ordre : Contraste, corrélation, dissimilarité, homogéneité, ASM, entropy. 
    - L[i][j][k], k ∈ [1, nbr de coupe du volume] : correspond concrètement à la valeur moyenne entre l'acquisition 1 série a et l'acquisition 1 série b du feature j, pour le sujet i, de la coupe k du volume 3D. 
    """
    haralick_feature = []

    if parameter_based == "distance" and acq =="acq1": 
        for distance in distance_liste : 

            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IH1_acq1_seriea_vf, seg_IH1_acq1_seriea_vf, 259, 339, echo_IH1_acq2_seriea_vf, seg_IH1_acq2_seriea_vf, 259, 339, distance, angle_liste, "IH1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB2_acq1_seriea_vf, seg_CB2_acq1_seriea_vf, 204, 304, echo_CB2_acq2_seriea_vf, seg_CB2_acq2_seriea_vf, 204, 304, distance, angle_liste, "CB2")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_HT1_acq1_seriea_vf, seg_HT1_acq1_seriea_vf, 258, 348, echo_HT1_acq2_seriea_vf, seg_HT1_acq2_seriea_vf, 258, 348, distance, angle_liste, "HT1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CA1_acq1_seriea_vf, seg_CA1_acq1_seriea_vf, 240, 340, echo_CA1_acq2_seriea_vf, seg_CA1_acq2_seriea_vf, 240, 340, distance, angle_liste, "CA1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_ET1_acq1_seriea_vf, seg_ET1_acq1_seriea_vf, 230, 320, echo_ET1_acq2_seriea_vf, seg_ET1_acq2_seriea_vf, 230, 320, distance, angle_liste, "ET1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SLT_acq1_seriea_vf, seg_SLT_acq1_seriea_vf, 230, 330, echo_SLT_acq2_seriea_vf, seg_SLT_acq2_seriea_vf, 230, 330, distance, angle_liste, "SLT")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_JH1_acq1_seriea_vf, seg_JH1_acq1_seriea_vf, 232, 332, echo_JH1_acq2_seriea_vf, seg_JH1_acq2_seriea_vf, 232, 332, distance, angle_liste, "JH1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_MK1_acq1_seriea_vf, seg_MK1_acq1_seriea_vf, 172, 262, echo_MK1_acq2_seriea_vf, seg_MK1_acq2_seriea_vf, 172, 262, distance, angle_liste, "MK1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SR1_acq1_seriea_vf, seg_SR1_acq1_seriea_vf, 215, 305, echo_SR1_acq2_seriea_vf, seg_SR1_acq2_seriea_vf, 215, 305, distance, angle_liste, "SR1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AV1_acq1_seriea_vf, seg_AV1_acq1_seriea_vf, 220, 310, echo_AV1_acq2_seriea_vf, seg_AV1_acq2_seriea_vf, 220, 310, distance, angle_liste, "SR1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB1_acq1_seriea_vf, seg_AB1_acq1_seriea_vf, 200, 300, echo_AB1_acq2_seriea_vf, seg_AB1_acq2_seriea_vf, 200, 300, distance, angle_liste, "AB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB2_acq1_seriea_vf, seg_AB2_acq1_seriea_vf, 210, 310, echo_AB2_acq2_seriea_vf, seg_AB2_acq2_seriea_vf, 210, 310, distance, angle_liste, "AB2")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SB1_acq1_seriea_vf, seg_SB1_acq1_seriea_vf, 225, 315, echo_SB1_acq2_seriea_vf, seg_SB1_acq2_seriea_vf, 225, 315, distance, angle_liste, "SB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_FB1_acq1_seriea_vf, seg_FB1_acq1_seriea_vf, 260, 360, echo_FB1_acq2_seriea_vf, seg_FB1_acq2_seriea_vf, 260, 360, distance, angle_liste, "FB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CS1_acq1_seriea_vf, seg_CS1_acq1_seriea_vf, 258, 358, echo_CS1_acq2_seriea_vf, seg_CS1_acq2_seriea_vf, 258, 358, distance, angle_liste, "CS1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB1_acq1_seriea_vf, seg_CB1_acq1_seriea_vf, 240, 330, echo_CB1_acq2_seriea_vf, seg_CB1_acq2_seriea_vf, 240, 330, distance, angle_liste, "CB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_PG_acq1_seriea_vf,  seg_PG_acq1_seriea_vf,  190, 290, echo_PG_acq2_seriea_vf,  seg_PG_acq2_seriea_vf,  190, 290, distance, angle_liste, "PG")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_YG_acq1_seriea_vf,  seg_YG_acq1_seriea_vf,  245, 345, echo_YG_acq2_seriea_vf,  seg_YG_acq2_seriea_vf,  245, 345, distance, angle_liste, "YG")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IR_acq1_seriea_vf,  seg_IR_acq1_seriea_vf,  255, 355, echo_IR_acq2_seriea_vf,  seg_IR_acq2_seriea_vf,  255, 355, distance, angle_liste, "IR")[0])

    if parameter_based == "distance" and acq =="acq2": 
        for distance in distance_liste : 

            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IH1_acq1_seriea_vf, seg_IH1_acq1_seriea_vf, 259, 339, echo_IH1_acq2_seriea_vf, seg_IH1_acq2_seriea_vf, 259, 339, distance, angle_liste, "IH1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB2_acq1_seriea_vf, seg_CB2_acq1_seriea_vf, 204, 304, echo_CB2_acq2_seriea_vf, seg_CB2_acq2_seriea_vf, 204, 304, distance, angle_liste, "CB2")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_HT1_acq1_seriea_vf, seg_HT1_acq1_seriea_vf, 258, 348, echo_HT1_acq2_seriea_vf, seg_HT1_acq2_seriea_vf, 258, 348, distance, angle_liste, "HT1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CA1_acq1_seriea_vf, seg_CA1_acq1_seriea_vf, 240, 340, echo_CA1_acq2_seriea_vf, seg_CA1_acq2_seriea_vf, 240, 340, distance, angle_liste, "CA1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_ET1_acq1_seriea_vf, seg_ET1_acq1_seriea_vf, 230, 320, echo_ET1_acq2_seriea_vf, seg_ET1_acq2_seriea_vf, 230, 320, distance, angle_liste, "ET1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SLT_acq1_seriea_vf, seg_SLT_acq1_seriea_vf, 230, 330, echo_SLT_acq2_seriea_vf, seg_SLT_acq2_seriea_vf, 230, 330, distance, angle_liste, "SLT")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_JH1_acq1_seriea_vf, seg_JH1_acq1_seriea_vf, 232, 332, echo_JH1_acq2_seriea_vf, seg_JH1_acq2_seriea_vf, 232, 332, distance, angle_liste, "JH1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_MK1_acq1_seriea_vf, seg_MK1_acq1_seriea_vf, 172, 262, echo_MK1_acq2_seriea_vf, seg_MK1_acq2_seriea_vf, 172, 262, distance, angle_liste, "MK1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SR1_acq1_seriea_vf, seg_SR1_acq1_seriea_vf, 215, 305, echo_SR1_acq2_seriea_vf, seg_SR1_acq2_seriea_vf, 215, 305, distance, angle_liste, "SR1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AV1_acq1_seriea_vf, seg_AV1_acq1_seriea_vf, 220, 310, echo_AV1_acq2_seriea_vf, seg_AV1_acq2_seriea_vf, 220, 310, distance, angle_liste, "SR1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB1_acq1_seriea_vf, seg_AB1_acq1_seriea_vf, 200, 300, echo_AB1_acq2_seriea_vf, seg_AB1_acq2_seriea_vf, 200, 300, distance, angle_liste, "AB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB2_acq1_seriea_vf, seg_AB2_acq1_seriea_vf, 210, 310, echo_AB2_acq2_seriea_vf, seg_AB2_acq2_seriea_vf, 210, 310, distance, angle_liste, "AB2")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SB1_acq1_seriea_vf, seg_SB1_acq1_seriea_vf, 225, 315, echo_SB1_acq2_seriea_vf, seg_SB1_acq2_seriea_vf, 225, 315, distance, angle_liste, "SB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_FB1_acq1_seriea_vf, seg_FB1_acq1_seriea_vf, 260, 360, echo_FB1_acq2_seriea_vf, seg_FB1_acq2_seriea_vf, 260, 360, distance, angle_liste, "FB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CS1_acq1_seriea_vf, seg_CS1_acq1_seriea_vf, 258, 358, echo_CS1_acq2_seriea_vf, seg_CS1_acq2_seriea_vf, 258, 358, distance, angle_liste, "CS1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB1_acq1_seriea_vf, seg_CB1_acq1_seriea_vf, 240, 330, echo_CB1_acq2_seriea_vf, seg_CB1_acq2_seriea_vf, 240, 330, distance, angle_liste, "CB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_PG_acq1_seriea_vf,  seg_PG_acq1_seriea_vf,  190, 290, echo_PG_acq2_seriea_vf,  seg_PG_acq2_seriea_vf,  190, 290, distance, angle_liste, "PG")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_YG_acq1_seriea_vf,  seg_YG_acq1_seriea_vf,  245, 345, echo_YG_acq2_seriea_vf,  seg_YG_acq2_seriea_vf,  245, 345, distance, angle_liste, "YG")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IR_acq1_seriea_vf,  seg_IR_acq1_seriea_vf,  255, 355, echo_IR_acq2_seriea_vf,  seg_IR_acq2_seriea_vf,  255, 355, distance, angle_liste, "IR")[1])

    if parameter_based == "angle" and acq =="acq1": 
        for angle in angle_liste : 
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IH1_acq1_seriea_vf, seg_IH1_acq1_seriea_vf, 259, 339, echo_IH1_acq2_seriea_vf, seg_IH1_acq2_seriea_vf, 259, 339, distance_liste, angle, "IH1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB2_acq1_seriea_vf, seg_CB2_acq1_seriea_vf, 204, 304, echo_CB2_acq2_seriea_vf, seg_CB2_acq2_seriea_vf, 204, 304, distance_liste, angle, "CB2")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_HT1_acq1_seriea_vf, seg_HT1_acq1_seriea_vf, 258, 348, echo_HT1_acq2_seriea_vf, seg_HT1_acq2_seriea_vf, 258, 348, distance_liste, angle, "HT1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CA1_acq1_seriea_vf, seg_CA1_acq1_seriea_vf, 240, 340, echo_CA1_acq2_seriea_vf, seg_CA1_acq2_seriea_vf, 240, 340, distance_liste, angle, "CA1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_ET1_acq1_seriea_vf, seg_ET1_acq1_seriea_vf, 230, 320, echo_ET1_acq2_seriea_vf, seg_ET1_acq2_seriea_vf, 230, 320, distance_liste, angle, "ET1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SLT_acq1_seriea_vf, seg_SLT_acq1_seriea_vf, 230, 330, echo_SLT_acq2_seriea_vf, seg_SLT_acq2_seriea_vf, 230, 330, distance_liste, angle, "SLT")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_JH1_acq1_seriea_vf, seg_JH1_acq1_seriea_vf, 232, 332, echo_JH1_acq2_seriea_vf, seg_JH1_acq2_seriea_vf, 232, 332, distance_liste, angle, "JH1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_MK1_acq1_seriea_vf, seg_MK1_acq1_seriea_vf, 172, 262, echo_MK1_acq2_seriea_vf, seg_MK1_acq2_seriea_vf, 172, 262, distance_liste, angle, "MK1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SR1_acq1_seriea_vf, seg_SR1_acq1_seriea_vf, 215, 305, echo_SR1_acq2_seriea_vf, seg_SR1_acq2_seriea_vf, 215, 305, distance_liste, angle, "SR1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AV1_acq1_seriea_vf, seg_AV1_acq1_seriea_vf, 220, 310, echo_AV1_acq2_seriea_vf, seg_AV1_acq2_seriea_vf, 220, 310, distance_liste, angle, "SR1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB1_acq1_seriea_vf, seg_AB1_acq1_seriea_vf, 200, 300, echo_AB1_acq2_seriea_vf, seg_AB1_acq2_seriea_vf, 200, 300, distance_liste, angle, "AB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB2_acq1_seriea_vf, seg_AB2_acq1_seriea_vf, 210, 310, echo_AB2_acq2_seriea_vf, seg_AB2_acq2_seriea_vf, 210, 310, distance_liste, angle, "AB2")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SB1_acq1_seriea_vf, seg_SB1_acq1_seriea_vf, 225, 315, echo_SB1_acq2_seriea_vf, seg_SB1_acq2_seriea_vf, 225, 315, distance_liste, angle, "SB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_FB1_acq1_seriea_vf, seg_FB1_acq1_seriea_vf, 260, 360, echo_FB1_acq2_seriea_vf, seg_FB1_acq2_seriea_vf, 260, 360, distance_liste, angle, "FB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CS1_acq1_seriea_vf, seg_CS1_acq1_seriea_vf, 258, 358, echo_CS1_acq2_seriea_vf, seg_CS1_acq2_seriea_vf, 258, 358, distance_liste, angle, "CS1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB1_acq1_seriea_vf, seg_CB1_acq1_seriea_vf, 240, 330, echo_CB1_acq2_seriea_vf, seg_CB1_acq2_seriea_vf, 240, 330, distance_liste, angle, "CB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_PG_acq1_seriea_vf,  seg_PG_acq1_seriea_vf,  190, 290, echo_PG_acq2_seriea_vf,  seg_PG_acq2_seriea_vf,  190, 290, distance_liste, angle, "PG")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_YG_acq1_seriea_vf,  seg_YG_acq1_seriea_vf,  245, 345, echo_YG_acq2_seriea_vf,  seg_YG_acq2_seriea_vf,  245, 345, distance_liste, angle, "YG")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IR_acq1_seriea_vf,  seg_IR_acq1_seriea_vf,  255, 355, echo_IR_acq2_seriea_vf,  seg_IR_acq2_seriea_vf,  255, 355, distance_liste, angle, "IR")[0])

    if parameter_based == "angle" and acq =="acq2": 
        for angle in angle_liste : 
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IH1_acq1_seriea_vf, seg_IH1_acq1_seriea_vf, 259, 339, echo_IH1_acq2_seriea_vf, seg_IH1_acq2_seriea_vf, 259, 339, distance_liste, angle, "IH1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB2_acq1_seriea_vf, seg_CB2_acq1_seriea_vf, 204, 304, echo_CB2_acq2_seriea_vf, seg_CB2_acq2_seriea_vf, 204, 304, distance_liste, angle, "CB2")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_HT1_acq1_seriea_vf, seg_HT1_acq1_seriea_vf, 258, 348, echo_HT1_acq2_seriea_vf, seg_HT1_acq2_seriea_vf, 258, 348, distance_liste, angle, "HT1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CA1_acq1_seriea_vf, seg_CA1_acq1_seriea_vf, 240, 340, echo_CA1_acq2_seriea_vf, seg_CA1_acq2_seriea_vf, 240, 340, distance_liste, angle, "CA1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_ET1_acq1_seriea_vf, seg_ET1_acq1_seriea_vf, 230, 320, echo_ET1_acq2_seriea_vf, seg_ET1_acq2_seriea_vf, 230, 320, distance_liste, angle, "ET1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SLT_acq1_seriea_vf, seg_SLT_acq1_seriea_vf, 230, 330, echo_SLT_acq2_seriea_vf, seg_SLT_acq2_seriea_vf, 230, 330, distance_liste, angle, "SLT")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_JH1_acq1_seriea_vf, seg_JH1_acq1_seriea_vf, 232, 332, echo_JH1_acq2_seriea_vf, seg_JH1_acq2_seriea_vf, 232, 332, distance_liste, angle, "JH1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_MK1_acq1_seriea_vf, seg_MK1_acq1_seriea_vf, 172, 262, echo_MK1_acq2_seriea_vf, seg_MK1_acq2_seriea_vf, 172, 262, distance_liste, angle, "MK1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SR1_acq1_seriea_vf, seg_SR1_acq1_seriea_vf, 215, 305, echo_SR1_acq2_seriea_vf, seg_SR1_acq2_seriea_vf, 215, 305, distance_liste, angle, "SR1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AV1_acq1_seriea_vf, seg_AV1_acq1_seriea_vf, 220, 310, echo_AV1_acq2_seriea_vf, seg_AV1_acq2_seriea_vf, 220, 310, distance_liste, angle, "SR1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB1_acq1_seriea_vf, seg_AB1_acq1_seriea_vf, 200, 300, echo_AB1_acq2_seriea_vf, seg_AB1_acq2_seriea_vf, 200, 300, distance_liste, angle, "AB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB2_acq1_seriea_vf, seg_AB2_acq1_seriea_vf, 210, 310, echo_AB2_acq2_seriea_vf, seg_AB2_acq2_seriea_vf, 210, 310, distance_liste, angle, "AB2")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SB1_acq1_seriea_vf, seg_SB1_acq1_seriea_vf, 225, 315, echo_SB1_acq2_seriea_vf, seg_SB1_acq2_seriea_vf, 225, 315, distance_liste, angle, "SB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_FB1_acq1_seriea_vf, seg_FB1_acq1_seriea_vf, 260, 360, echo_FB1_acq2_seriea_vf, seg_FB1_acq2_seriea_vf, 260, 360, distance_liste, angle, "FB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CS1_acq1_seriea_vf, seg_CS1_acq1_seriea_vf, 258, 358, echo_CS1_acq2_seriea_vf, seg_CS1_acq2_seriea_vf, 258, 358, distance_liste, angle, "CS1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB1_acq1_seriea_vf, seg_CB1_acq1_seriea_vf, 240, 330, echo_CB1_acq2_seriea_vf, seg_CB1_acq2_seriea_vf, 240, 330, distance_liste, angle, "CB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_PG_acq1_seriea_vf,  seg_PG_acq1_seriea_vf,  190, 290, echo_PG_acq2_seriea_vf,  seg_PG_acq2_seriea_vf,  190, 290, distance_liste, angle, "PG")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_YG_acq1_seriea_vf,  seg_YG_acq1_seriea_vf,  245, 345, echo_YG_acq2_seriea_vf,  seg_YG_acq2_seriea_vf,  245, 345, distance_liste, angle, "YG")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IR_acq1_seriea_vf,  seg_IR_acq1_seriea_vf,  255, 355, echo_IR_acq2_seriea_vf,  seg_IR_acq2_seriea_vf,  255, 355, distance_liste, angle, "IR")[1])
    
    
    return haralick_feature


# Import des échographies et segmentation de l'opératrice intermédiaire LG : acquisition 1 série a et b, acquisition 2 : CHANGE FIND_FILE PARAMETERS TO FIND THE FILE FROM YOUR PC

echo_IH1_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\IH1c\IH1 LG TP1","IH1 LGT  TP.nrrd") # Echographie de l'acquisition 1, série a 
seg_IH1_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\IH1c\IH1 LG TP1","Segmentation.seg.nrrd") # Segmentation associée à l'acquisition 1, série a 
#seg_IH1_acq1_serieb_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\IH1c\IH1 VF TP1b","Segmentation.IH1vfTP1b.nrrd")   # Segmentation associée à l'acquisition 1, série b 

echo_IH1_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\IH1c\IH1 LG TP2","IH1 LGT  TP2.nrrd")  # Echographie de l'acquisition 2, série a (une seule série disponible => pas de série b)
seg_IH1_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\IH1c\IH1 LG TP2","Segmentation.seg.nrrd")  # Segmentation associée à l'acquisition 2, série a (une seule série disponible => pas de série b)


echo_CB2_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CB2c\CB2 LG TP1","CB2 LG T TP.nrrd")
seg_CB2_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CB2c\CB2 LG TP1","Segmentation.seg.nrrd")
#seg_CB2_acq1_serieb_lg  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CB2c/CB2 VF TP1","Segmentation.CB2vfTp1b.nrrd")

echo_CB2_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CB2c\CB2 LG TP2","CB2 LG T TP2.nrrd")
seg_CB2_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CB2c\CB2 LG TP2","Segmentation.seg.nrrd")


echo_HT1_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\HT1c\HT1 LG TP1","HT1 LG TTP.nrrd")
seg_HT1_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\HT1c\HT1 LG TP1","Segmentation.seg.nrrd")
#seg_HT1_acq1_serieb_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\HT1c\HT1 LG TP1","Segmentation.HT1vfTP1b.nrrd")

echo_HT1_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\HT1c\HT1 LG TP2","HT1 LG TTP2.nrrd")
seg_HT1_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\HT1c\HT1 LG TP2","Segmentation.seg.nrrd")


echo_CA1_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CA1c\CA1 LG TP1","CA1 LG T TP.nrrd")
seg_CA1_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CA1c\CA1 LG TP1","Segmentation.seg.nrrd")
#seg_CA1_acq1_serieb_lg  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CA1c/CA1 VF TP1","Segmentation.CA1vfTP2b.nrrd")

echo_CA1_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CA1c\CA LG TP2","CA1 LG T TP2.nrrd")
seg_CA1_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CA1c\CA LG TP2","Segmentation.seg.nrrd")


echo_ET1_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\ET1c\ET1 LG TP1","ET1 VF T TP.nrrd")
seg_ET1_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\ET1c\ET1 LG TP1","Segmentation.seg.nrrd")
#seg_ET1_acq1_serieb_lg  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/ET1c/ET1 VF TP1","Segmentation.ET1vfTP1b.nrrd")

echo_ET1_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\ET1c\ET1 LG_TP2","ET1 LG T TPb.nrrd")
seg_ET1_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\ET1c\ET1 LG_TP2","Segmentation.seg.nrrd")


echo_SLT_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\SLTc\SLT LG T+TP1","SLT LG T TP.nrrd")
seg_SLT_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\SLTc\SLT LG T+TP1","Segmentation.seg.nrrd")
#seg_SLT_acq1_serieb_lg = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SLTc/SLT VF TP1","Segmentation.SLTvfTP1b.nrrd")

echo_SLT_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\SLTc\SLT LG T+ TP2","SB1 LG T TP2.nrrd")
seg_SLT_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\SLTc\SLT LG T+ TP2","Segmentation.seg.nrrd")


echo_JH1_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\JH1c\JH1 LG TP1","JH1 LG T TP.nrrd")
seg_JH1_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\JH1c\JH1 LG TP1","Segmentation.seg.nrrd")
#seg_JH1_acq1_serieb_lg = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/JH1c/JH1 VF TP1","Segmentation.JH1vfTP1b.nrrd")

echo_JH1_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\JH1c\JH1 LG TP2","JH1 LG T TP2.nrrd")
seg_JH1_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\JH1c\JH1 LG TP2","Segmentation.seg.nrrd")

### corriger a partir de là
echo_MK1_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\MK1c\MK1 LG TP1","MK1 LG T TP.nrrd")
seg_MK1_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\MK1c\MK1 LG TP1","Segmentation.seg.nrrd")
#seg_MK1_acq1_serieb_lg  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/MK1c/MK1 VF TP1","Segmentation.MK1vfTP1b.nrrd")

echo_MK1_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\MK1c\MK1 LG TP2","MK1 LG T TP2.nrrd")
seg_MK1_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\MK1c\MK1 LG TP2","Segmentation.seg.nrrd")


echo_SR1_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\SR1c\SR1 LG TP1","SR1 LG T TP.nrrd")
seg_SR1_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\SR1c\SR1 LG TP1","Segmentation.seg.nrrd")
#seg_SR1_acq1_serieb_lg = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SR1c/SR1 VF TP1","Segmentation.SR1vfTP1b.nrrd")

echo_SR1_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\SR1c\SR1 LG TP2","SR1 LG T TP2.nrrd")
seg_SR1_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\SR1c\SR1 LG TP2","Segmentation.seg.nrrd")


echo_AV1_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\AV1c\AV1 LG TP1","AV1 LG T TP.nrrd")
seg_AV1_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\AV1c\AV1 LG TP1","Segmentation.seg.nrrd")
#seg_AV1_acq1_serieb_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\AV1c\AV1 LG TP1","Segmentation.AV1vfTP1b.nrrd")

echo_AV1_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\AV1c\AV1 LG TP2","AV1 LG T TP2.nrrd")
seg_AV1_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\AV1c\AV1 LG TP2","Segmentation.seg.nrrd")


echo_AB1_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\AB1\AB1 LG TP1","AB1 LG T TP.nrrd")
seg_AB1_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\AB1\AB1 LG TP1","Segmentation.seg.nrrd")
#seg_AB1_acq1_serieb_lg = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AB1/AB1 VF TPpost","Segmentation_AB1vfTP1b.seg.nrrd")

echo_AB1_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\AB1\AB1 LG TP2","AB1 LG T TP_2.nrrd")
seg_AB1_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\AB1\AB1 LG TP2","Segmentation.seg.nrrd")


echo_AB2_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\AB2c\AB2 LG TP1","AB2 LG T TP.nrrd")
seg_AB2_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\AB2c\AB2 LG TP1","Segmentation.seg.nrrd")
#seg_AB2_acq1_serieb_lg = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/AB2c/AB2 VF TP1","Segmentation.sAB2vfTP1b.nrrd")

echo_AB2_acq2_seriea_lg = find_file(r"c:\Users\rmohane\Desktop\base_de_donnees\TP\AB2c\AB2 LG TP2","AB2 LG T TP2.nrrd")
seg_AB2_acq2_seriea_lg  = find_file(r"c:\Users\rmohane\Desktop\base_de_donnees\TP\AB2c\AB2 LG TP2","Segmentation.seg.nrrd")


echo_SB1_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\SB1c\SB1 LG TP1","SB1 LG T TP.nrrd")
seg_SB1_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\SB1c\SB1 LG TP1","Segmentation.seg.nrrd")
#seg_SB1_acq1_serieb_lg = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/SB1c/SB1 VF TP1","Segmentation.SB1vfTP1b.nrrd")

echo_SB1_acq2_seriea_lg = find_file(r"c:\Users\rmohane\Desktop\base_de_donnees\TP\SB1c\SB LG TP2","SB1 LG T TP2.nrrd")
seg_SB1_acq2_seriea_lg  = find_file(r"c:\Users\rmohane\Desktop\base_de_donnees\TP\SB1c\SB LG TP2","Segmentation.seg.nrrd")


echo_FB1_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\FB1c\FB1 LG TP1","FB1 LG T TP.nrrd")
seg_FB1_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\FB1c\FB1 LG TP1","Segmentation.seg.nrrd")
#seg_FB1_acq1_serieb_lg = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/FB1c/FB1 VF TP1","Segmentation.FB1vfTP1b.nrrd")

echo_FB1_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\FB1c\FB1LG TP2","FB1 LG T TP2.nrrd")
seg_FB1_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\FB1c\FB1LG TP2","Segmentation.seg.nrrd")


echo_CS1_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CS1c\CS1 LG TP1","CS1 LG T TP.nrrd")
seg_CS1_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CS1c\CS1 LG TP1","Segmentation.seg.nrrd")
#seg_CS1_acq1_serieb_lg = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CS1c/CS1 VF TP1","Segmentation.CS1vfTP1b.nrrd")

echo_CS1_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CS1c\CS1 LG TP2","CS1 LG T TP2.nrrd")
seg_CS1_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CS1c\CS1 LG TP2","Segmentation.seg.nrrd")


echo_CB1_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CB1c\CB1 LG TP1","CB1 LG T TP.nrrd")
seg_CB1_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CB1c\CB1 LG TP1","Segmentation.seg.nrrd")
#seg_CB1_acq1_serieb_lg = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/CB1c/CB1 VF TP1","SegmentationCB1vfTP1b.nrrd")

echo_CB1_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CB1c\CB1 LG TP2","CB1 LG T TP_2.nrrd")
seg_CB1_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\CB1c\CB1 LG TP2","Segmentation.seg.nrrd")


echo_PG_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\PGc\PG LG TP1","PG LG T SP.nrrd")
seg_PG_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\PGc\PG LG TP1","Segmentation.seg.nrrd")
#seg_PG_acq1_serieb_lg = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/PGc/PG VF TP1","Segmentation.PGvfTp1b.nrrd")

echo_PG_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\PGc\PG LG TP2","PG LG T SP2.nrrd")
seg_PG_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\PGc\PG LG TP2","Segmentation.seg.nrrd")


echo_YG_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\YGc\YG LG TP1","YG LG T TP.nrrd")
seg_YG_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\YGc\YG LG TP1","Segmentation.seg.nrrd")
#seg_YG_acq1_serieb_lg  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/YGc/YG VF TP1","Segmentation.YGvfTP1b.nrrd")

echo_YG_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\YGc\YG LG TP2","YG LG T TP2.nrrd")
seg_YG_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\YGc\YG LG TP2","Segmentation.seg.nrrd")


echo_IR_acq1_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\IRc\IR LG TP1","IR LG T TP.nrrd")
seg_IR_acq1_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\IRc\IR LG TP1","Segmentation.seg.nrrd")
#seg_IR_acq1_serieb_lg  = find_file(r"C:/Users/rmohane/Desktop/base_de_donnees/TP/IRc/IR VF TP1","Segmentation.IRvfTP1b.nrrd")

echo_IR_acq2_seriea_lg = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\IRc\IR LG TP2","IR LG T TP2.nrrd")
seg_IR_acq2_seriea_lg  = find_file(r"C:\Users\rmohane\Desktop\base_de_donnees\TP\IRc\IR LG TP2","Segmentation.seg.nrrd")


def GLCM_acq1_2_LG_tendon(distance_liste, angle_liste, parameter_based, acq):      ## UTILISER CETTE FONCTION : MODIFIABLE
    """ 
    Fonction qui prend en compte les acquisitions 1, série a et acquisition 2, série a et qui renvoie une liste L tel que 
    - L[i] i ∈ [1,(nbr dans la liste de paramètre)*nbr de sujets traités = (nbr dans la liste de paramètre)*19], donc si distance_liste = [1,2], i ∈ [1,2*19=38], L[i] aura les caractéristiques pour les 19 sujets 
    pour la distance = 1, puis les caractéristiques pour les 19 sujets pour la distance = 2. 
    - L[i][j] j ∈ [1,6], où j correspond aux Haralick features dans l'ordre : Contraste, corrélation, dissimilarité, homogéneité, ASM, entropy. 
    - L[i][j][k], k ∈ [1, nbr de coupe du volume] : correspond concrètement à la valeur du feature j, pour le sujet i, de la coupe k du volume 3D. 
    """
    haralick_feature = []

    if parameter_based == "distance" and acq == "acq1": 
        for distance in distance_liste : 

            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IH1_acq1_seriea_lg, seg_IH1_acq1_seriea_lg, 260, 343, echo_IH1_acq2_seriea_lg, seg_IH1_acq2_seriea_lg, 246, 329, distance, angle_liste, "IH1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB2_acq1_seriea_lg, seg_CB2_acq1_seriea_lg, 160, 288, echo_CB2_acq2_seriea_lg, seg_CB2_acq2_seriea_lg, 214, 342, distance, angle_liste, "CB2")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_HT1_acq1_seriea_lg, seg_HT1_acq1_seriea_lg, 258, 353, echo_HT1_acq2_seriea_lg, seg_HT1_acq2_seriea_lg, 248, 343, distance, angle_liste, "HT1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CA1_acq1_seriea_lg, seg_CA1_acq1_seriea_lg, 224,380, echo_CA1_acq2_seriea_lg, seg_CA1_acq2_seriea_lg, 167, 323, distance, angle_liste, "CA1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_ET1_acq1_seriea_lg, seg_ET1_acq1_seriea_lg, 152, 363, echo_ET1_acq2_seriea_lg, seg_ET1_acq2_seriea_lg, 139, 350, distance, angle_liste, "ET1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SLT_acq1_seriea_lg, seg_SLT_acq1_seriea_lg, 188, 365, echo_SLT_acq2_seriea_lg, seg_SLT_acq2_seriea_lg, 164, 341, distance, angle_liste, "SLT")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_JH1_acq1_seriea_lg, seg_JH1_acq1_seriea_lg, 163, 390, echo_JH1_acq2_seriea_lg, seg_JH1_acq2_seriea_lg, 160, 387, distance, angle_liste, "JH1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_MK1_acq1_seriea_lg, seg_MK1_acq1_seriea_lg, 152, 283, echo_MK1_acq2_seriea_lg, seg_MK1_acq2_seriea_lg, 175, 305, distance, angle_liste, "MK1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SR1_acq1_seriea_lg, seg_SR1_acq1_seriea_lg, 184, 348, echo_SR1_acq2_seriea_lg, seg_SR1_acq2_seriea_lg, 196, 360, distance, angle_liste, "SR1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AV1_acq1_seriea_lg, seg_AV1_acq1_seriea_lg, 176, 338, echo_AV1_acq2_seriea_lg, seg_AV1_acq2_seriea_lg, 212, 374, distance, angle_liste, "SR1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB1_acq1_seriea_lg, seg_AB1_acq1_seriea_lg, 247, 312, echo_AB1_acq2_seriea_lg, seg_AB1_acq2_seriea_lg, 191, 383, distance, angle_liste, "AB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB2_acq1_seriea_lg, seg_AB2_acq1_seriea_lg, 211, 377, echo_AB2_acq2_seriea_lg, seg_AB2_acq2_seriea_lg, 213, 379, distance, angle_liste, "AB2")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SB1_acq1_seriea_lg, seg_SB1_acq1_seriea_lg, 185, 406, echo_SB1_acq2_seriea_lg, seg_SB1_acq2_seriea_lg, 160, 381, distance, angle_liste, "SB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_FB1_acq1_seriea_lg, seg_FB1_acq1_seriea_lg, 206, 441, echo_FB1_acq2_seriea_lg, seg_FB1_acq2_seriea_lg, 214, 449, distance, angle_liste, "FB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CS1_acq1_seriea_lg, seg_CS1_acq1_seriea_lg, 260, 462, echo_CS1_acq2_seriea_lg, seg_CS1_acq2_seriea_lg, 173, 375, distance, angle_liste, "CS1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB1_acq1_seriea_lg, seg_CB1_acq1_seriea_lg, 230, 348, echo_CB1_acq2_seriea_lg, seg_CB1_acq2_seriea_lg, 253, 371, distance, angle_liste, "CB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_PG_acq1_seriea_lg,  seg_PG_acq1_seriea_lg,  176, 318, echo_PG_acq2_seriea_lg,  seg_PG_acq2_seriea_lg,  166, 308, distance, angle_liste, "PG")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_YG_acq1_seriea_lg,  seg_YG_acq1_seriea_lg,  206, 357, echo_YG_acq2_seriea_lg,  seg_YG_acq2_seriea_lg,  194, 345, distance, angle_liste, "YG")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IR_acq1_seriea_lg,  seg_IR_acq1_seriea_lg,  233, 399, echo_IR_acq2_seriea_lg,  seg_IR_acq2_seriea_lg,  227, 393, distance, angle_liste, "IR")[0])
    
    if parameter_based == "distance" and acq == "acq2": 
        for distance in distance_liste : 

            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IH1_acq1_seriea_lg, seg_IH1_acq1_seriea_lg, 325, 395, echo_IH1_acq2_seriea_lg, seg_IH1_acq2_seriea_lg, 310, 380, distance, angle_liste, "IH1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB2_acq1_seriea_lg, seg_CB2_acq1_seriea_lg, 275, 375, echo_CB2_acq2_seriea_lg, seg_CB2_acq2_seriea_lg, 305, 405, distance, angle_liste, "CB2")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_HT1_acq1_seriea_lg, seg_HT1_acq1_seriea_lg, 310, 390, echo_HT1_acq2_seriea_lg, seg_HT1_acq2_seriea_lg, 300, 380, distance, angle_liste, "HT1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CA1_acq1_seriea_lg, seg_CA1_acq1_seriea_lg, 290, 390, echo_CA1_acq2_seriea_lg, seg_CA1_acq2_seriea_lg, 220, 320, distance, angle_liste, "CA1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_ET1_acq1_seriea_lg, seg_ET1_acq1_seriea_lg, 290, 390, echo_ET1_acq2_seriea_lg, seg_ET1_acq2_seriea_lg, 220, 320, distance, angle_liste, "ET1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SLT_acq1_seriea_lg, seg_SLT_acq1_seriea_lg, 190, 290, echo_SLT_acq2_seriea_lg, seg_SLT_acq2_seriea_lg, 220, 320, distance, angle_liste, "SLT")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_JH1_acq1_seriea_lg, seg_JH1_acq1_seriea_lg, 250, 350, echo_JH1_acq2_seriea_lg, seg_JH1_acq2_seriea_lg, 250, 350, distance, angle_liste, "JH1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_MK1_acq1_seriea_lg, seg_MK1_acq1_seriea_lg, 240, 340, echo_MK1_acq2_seriea_lg, seg_MK1_acq2_seriea_lg, 230, 330, distance, angle_liste, "MK1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SR1_acq1_seriea_lg, seg_SR1_acq1_seriea_lg, 280, 380, echo_SR1_acq2_seriea_lg, seg_SR1_acq2_seriea_lg, 300, 400, distance, angle_liste, "SR1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AV1_acq1_seriea_lg, seg_AV1_acq1_seriea_lg, 260, 360, echo_AV1_acq2_seriea_lg, seg_AV1_acq2_seriea_lg, 260, 360, distance, angle_liste, "SR1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB1_acq1_seriea_lg, seg_AB1_acq1_seriea_lg, 255, 355, echo_AB1_acq2_seriea_lg, seg_AB1_acq2_seriea_lg, 270, 370, distance, angle_liste, "AB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB2_acq1_seriea_lg, seg_AB2_acq1_seriea_lg, 258, 358, echo_AB2_acq2_seriea_lg, seg_AB2_acq2_seriea_lg, 264, 364, distance, angle_liste, "AB2")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SB1_acq1_seriea_lg, seg_SB1_acq1_seriea_lg, 210, 310, echo_SB1_acq2_seriea_lg, seg_SB1_acq2_seriea_lg, 200, 300, distance, angle_liste, "SB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_FB1_acq1_seriea_lg, seg_FB1_acq1_seriea_lg, 210, 310, echo_FB1_acq2_seriea_lg, seg_FB1_acq2_seriea_lg, 200, 300, distance, angle_liste, "FB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CS1_acq1_seriea_lg, seg_CS1_acq1_seriea_lg, 210, 310, echo_CS1_acq2_seriea_lg, seg_CS1_acq2_seriea_lg, 220, 320, distance, angle_liste, "CS1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB1_acq1_seriea_lg, seg_CB1_acq1_seriea_lg, 160, 260, echo_CB1_acq2_seriea_lg, seg_CB1_acq2_seriea_lg, 162, 262, distance, angle_liste, "CB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_PG_acq1_seriea_lg,  seg_PG_acq1_seriea_lg,  260, 360, echo_PG_acq2_seriea_lg,  seg_PG_acq2_seriea_lg,  253, 353, distance, angle_liste, "PG")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_YG_acq1_seriea_lg,  seg_YG_acq1_seriea_lg,  260, 360, echo_YG_acq2_seriea_lg,  seg_YG_acq2_seriea_lg,  249, 349, distance, angle_liste, "YG")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IR_acq1_seriea_lg,  seg_IR_acq1_seriea_lg,  260, 360, echo_IR_acq2_seriea_lg,  seg_IR_acq2_seriea_lg,  230, 330, distance, angle_liste, "IR")[1])

    if parameter_based == "angle" and acq =="acq1": 
        for angle in angle_liste : 
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IH1_acq1_seriea_lg, seg_IH1_acq1_seriea_lg, 325, 395, echo_IH1_acq2_seriea_lg, seg_IH1_acq2_seriea_lg, 310, 380, distance_liste, angle, "IH1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB2_acq1_seriea_lg, seg_CB2_acq1_seriea_lg, 275, 375, echo_CB2_acq2_seriea_lg, seg_CB2_acq2_seriea_lg, 305, 405, distance_liste, angle, "CB2")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_HT1_acq1_seriea_lg, seg_HT1_acq1_seriea_lg, 310, 390, echo_HT1_acq2_seriea_lg, seg_HT1_acq2_seriea_lg, 300, 380, distance_liste, angle, "HT1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CA1_acq1_seriea_lg, seg_CA1_acq1_seriea_lg, 290, 390, echo_CA1_acq2_seriea_lg, seg_CA1_acq2_seriea_lg, 220, 320, distance_liste, angle, "CA1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_ET1_acq1_seriea_lg, seg_ET1_acq1_seriea_lg, 290, 390, echo_ET1_acq2_seriea_lg, seg_ET1_acq2_seriea_lg, 220, 320, distance_liste, angle, "ET1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SLT_acq1_seriea_lg, seg_SLT_acq1_seriea_lg, 190, 290, echo_SLT_acq2_seriea_lg, seg_SLT_acq2_seriea_lg, 220, 320, distance_liste, angle, "SLT")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_JH1_acq1_seriea_lg, seg_JH1_acq1_seriea_lg, 250, 350, echo_JH1_acq2_seriea_lg, seg_JH1_acq2_seriea_lg, 250, 350, distance_liste, angle, "JH1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_MK1_acq1_seriea_lg, seg_MK1_acq1_seriea_lg, 240, 340, echo_MK1_acq2_seriea_lg, seg_MK1_acq2_seriea_lg, 230, 330, distance_liste, angle, "MK1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SR1_acq1_seriea_lg, seg_SR1_acq1_seriea_lg, 280, 380, echo_SR1_acq2_seriea_lg, seg_SR1_acq2_seriea_lg, 300, 400, distance_liste, angle, "SR1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AV1_acq1_seriea_lg, seg_AV1_acq1_seriea_lg, 260, 360, echo_AV1_acq2_seriea_lg, seg_AV1_acq2_seriea_lg, 260, 360, distance_liste, angle, "SR1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB1_acq1_seriea_lg, seg_AB1_acq1_seriea_lg, 255, 355, echo_AB1_acq2_seriea_lg, seg_AB1_acq2_seriea_lg, 270, 370, distance_liste, angle, "AB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB2_acq1_seriea_lg, seg_AB2_acq1_seriea_lg, 258, 358, echo_AB2_acq2_seriea_lg, seg_AB2_acq2_seriea_lg, 264, 364, distance_liste, angle, "AB2")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SB1_acq1_seriea_lg, seg_SB1_acq1_seriea_lg, 210, 310, echo_SB1_acq2_seriea_lg, seg_SB1_acq2_seriea_lg, 200, 300, distance_liste, angle, "SB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_FB1_acq1_seriea_lg, seg_FB1_acq1_seriea_lg, 210, 310, echo_FB1_acq2_seriea_lg, seg_FB1_acq2_seriea_lg, 200, 300, distance_liste, angle, "FB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CS1_acq1_seriea_lg, seg_CS1_acq1_seriea_lg, 210, 310, echo_CS1_acq2_seriea_lg, seg_CS1_acq2_seriea_lg, 220, 320, distance_liste, angle, "CS1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB1_acq1_seriea_lg, seg_CB1_acq1_seriea_lg, 160, 260, echo_CB1_acq2_seriea_lg, seg_CB1_acq2_seriea_lg, 162, 262, distance_liste, angle, "CB1")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_PG_acq1_seriea_lg,  seg_PG_acq1_seriea_lg,  260, 360, echo_PG_acq2_seriea_lg,  seg_PG_acq2_seriea_lg,  253, 353, distance_liste, angle, "PG")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_YG_acq1_seriea_lg,  seg_YG_acq1_seriea_lg,  260, 360, echo_YG_acq2_seriea_lg,  seg_YG_acq2_seriea_lg,  249, 349, distance_liste, angle, "YG")[0])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IR_acq1_seriea_lg,  seg_IR_acq1_seriea_lg,  260, 360, echo_IR_acq2_seriea_lg,  seg_IR_acq2_seriea_lg,  230, 330, distance_liste, angle, "IR")[0])
            
    if parameter_based == "angle" and acq =="acq2": 
        for angle in angle_liste : 
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IH1_acq1_seriea_lg, seg_IH1_acq1_seriea_lg, 325, 395, echo_IH1_acq2_seriea_lg, seg_IH1_acq2_seriea_lg, 310, 380, distance_liste, angle, "IH1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB2_acq1_seriea_lg, seg_CB2_acq1_seriea_lg, 275, 375, echo_CB2_acq2_seriea_lg, seg_CB2_acq2_seriea_lg, 305, 405, distance_liste, angle, "CB2")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_HT1_acq1_seriea_lg, seg_HT1_acq1_seriea_lg, 310, 390, echo_HT1_acq2_seriea_lg, seg_HT1_acq2_seriea_lg, 300, 380, distance_liste, angle, "HT1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CA1_acq1_seriea_lg, seg_CA1_acq1_seriea_lg, 290, 390, echo_CA1_acq2_seriea_lg, seg_CA1_acq2_seriea_lg, 220, 320, distance_liste, angle, "CA1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_ET1_acq1_seriea_lg, seg_ET1_acq1_seriea_lg, 290, 390, echo_ET1_acq2_seriea_lg, seg_ET1_acq2_seriea_lg, 220, 320, distance_liste, angle, "ET1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SLT_acq1_seriea_lg, seg_SLT_acq1_seriea_lg, 190, 290, echo_SLT_acq2_seriea_lg, seg_SLT_acq2_seriea_lg, 220, 320, distance_liste, angle, "SLT")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_JH1_acq1_seriea_lg, seg_JH1_acq1_seriea_lg, 250, 350, echo_JH1_acq2_seriea_lg, seg_JH1_acq2_seriea_lg, 250, 350, distance_liste, angle, "JH1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_MK1_acq1_seriea_lg, seg_MK1_acq1_seriea_lg, 240, 340, echo_MK1_acq2_seriea_lg, seg_MK1_acq2_seriea_lg, 230, 330, distance_liste, angle, "MK1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SR1_acq1_seriea_lg, seg_SR1_acq1_seriea_lg, 280, 380, echo_SR1_acq2_seriea_lg, seg_SR1_acq2_seriea_lg, 300, 400, distance_liste, angle, "SR1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AV1_acq1_seriea_lg, seg_AV1_acq1_seriea_lg, 260, 360, echo_AV1_acq2_seriea_lg, seg_AV1_acq2_seriea_lg, 260, 360, distance_liste, angle, "SR1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB1_acq1_seriea_lg, seg_AB1_acq1_seriea_lg, 255, 355, echo_AB1_acq2_seriea_lg, seg_AB1_acq2_seriea_lg, 270, 370, distance_liste, angle, "AB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_AB2_acq1_seriea_lg, seg_AB2_acq1_seriea_lg, 258, 358, echo_AB2_acq2_seriea_lg, seg_AB2_acq2_seriea_lg, 264, 364, distance_liste, angle, "AB2")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_SB1_acq1_seriea_lg, seg_SB1_acq1_seriea_lg, 210, 310, echo_SB1_acq2_seriea_lg, seg_SB1_acq2_seriea_lg, 200, 300, distance_liste, angle, "SB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_FB1_acq1_seriea_lg, seg_FB1_acq1_seriea_lg, 210, 310, echo_FB1_acq2_seriea_lg, seg_FB1_acq2_seriea_lg, 200, 300, distance_liste, angle, "FB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CS1_acq1_seriea_lg, seg_CS1_acq1_seriea_lg, 210, 310, echo_CS1_acq2_seriea_lg, seg_CS1_acq2_seriea_lg, 220, 320, distance_liste, angle, "CS1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_CB1_acq1_seriea_lg, seg_CB1_acq1_seriea_lg, 160, 260, echo_CB1_acq2_seriea_lg, seg_CB1_acq2_seriea_lg, 162, 262, distance_liste, angle, "CB1")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_PG_acq1_seriea_lg,  seg_PG_acq1_seriea_lg,  260, 360, echo_PG_acq2_seriea_lg,  seg_PG_acq2_seriea_lg,  253, 353, distance_liste, angle, "PG")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_YG_acq1_seriea_lg,  seg_YG_acq1_seriea_lg,  260, 360, echo_YG_acq2_seriea_lg,  seg_YG_acq2_seriea_lg,  249, 349, distance_liste, angle, "YG")[1])
            haralick_feature.append(GLCM_treatement_tendon_3D(echo_IR_acq1_seriea_lg,  seg_IR_acq1_seriea_lg,  260, 360, echo_IR_acq2_seriea_lg,  seg_IR_acq2_seriea_lg,  230, 330, distance_liste, angle, "IR")[1])
    
    return haralick_feature

## Fonction de calcul de reproductibilité
"""
Mode d'emploi : 
1) Etre en possession d'une première liste correspondant à une paire de (distance, angle) pour les 19 patients, pour la première acquisition. Chaque patient a une valeur pour une 
caractéristique et une valeur par coupe du volume. De même pour soit la deuxième acquisition ou pour la deuxième série de la première acquisition. 

2) Ensuite passer la liste acq1, et acq2 avec l'indice de la caractéristique étudiée (allant de 0 à 5).

3) Ensuite moyenner les valeurs par volume donc une valeur par patient et non plus une valeur par coupe.
On est donc en possession de 19 valeurs pour les 19 patients pour la première acq et 19 valeurs pour les 19 valeurs pour la deuxième acquisition.

4)Mettre la première liste de 19 valeurs de acq1 et la deuxième liste de 19 valeurs dans soit le RMS_CV soit dans le cv_robuste_mad.

"""
# Fonction qui extrait les moyennes pour tous les patients de la premiere acq et de la deuxieme acq
def extract_feature_per_patient(features_acq1: list, features_acq2: list, idx_feature: int) -> Tuple[list, list]:    ## NE PAS UTILISER CETTE FONCTION
    """ 
    This function returns values of only one feature for each slide of the volume for the first acquisition and then for the second one.  

    Args:
        features_acq1 (list): list containing 19 sublists associated to 19 patients. Each sublist has 6 subsublists associated to 6 Haralick features. Each subsublist has the volume size corresponding to the Haralick 
                              value for the slide x for the first acquisition. 
                              Result of the function extract_feature_distance_classic if there is many distances (or angle) calculated, or result of GLCM_acq1_seriea_b_mean_VF or GLCM_acq1_2_VF if 
                              there is just one combinaison (distance, angle) for the first acquisition. 
        features_acq2 (list): Same as features_acq1 but for the second acquisition. 
        idx_feature (int): Associated with the Haralick feature that we want to extract : 
                           Contrast -> idx_feature = 0
                           Correlation idx_feature = 1
                           Dissimilarity -> idx_feature = 2
                           Homogeneity -> idx_feature = 3
                           ASM -> idx_feature = 4
                           Entropy -> idx_feature = 5

    Returns:
        Tuple [list, list]: the first list contains values of the feature associated with idx_feature for the first acquisition and the second list is the same for the second acquisition. 
    """
    feature_acq1 = []       # list will contains only one Haralick feature per slide for the first acquisition 
    feature_acq2 = []       # list will contains only one Haralick feature per slide for the first acquisition

    for i in range(len(features_acq1)): 
        feature_acq1.append(features_acq1[i][idx_feature])
        feature_acq2.append(features_acq2[i][idx_feature])
         
    return feature_acq1, feature_acq2


def RMS_CV(acq1: np.ndarray, acq2: np.ndarray) -> float:         ## NE PAS UTILISER CETTE FONCTION
    """
    This functions calculates the RMSCV for the values of one Haralick feature between the first and the second acquisition. 

    Args:
        acq1 (np.ndarray): Containing average on the volume -> one Haralick value per patient for the first acquisition. 
        acq2 (np.ndarray): Containing average on the volume -> one Haralick value per patient for the second acquisition. 

    Returns:
        float: CV in % corresponding to the coefficient of variation between two acquisitions. More CV is low, better it is. 
    """
    diff_sqrd = np.sum((acq1-acq2)**2)
    mean1 = np.mean(acq1)
    mean2 = np.mean(acq2)
    cv = (np.sqrt(diff_sqrd/(2*len(acq1))))/((mean1+mean2)/2)
    return cv*100


# Estimateur robuste aux outliers MAD
def cv_robuste_mad(acq1: np.ndarray,acq2: np.ndarray) -> float:      ## NE PAS UTILISER CETTE FONCTION
    """
    This function returns a coefficient of variation caclculated around the median, that helps to ignore extreme value of the date; the median is more robust to extreme values ​​than the mean

    Args:
        acq1 (np.ndarray): Containing average on the volume -> one Haralick value per patient for the first acquisition.
        acq2 (np.ndarray): Containing average on the volume -> one Haralick value per patient for the second acquisition.

    Returns:
        float: CV in % corresponding to the coefficient of variation between two acquisitions. More CV is low, better it is. 
    """
    x = np.array(acq1)
    y = np.array(acq2)

    diff = x-y 

    median_diff = np.median(x-y)

    mad = np.median(np.abs(diff-median_diff))

    mean_central = np.mean([np.median(x), np.median(y)])

    cv_robuste = mad/mean_central
    return cv_robuste*100


def calcul_repro(features_acq1: list, features_acq2: list, idx_feature: int) -> None:        ## UTILISER CETTE FONCTION
    """
    This function calculates the RMSCV and the CV considering the median instead of mean between both acquisitions. 

    Args:
        features_acq1 (list): list containing 19 sublists associated to 19 patients. Each sublist has 6 subsublists associated to 6 Haralick features. 
                              Each subsublist has the volume size corresponding to the Haralick value for the slide x for the first acquisition. 
                              Result of the function extract_feature_distance_classic if there is many distances (or angle) calculated, or result of GLCM_acq1_seriea_b_mean_VF or GLCM_acq1_2_VF if 
                              there is just one combinaison (distance, angle) for the first acquisition. 
        features_acq2 (list): Same as features_acq1 but for the second acquisition. 
        idx_feature (int): Associated with the Haralick feature that we want to extract : 
                           Contrast -> idx_feature = 0
                           Correlation idx_feature = 1
                           Dissimilarity -> idx_feature = 2
                           Homogeneity -> idx_feature = 3
                           ASM -> idx_feature = 4
                           Entropy -> idx_feature = 5
    
    Returns: 
        None: Deux sentences explaining the RMSCV and the coefficient of variation between thoses twos acquisitions. 
    """
    feature_name = ["Contrast", "Correlation", "Dissimilarity", "Homogeneity", "ASM", "Entropy"]

    feature_acq1, feature_acq2 = extract_feature_per_patient(features_acq1, features_acq2, idx_feature)

    mean_per_patient_acq1 = []
    for i in range(len(feature_acq1)): 
        mean_per_patient_acq1.append(np.mean(feature_acq1[i]))

    mean_per_patient_acq2 = []
    for i in range(len(feature_acq2)): 
        mean_per_patient_acq2.append(np.mean(feature_acq2[i]))

    print("Le RMSCV entre les deux acquisitions pour le Haralick feature", feature_name[idx_feature], "est", str(RMS_CV(np.array(mean_per_patient_acq1), np.array(mean_per_patient_acq2))))
    print("Le coefficient de variation par rapport à la médiane pour le Haralick feature", feature_name[idx_feature], "est", str(cv_robuste_mad(mean_per_patient_acq1, mean_per_patient_acq2)))


def calcul_repro2(features_acq1: list, features_acq2: list, idx_feature: int) -> None:        ## UTILISER CETTE FONCTION
    """
    This function calculates the RMSCV and the CV considering the median instead of mean between both acquisitions. 

    Args:
        features_acq1 (list): list containing 19 sublists associated to 19 patients. Each sublist has 6 subsublists associated to 6 Haralick features. 
                              Each subsublist has the volume size corresponding to the Haralick value for the slide x for the first acquisition. 
                              Result of the function extract_feature_distance_classic if there is many distances (or angle) calculated, or result of GLCM_acq1_seriea_b_mean_VF or GLCM_acq1_2_VF if 
                              there is just one combinaison (distance, angle) for the first acquisition. 
        features_acq2 (list): Same as features_acq1 but for the second acquisition. 
        idx_feature (int): Associated with the Haralick feature that we want to extract : 
                           Contrast -> idx_feature = 0
                           Correlation idx_feature = 1
                           Dissimilarity -> idx_feature = 2
                           Homogeneity -> idx_feature = 3
                           ASM -> idx_feature = 4
                           Entropy -> idx_feature = 5
    
    Returns: 
        None: Deux sentences explaining the RMSCV and the coefficient of variation between thoses twos acquisitions. 
    """
    feature_name = ["Contrast", "Correlation", "Dissimilarity", "Homogeneity", "ASM", "Entropy"]

    feature_acq1, feature_acq2 = extract_feature_per_patient(features_acq1, features_acq2, idx_feature)

    mean_per_patient_acq1 = []
    for i in range(len(feature_acq1)): 
        mean_per_patient_acq1.append(np.mean(feature_acq1[i]))

    mean_per_patient_acq2 = []
    for i in range(len(feature_acq2)): 
        mean_per_patient_acq2.append(np.mean(feature_acq2[i]))

    return RMS_CV(np.array(mean_per_patient_acq1), np.array(mean_per_patient_acq2)), cv_robuste_mad(mean_per_patient_acq1, mean_per_patient_acq2)

    
def print_repro(features_acq1, features_acq2) : 

    print("Le RMSCV entre les deux acquisitions pour le Haralick feature contraste est", str(calcul_repro2(features_acq1, features_acq2, 0)[0]), 
    ". Le coefficient de variation par rapport à la médiane pour le Haralick feature contraste est", str(calcul_repro2(features_acq1, features_acq2, 0)[1]))

    print("Le RMSCV entre les deux acquisitions pour le Haralick feature corrélation est", str(calcul_repro2(features_acq1, features_acq2, 1)[0]),
    ". Le coefficient de variation par rapport à la médiane pour le Haralick feature corrélation est", str(calcul_repro2(features_acq1, features_acq2, 1)[1]))

    print("Le RMSCV entre les deux acquisitions pour le Haralick feature dissimilarité est", str(calcul_repro2(features_acq1, features_acq2, 2)[0]),
    ". Le coefficient de variation par rapport à la médiane pour le Haralick feature dissimilarité est", str(calcul_repro2(features_acq1, features_acq2, 2)[1]))

    print("Le RMSCV entre les deux acquisitions pour le Haralick feature homogéneité est", str(calcul_repro2(features_acq1, features_acq2, 3)[0]),
    ". Le coefficient de variation par rapport à la médiane pour le Haralick feature homogéneité est", str(calcul_repro2(features_acq1, features_acq2, 3)[1]))

    print("Le RMSCV entre les deux acquisitions pour le Haralick feature ASM est", str(calcul_repro2(features_acq1, features_acq2, 4)[0]),
    ". Le coefficient de variation par rapport à la médiane pour le Haralick feature ASM est", str(calcul_repro2(features_acq1, features_acq2, 4)[1]))

    print("Le RMSCV entre les deux acquisitions pour le Haralick feature entropie est", str(calcul_repro2(features_acq1, features_acq2, 5)[0]),
    ". Le coefficient de variation par rapport à la médiane pour le Haralick feature entropie est", str(calcul_repro2(features_acq1, features_acq2, 5)[1]))

# Dans le cas où le GLCM_treatement est effectué pour plusieurs distances ou plusieurs angles, il nous faut isoler une combinaison (distance, angle), et pour se faire, les fonctions 
# suivantes permettent de ressortir les features pour acq1 ensuite acq2 pour une combinaison (distance, angle) car on a calculé pour une distance et plusieurs angle
def extract_feature_distance_classic(list_acq):     ## UTILISER CETTE FONCTION
    
    distance1 = distance2 = distance3 = distance4 = distance5 = distance6 = None
    distance1 =  list_acq[:19]
    distance2 = list_acq[19:38]
    distance3 = list_acq[38:57]
    distance4 = list_acq[57:76]
    distance5 = list_acq[76:95]
    distance6 = list_acq[95:114]

    return distance1, distance2, distance3, distance4, distance5, distance6


def extract_feature_angle_classic(list_acq):        ## UTILISER CETTE FONCTION
    
    angle1 = angle2 = angle3 = angle4 = angle5 = angle6 = None
    angle1 =  list_acq[:19]
    angle2 = list_acq[19:38]
    angle3 = list_acq[38:57]
    angle4 = list_acq[57:76]
    angle5 = list_acq[76:95]
    angle6 = list_acq[95:114]

    return angle1, angle2, angle3, angle4, angle5, angle6

# On applique ensuite calcul_repro

# 4. Selection of best Haralick features using PCA, Shapiro test, Spearman and Pearson coefficient for the classification 

# Moyenne d'angle : une distance, plusieurs angle -> faire moyenne des angles
def mean_volume(acq, idx_feature):
    """
    Calcule la moyenne du volume d'une feature sur chaque acquisition.
    """
    return [np.mean(patient[idx_feature]) for patient in acq]

def mean_column(matrix):
    """
    Calcule la moyenne de chaque ligne d'une matrice : 19 lignes pour les 19 patients et 7 colonnes pour les 7 angles
    """
    return np.mean(matrix, axis=1)

def mean_volume_angle(acq_all_patients, idx_feature, n_angles=7):
    """
    Calcule la moyenne du volume pour un index de feature donné, 
    puis regroupe les résultats selon l’angle d’acquisition.

    Parameters:
    - acq_all_patients: liste des acquisitions (len = n_angles * n_patients_par_angle)
    - idx_feature: index de la feature dont on veut la moyenne
    - n_angles: nombre d'angles (colonnes)

    Returns:
    - np.ndarray contenant la moyenne des volumes par patient (ligne) sur les angles (colonnes)
    """
    volume_means = mean_volume(acq_all_patients, idx_feature)
    volume_array = np.array(volume_means)

    n_total = len(volume_array)
    assert n_total % n_angles == 0, "Nombre total d'acquisitions non divisible par le nombre d'angles."
    n_patients = n_total // n_angles

    reshaped = volume_array.reshape((n_angles, n_patients)).T  # shape: (n_patients, n_angles)
    return mean_column(reshaped)

def print_repro_mean_parameter(features_acq1, features_acq2, n_angles):
    """
    Affiche les mesures de reproductibilité (RMSCV et CV robuste MAD) entre deux acquisitions
    pour les features Haralick moyennées par patient et par angle.

    Parameters:
    - features_acq1 : liste des features de l'acquisition 1
    - features_acq2 : liste des features de l'acquisition 2
    - n_angles : nombre d'angles considérés
    """
    feature_names = [
        "contraste", "corrélation", "dissimilarité",
        "homogénéité", "ASM", "entropie"
    ]

    for idx, name in enumerate(feature_names):
        feat_acq1 = mean_volume_angle(features_acq1, idx, n_angles)
        feat_acq2 = mean_volume_angle(features_acq2, idx, n_angles)

        rmscv = RMS_CV(feat_acq1, feat_acq2)
        cvmad = cv_robuste_mad(feat_acq1, feat_acq2)

        print(f"▶ Haralick feature : {name}")
        print(f"    - RMSCV : {rmscv:.4f}")
        print(f"    - CV robuste (MAD/médiane) : {cvmad:.4f}\n")


### Function for MUSCLE 

def GLCM_treatement_muscle_3D(
    echo_acq1: np.ndarray, 
    seg_acq1: np.ndarray, 
    frame_start1: int, 
    frame_end1: int, 
    echo_acq2: np.ndarray, 
    seg_acq2: np.ndarray, 
    frame_start2: int, 
    frame_end2: int, 
    distance: int, 
    angle: int, 
    patient_label: str) -> tuple[list, list, list]: ## UTILISER CETTE FONCTION
    
    """
    This functions applies second order statistics on muscle segmentation and extract Haralick features for both acquisitions. 

    Args : 
    echo_acq1 (np.ndarray): Muscle 3D ultrasound image of the first acquisition. Dimension of the image is (512,512,512). The first dimension is sagital plane of the muscle, 
                            the second one is the coronal plane, the last one is the transversal plane. 
    seg_acq1 (np.ndarray) : Segmentation of the muscle 3D ultrasound image of the first acquisition. Dimension of the image is (512,512,512). The first dimension is sagital plane of the muscle, 
                            the second one is the coronal plane, the last one is the transversal plane.
    frame_start1 (int)    : The first slide where the segmentation of the muscle is visible. frame_start1 ∈ [0,412]
    frame_end1 (int)      : The last slide of the muscle segmentation volume; Usually frame_end1 = frame_start1 + nx where nx can be equal to 90 or 100, depends on the patient's height cf Christine for more details.  
                            The volume studied is of size (512 row, 512 columns, frame_end1-frame_start1)
    Same for echo_acq2, seg_acq2, frame_start2, frame_end2, but for the second acquisition. 
    distance (int)        : Parameter of the GLCM, distance between two pixels that we are studying. 
    angle (int)           : Parameter of the GLCM, angle between two pixels that we are studying; BE CAREFUL !! Angle's unity is not degree but radians, then must write e.g np.radians(0), for angle = 0 radians. 
    patient_label (str)   : Name of the patient, using for the title of plot.
    
    Returns:
    Tuple [list, list, list] : the first list (respectively the second list) is about 6 features (contrast, correlation, dissimilarity, homogeneity, ASM, entropy) extracted for the first acquisition (respectively 
                                for the second acquisition) and displays into 6 graphs, one graph per feature. A graph is the evolution of the feature in the volume (as a function of slices). 
                                The last list is the mean between feature extracted from the first acquisition and from the second one. 
    """
    # Ouvrir la segmentation de la première acquisition 
    pre_muscle1 = echo_acq1*(seg_acq1>0)
    muscle1 = np.rot90(pre_muscle1, k = 3)

    # Ouvrir la segmentation de la deuxième acquisition
    pre_muscle2 = echo_acq2*(seg_acq2>0)
    muscle2 = np.rot90(pre_muscle2, k = 3)

    # Matrice vide qui contiendra les tendons érodés. 
    muscle_erode_acq1 = np.zeros_like(muscle1)
    muscle_erode_acq2 = np.zeros_like(muscle2)

    # Conversion des intensités des pixels de [0,1] à [0,255]
    muscle_convert1 = convert_intensities(muscle1)
    muscle_convert2 = convert_intensities(muscle2)

    # Pour stocker les features de la première acquisition 
    contrast1 = []
    correlation1 = []
    dissimilarity1 = []
    homogeneity1 = []
    ASM1 = []
    entropy1 = []

    # Pour stocker les features de la deuxième acquisition 
    contrast2 = []
    correlation2 = []
    dissimilarity2 = []
    homogeneity2 = []
    ASM2 = []
    entropy2 = []

    for i in range(muscle1.shape[2]): # boucle qui parcourt tout le volume de la première acquisition (taille du volume acq1 = taille du volume acq2)

        # Ouverture morphologiques : Erosion par un disque de rayon 1 pixel puis dilatation par un disque de rayon 1 des tendons pour enlever l'épimysium (la gaine) 
        muscle_erode_acq1[:,:,i] = dilation(erosion(muscle_convert1[:,:,i], disk(2)), disk(1))       
        muscle_erode_acq2[:,:,i] = dilation(erosion(muscle_convert2[:,:,i], disk(2)), disk(1))       

    # Calcul de la glcm pour la première acquisition
    for i in range(frame_start1, frame_end1):
        
        glcm_acq1 = graycomatrix(muscle_erode_acq1[:,:,i], distances = [distance], angles = [angle], levels = 256, 
                                        symmetric = False, normed = True)
        
        # La glcm est calculé pour un rectangle, mais la segmentation n'a pas la forme d'un rectangle, donc lors du calcul de la glcm on prend le fond noir en compte :
        # pour ignorer ces valeurs, on force les relations entre un pixel non noir et un pixel noir à être ignoré. Donc on met la première ligne et première colonne à 0
        
        glcm_acq1[0,:,:,:] = 0
        glcm_acq1[:,0,:,:] = 0

        # Extraction des Haralick features : on peut changer le type de Haralick feature (à regarder les autres Haralick features disponible sur la doc en ligne)
        contrast1.append(graycoprops(glcm_acq1[:,:,:,:],'contrast'))
        correlation1.append(graycoprops(glcm_acq1[:,:,:,:],'correlation'))
        dissimilarity1.append(graycoprops(glcm_acq1[:,:,:,:],'dissimilarity'))
        homogeneity1.append(graycoprops(glcm_acq1[:,:,:,:],'homogeneity'))
        ASM1.append(graycoprops(glcm_acq1[:,:,:,:],'ASM'))
        entropy1.append(graycoprops(glcm_acq1[:,:,:,:],'entropy'))

    # Analyse de la glcm pour la deuxième acquisition : même processus que pour la première acquisition
    for i in range(frame_start2, frame_end2):

        glcm_acq2 = graycomatrix(muscle_erode_acq2[:,:,i], distances = [distance], angles = [angle], levels = 256, 
                                        symmetric = False, normed = True)
        
        glcm_acq2[0,:,:,:] = 0
        glcm_acq2[:,0,:,:] = 0

        contrast2.append(graycoprops(glcm_acq2[:,:,:,:],'contrast'))
        correlation2.append(graycoprops(glcm_acq2[:,:,:,:],'correlation'))
        dissimilarity2.append(graycoprops(glcm_acq2[:,:,:,:],'dissimilarity'))
        homogeneity2.append(graycoprops(glcm_acq2[:,:,:,:],'homogeneity'))
        ASM2.append(graycoprops(glcm_acq2[:,:,:,:],'ASM'))
        entropy2.append(graycoprops(glcm_acq2[:,:,:,:],'entropy'))

    # Affichage des 6 Haralick features pour les 2 acquisitions : évolution des features en fonction dans le volume
    display_six_plot_two_acq([np.array(contrast1).reshape(len(contrast1)), np.array(correlation1).reshape(len(contrast1)), np.array(dissimilarity1).reshape(len(contrast1)), np.array(homogeneity1).reshape(len(contrast1)), np.array(ASM1).reshape(len(contrast1)), np.array(entropy1).reshape(len(contrast1))], 
                            [np.array(contrast2).reshape(len(contrast1)), np.array(correlation2).reshape(len(contrast1)), np.array(dissimilarity2).reshape(len(contrast1)), np.array(homogeneity2).reshape(len(contrast1)), np.array(ASM2).reshape(len(contrast1)), np.array(entropy2).reshape(len(contrast1))],
    f"Haralick features for patient "+ str(patient_label)+ f" distance = {distance}, angle = {angle}")

    save_glcm_features_to_excel_onepatient(
    acq1=[np.array(contrast1).reshape(len(contrast1)),
          np.array(correlation1).reshape(len(contrast1)),
          np.array(dissimilarity1).reshape(len(contrast1)),
          np.array(homogeneity1).reshape(len(contrast1)),
          np.array(ASM1).reshape(len(contrast1)),
          np.array(entropy1).reshape(len(contrast1))],

    acq2=[np.array(contrast2).reshape(len(contrast2)),
          np.array(correlation2).reshape(len(contrast2)),
          np.array(dissimilarity2).reshape(len(contrast2)),
          np.array(homogeneity2).reshape(len(contrast2)),
          np.array(ASM2).reshape(len(contrast2)),
          np.array(entropy2).reshape(len(contrast2))],

    filename=f"GLCM_features_{patient_label}.xlsx",  # base name
    distance=distance,
    angle=angle,
    patient_label=patient_label)
    
    return [contrast1, correlation1, dissimilarity1, homogeneity1, ASM1, entropy1], [contrast2, correlation2, dissimilarity2, homogeneity2, ASM2, entropy2], mean_haralick_feature([contrast1, correlation1, dissimilarity1, homogeneity1, ASM1, entropy1], [contrast2, correlation2, dissimilarity2, homogeneity2, ASM2, entropy2])

#Fonction après réorientation, à décommenter les sujets si les écho réorientés sont disponibles. 
def GLCM_acq1_2_VF_muscle(distance_liste, angle_liste, parameter_based):
    """ 
    Fonction qui prend en compte les acquisitions 1, série a et acquisition 2, série a et qui renvoie une liste L tel que 
    - L[i] i ∈ [1,(nbr dans la liste de paramètre)*nbr de sujets traités = (nbr dans la liste de paramètre)*19], donc si distance_liste = [1,2], i ∈ [1,2*19=38], L[i] aura les caractéristiques pour les 19 sujets 
    pour la distance = 1, puis les caractéristiques pour les 19 sujets pour la distance = 2. 
    - L[i][j] j ∈ [1,6], où j correspond aux Haralick features dans l'ordre : Contraste, corrélation, dissimilarité, homogéneité, ASM, entropy. 
    - L[i][j][k], k ∈ [1, nbr de coupe du volume] : correspond concrètement à la valeur du feature j, pour le sujet i, de la coupe k du volume 3D. 
    """
    subjects = [
        ("IH1", echo_IH1_acq1_seriea_vf, seg_IH1_acq1_seriea_vf, 254, 344, echo_IH1_acq2_seriea_vf, seg_IH1_acq2_seriea_vf, 221, 311),
        #("CB2", echo_CB2_acq1_seriea_vf, seg_CB2_acq1_seriea_vf, 254, 344, echo_CB2_acq2_seriea_vf, seg_CB2_acq2_seriea_vf, 221, 311),
        ("HT1", echo_HT1_acq1_seriea_vf, seg_HT1_acq1_seriea_vf, 258, 348, echo_HT1_acq2_seriea_vf, seg_HT1_acq2_seriea_vf, 260, 350),
        ("CA1", echo_CA1_acq1_seriea_vf, seg_CA1_acq1_seriea_vf, 278, 378, echo_CA1_acq2_seriea_vf, seg_CA1_acq2_seriea_vf, 300, 400),
        ("ET1", echo_ET1_acq1_seriea_vf, seg_ET1_acq1_seriea_vf, 271, 361, echo_ET1_acq2_seriea_vf, seg_ET1_acq2_seriea_vf, 260, 350),
        #("SLT", echo_SLT_acq1_seriea_vf, seg_SLT_acq1_seriea_vf, 230, 330, echo_SLT_acq2_seriea_vf, seg_SLT_acq2_seriea_vf, 200, 300),
        ("JH1", echo_JH1_acq1_seriea_vf, seg_JH1_acq1_seriea_vf, 217, 317, echo_JH1_acq2_seriea_vf, seg_JH1_acq2_seriea_vf, 233, 333),
        #("MK1", echo_MK1_acq1_seriea_vf, seg_MK1_acq1_seriea_vf, 172, 262, echo_MK1_acq2_seriea_vf, seg_MK1_acq2_seriea_vf, 198, 288),
        #("SR1", echo_SR1_acq1_seriea_vf, seg_SR1_acq1_seriea_vf, 215, 305, echo_SR1_acq2_seriea_vf, seg_SR1_acq2_seriea_vf, 230, 320),
        #("AV1", echo_AV1_acq1_seriea_vf, seg_AV1_acq1_seriea_vf, 220, 310, echo_AV1_acq2_seriea_vf, seg_AV1_acq2_seriea_vf, 245, 335),
        #("AB1", echo_AB1_acq1_seriea_vf, seg_AB1_acq1_seriea_vf, 200, 300, echo_AB1_acq2_seriea_vf, seg_AB1_acq2_seriea_vf, 189, 289),
        ("AB2", echo_AB2_acq1_seriea_vf, seg_AB2_acq1_seriea_vf, 240, 340, echo_AB2_acq2_seriea_vf, seg_AB2_acq2_seriea_vf, 229, 329),
        #("SB1", echo_SB1_acq1_seriea_vf, seg_SB1_acq1_seriea_vf, 225, 315, echo_SB1_acq2_seriea_vf, seg_SB1_acq2_seriea_vf, 230, 320),
        #("FB1", echo_FB1_acq1_seriea_vf, seg_FB1_acq1_seriea_vf, 260, 360, echo_FB1_acq2_seriea_vf, seg_FB1_acq2_seriea_vf, 240, 340),
        #("CS1", echo_CS1_acq1_seriea_vf, seg_CS1_acq1_seriea_vf, 276, 366, echo_CS1_acq2_seriea_vf, seg_CS1_acq2_seriea_vf, 297, 387),
        #("CB1", echo_CB1_acq1_seriea_vf, seg_CB1_acq1_seriea_vf, 240, 330, echo_CB1_acq2_seriea_vf, seg_CB1_acq2_seriea_vf, 265, 355),
        #("PG",  echo_PG_acq1_seriea_vf,  seg_PG_acq1_seriea_vf,  190, 290, echo_PG_acq2_seriea_vf,  seg_PG_acq2_seriea_vf,  213, 313),
        ("YG",  echo_YG_acq1_seriea_vf,  seg_YG_acq1_seriea_vf,  247, 347, echo_YG_acq2_seriea_vf,  seg_YG_acq2_seriea_vf,  249, 349),
        #("IR",  echo_IR_acq1_seriea_vf,  seg_IR_acq1_seriea_vf,  250, 350, echo_IR_acq2_seriea_vf,  seg_IR_acq2_seriea_vf,  264, 364),
    ]

    haralick_feature_acq1 = []
    haralick_feature_acq2 = []

    # Choix de la liste d’itération
    if parameter_based == "distance":
        iterable = distance_liste
        is_angle = False
    elif parameter_based == "angle":
        iterable = angle_liste
        is_angle = True
    else:
        raise ValueError("parameter_based must be 'distance' or 'angle'")

    for param in iterable:
        for subj in subjects:
            name, e1, s1, x1, y1, e2, s2, x2, y2 = subj
            if is_angle:
                res_acq1, res_acq2, _ = GLCM_treatement_muscle_3D(
                    e1, s1, x1, y1, e2, s2, x2, y2,
                    distance_liste, param, name
                )
            else:
                res_acq1, res_acq2, _ = GLCM_treatement_muscle_3D(
                    e1, s1, x1, y1, e2, s2, x2, y2,
                    param, angle_liste, name
                )
            haralick_feature_acq1.append(res_acq1)
            haralick_feature_acq2.append(res_acq2)

    return haralick_feature_acq1, haralick_feature_acq2


### LBP 
def display_two_plot_lbp(features_list1: list, features_list2: list, suptitle : str) -> None:
    """
    This functions displays 6 graphs in 2 rows, 3 columns with only one curve in each plot.

    Args : 
        features_list (list) : list of 6 features (contrast, correlation, dissimilarity, homogeneity, ASM, entropy) 
                               for a patient. Each feature is a numpy.ndarray type;   
        suptitle (str) : suptitle for the 6 graphs.
    
    Returns : 
        NoneType : 6 graphs showing the evolution of each feature as a function of acquisitions.    
    """
    plt.figure(figsize = (12,6))

    title_list = ["Entropy", "Energy"]
    for i in range(2):

        plt.subplot(1,2, i+1)
        plt.plot(features_list1[i],marker = 'o', color = "#69B3A2", label = "Acquisition 1")
        plt.plot(features_list2[i],marker = 'o', color = "indianred", label = "Acquisition 2")
        plt.title(title_list[i],fontsize=18)
        plt.grid(True)
        plt.xlabel("Image 3D cuts", fontsize = 18)
        plt.ylabel("Feature", fontsize = 18)
        plt.legend()

    plt.suptitle(suptitle, fontsize=20)
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    plt.show()


def save_LBP_to_excel(features_acq1, features_acq2, filename="LBP_features.xlsx"):
    """
    Enregistre les features LBP de deux acquisitions dans un fichier Excel.
    
    features_acq1 : [lbp_entropy1, lbp_energy1]
    features_acq2 : [lbp_entropy2, lbp_energy2]
    filename : nom du fichier Excel
    """
    # Vérifier que les listes ont la même longueur
    assert len(features_acq1[0]) == len(features_acq1[1]), "Acq1 : Les listes doivent avoir la même taille"
    assert len(features_acq2[0]) == len(features_acq2[1]), "Acq2 : Les listes doivent avoir la même taille"
    
    # Créer un DataFrame
    data = {
        "Entropy_Acq1": features_acq1[0],
        "Energy_Acq1": features_acq1[1],
        "Entropy_Acq2": features_acq2[0],
        "Energy_Acq2": features_acq2[1]
    }
    
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    print(f"Features LBP enregistrées dans {filename}")

def LBP_treatement_tendon(echo_acq1, seg_acq1, frame_start1, frame_end1, echo_acq2, seg_acq2, frame_start2, frame_end2, n_points, radius, suptitle): 

    # Ouvrir la segmentation de la première acquisition 
    pre_tendon1 = echo_acq1*(seg_acq1>0)
    tendon1 = np.rot90(pre_tendon1, k = 3)

    # Ouvrir la segmentation de la deuxième acquisition
    pre_tendon2 = echo_acq2*(seg_acq2>0)
    tendon2 = np.rot90(pre_tendon2, k = 3)

    tendon_erode_acq1 = np.zeros_like(tendon1)
    tendon_erode_acq2 = np.zeros_like(tendon2)

    lbp_energy1 = []
    lbp_entropy1 = []

    lbp_energy2 = []
    lbp_entropy2 = []

    for i in range(tendon1.shape[2]): # boucle qui va de 0 à 82 coupes

        # Erosion des tendons pour enlever la gaine 
        tendon_erode_acq1[:,:,i] = dilation(erosion(tendon1[:,:,i], disk(1)), disk(1))       # Volume de 83 coupes contenant uniquement la segmentation qui commence à 260 jusqu'à 343
        tendon_erode_acq2[:,:,i] = dilation(erosion(tendon2[:,:,i], disk(1)), disk(1))       # Volume de 83 coupes contenant uniquement la segmentation qui commence à 246 jusqu'à 329
        
    for i in range(frame_start1, frame_end1):
        # Analyse de LBP pour la première acquisition
        lbp = local_binary_pattern(tendon_erode_acq1[:,:,i], n_points, radius)
        
        hist,_ = np.histogram(lbp.ravel(), bins = int(lbp.max()+1), range =(0,int(lbp.max()+1)), density = True)
        lbp_entropy1.append(entropy(hist))
        lbp_energy1.append(np.sum(hist**2))

    for i in range(frame_start2, frame_end2):

        # Analyse de LBP pour la deuxième acquisition

        lbp2 = local_binary_pattern(tendon_erode_acq2[:,:,i], n_points, radius)
      
        hist2,_ = np.histogram(lbp2.ravel(), bins = int(lbp2.max()+1), range = (0,int(lbp2.max()+1)), density = True)
        lbp_entropy2.append(entropy(hist2))
        lbp_energy2.append(np.sum(hist2**2))

    entropy_mean = np.mean((np.array(lbp_entropy1), np.array(lbp_entropy2)))
    energy_mean = np.mean((np.array(lbp_energy1), np.array(lbp_energy2)))

    display_six_plot_two_acq([lbp_entropy1, lbp_energy1], [lbp_entropy2, lbp_energy2], suptitle)

    CV_entropy = (np.sqrt((np.sum((np.array(lbp_entropy1)-np.array(lbp_entropy2))**2))/(2*(frame_end1-frame_start1)))/((np.mean(lbp_entropy1)+np.mean(lbp_entropy2))/2))*100
    CV_energy = (np.sqrt((np.sum((np.array(lbp_energy1)-np.array(lbp_energy2))**2))/(2*(frame_end1-frame_start1)))/((np.mean(lbp_energy1)+np.mean(lbp_energy2))/2))*100 

    print("Le coefficient de variation de l'entropie entre les deux acquisitions est "+str(CV_entropy))
    print("Le coefficient de variation de l'énergie entre les deux acquisitions est " + str(CV_energy))

    save_LBP_to_excel([lbp_entropy1, lbp_energy1], [lbp_entropy2, lbp_energy2], filename="LBP_features.xlsx")
    
    return [lbp_entropy1, lbp_energy1], [lbp_entropy2, lbp_energy2], [entropy_mean, energy_mean]


def LBP_treatement_muscle(echo_acq1, seg_acq1, frame_start1, frame_end1, echo_acq2, seg_acq2, frame_start2, frame_end2, n_points, radius, suptitle): 
"""Varie de LBP_treatement_tendon car on érode par un disque de rayon 2 au lieu de 1"""
    # Ouvrir la segmentation de la première acquisition 
    pre_tendon1 = echo_acq1*(seg_acq1>0)
    tendon1 = np.rot90(pre_tendon1, k = 3)

    # Ouvrir la segmentation de la deuxième acquisition
    pre_tendon2 = echo_acq2*(seg_acq2>0)
    tendon2 = np.rot90(pre_tendon2, k = 3)

    tendon_erode_acq1 = np.zeros_like(tendon1)
    tendon_erode_acq2 = np.zeros_like(tendon2)

    lbp_energy1 = []
    lbp_entropy1 = []

    lbp_energy2 = []
    lbp_entropy2 = []

    for i in range(tendon1.shape[2]): # boucle qui va de 0 à 82 coupes

        # Erosion des tendons pour enlever la gaine 
        tendon_erode_acq1[:,:,i] = dilation(erosion(tendon1[:,:,i], disk(2)), disk(1))       # Volume de 83 coupes contenant uniquement la segmentation qui commence à 260 jusqu'à 343
        tendon_erode_acq2[:,:,i] = dilation(erosion(tendon2[:,:,i], disk(2)), disk(1))       # Volume de 83 coupes contenant uniquement la segmentation qui commence à 246 jusqu'à 329
        
    for i in range(frame_start1, frame_end1):
        # Analyse de LBP pour la première acquisition
        lbp = local_binary_pattern(tendon_erode_acq1[:,:,i], n_points, radius)
        
        hist,_ = np.histogram(lbp.ravel(), bins = int(lbp.max()+1), range =(0,int(lbp.max()+1)), density = True)
        lbp_entropy1.append(entropy(hist))
        lbp_energy1.append(np.sum(hist**2))

    for i in range(frame_start2, frame_end2):

        # Analyse de LBP pour la deuxième acquisition

        lbp2 = local_binary_pattern(tendon_erode_acq2[:,:,i], n_points, radius)
      
        hist2,_ = np.histogram(lbp2.ravel(), bins = int(lbp2.max()+1), range = (0,int(lbp2.max()+1)), density = True)
        lbp_entropy2.append(entropy(hist2))
        lbp_energy2.append(np.sum(hist2**2))

    entropy_mean = np.mean((np.array(lbp_entropy1), np.array(lbp_entropy2)))
    energy_mean = np.mean((np.array(lbp_energy1), np.array(lbp_energy2)))

    display_six_plot_two_acq([lbp_entropy1, lbp_energy1], [lbp_entropy2, lbp_energy2], suptitle)

    CV_entropy = (np.sqrt((np.sum((np.array(lbp_entropy1)-np.array(lbp_entropy2))**2))/(2*(frame_end1-frame_start1)))/((np.mean(lbp_entropy1)+np.mean(lbp_entropy2))/2))*100
    CV_energy = (np.sqrt((np.sum((np.array(lbp_energy1)-np.array(lbp_energy2))**2))/(2*(frame_end1-frame_start1)))/((np.mean(lbp_energy1)+np.mean(lbp_energy2))/2))*100 

    print("Le coefficient de variation de l'entropie entre les deux acquisitions est "+str(CV_entropy))
    print("Le coefficient de variation de l'énergie entre les deux acquisitions est " + str(CV_energy))

    save_LBP_to_excel([lbp_entropy1, lbp_energy1], [lbp_entropy2, lbp_energy2], filename="LBP_features.xlsx")

    return [lbp_entropy1, lbp_energy1], [lbp_entropy2, lbp_energy2], [entropy_mean, energy_mean]
