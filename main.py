import os
import warnings
import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
import seaborn as sb
import matplotlib.pyplot as plt
import pvl
import ray

warnings.filterwarnings("ignore", message="Could not find isis. Is `ISISROOT` set?")
from pysis import CubeFile



def fetch_d_from_lbl(filepath):
    with open(filepath, 'r') as file:
        lbl_data = pvl.load(file)

    d_km = float(lbl_data['SPACECRAFT_SOLAR_DISTANCE'][0])
    d_au = d_km / 149597870.7
    
    return d_au


def mk_data_dict(directory):
    all_files = [file for file in os.listdir(directory)]
    if len(all_files) == 0:
        return print("The directory is empty.")
    else:
        cubefiles_dict = {}
        for file in all_files:
            if file.endswith('.cub') and '_phodata' not in file:
                base_name = file.replace('.cub', '')

                # data_file_path = os.path.join(directory, file)

                # geo_info_file = f"{base_name}_phodata.cub"
                # geo_info_path = os.path.join(directory, geo_info_file)

                denoise_iof_file = f"{base_name}_1_artifact_corrected.npy"
                denoised_directory = '/Data/sourav/Artifact_correction/Data/artifact-corrected/artifact-corrected-survey-ir-ceres-nasa-pds/'
                denoised_iof_file_path = os.path.join(denoised_directory, denoise_iof_file)

                lbl_file = f"{base_name}_1.LBL"
                lbl_file_path = os.path.join(directory, lbl_file)

                cubefiles_dict[base_name] = {
                    # "data": data_file_path,
                    "lbl_file": lbl_file_path,
                    # "geo_info": geo_info_path,
                    "iof_data": denoised_iof_file_path,
                }
        if len(cubefiles_dict) == 0:
            print("No cubefiles detected in the provided repository.")
        else:
            print(str(len(cubefiles_dict)) + " files has been mapped.")

        return cubefiles_dict


def cubefile_to_numpy(imagefile):
    image = CubeFile.open(imagefile)
    image_datalist = image.apply_numpy_specials()
    return image_datalist



source_directory = '/Data/soumik_backup/'

data_directory = source_directory + 'survey_1b/cub_converted/'
save_directory = source_directory + 'survey_1b/clark_2/'

ss_data = np.loadtxt(source_directory+"ss-ceres-dawn.txt", delimiter='\t', skiprows=1)

wavelengths = ss_data[:, 0]
solar_flux =  ss_data[:, 1]

vir_corr_factor = np.loadtxt(source_directory+'VIR_correction_factor_IR.txt', delimiter=',', skiprows=1)

survey_dict = mk_data_dict(data_directory)
keys = [key for key in survey_dict.keys()]


def blackbody_rad(wav,T):
    c1 = 3.74177*(10**-16)          #2*const.pi*const.h*(const.c**2)            #3.7417718521927573e-16
    c2 = 0.014387774                #const.c*const.h/const.Boltzmann            #0.014387768775039337

    radiance= ((c1/((wav/(10**6))**5))*(1/(np.exp(c2/((wav*T)/(10**6)))-1)))/((10**6)*np.pi)
    return radiance


def clark_model(image_array,
                d=2.96,
                fitting_wav_range=[3, 4.1],
                t_init=210,
                t_bound=False,
                fit_e=False,
                e_init=0.9,
                ):

    wav_array = wavelengths

    fit_mask = (wav_array >= fitting_wav_range[0]) & (wav_array <= fitting_wav_range[1])
    wav_fit = wav_array[fit_mask]

    fitted_temp = np.zeros((image_array.shape[1], image_array.shape[2]))
    temp_error = np.zeros((image_array.shape[1], image_array.shape[2]))

    fitted_emissivity = np.zeros((image_array.shape[1], image_array.shape[2]))
    emissivity_error = np.zeros((image_array.shape[1], image_array.shape[2]))


    for i in range(image_array.shape[1]):
        # if i % 10 == 0:
        #     print("i= ", i)
        for j in range(image_array.shape[2]):
            ref_array = image_array[:, i, j]

            ref_array_thermal = ref_array - np.mean(ref_array[(wav_array >= 1.5) & (wav_array <= 2.5)])

            gray_body_ref = ref_array_thermal * ss_data[:, 1] / (np.pi * (d**2))

            gray_body_fit = gray_body_ref[fit_mask]

            if fit_e==False:
                emissivity = 1 - np.mean(ref_array[(wav_array >= 1.5) & (wav_array <= 2.5)])
                emissivity_sem = np.std(ref_array[(wav_array >= 1.5) & (wav_array <= 2.5)], ddof=1) / np.sqrt(len(ref_array[(wav_array >= 1.5) & (wav_array <= 2.5)]))

                black_body_fit = gray_body_fit / emissivity

                non_nan_indices = ~np.isnan(black_body_fit)

                filtered_wav_fit = wav_fit[non_nan_indices]
                filtered_black_body_fit = black_body_fit[non_nan_indices]

                def fit_func(wav, T):
                    return blackbody_rad(wav, T)
                
                try:
                    popt, pcov = curve_fit(fit_func, filtered_wav_fit, filtered_black_body_fit, p0=[t_init])

                    fitted_temp[i, j] = popt[0]
                    temp_error[i, j] = np.sqrt(pcov[0, 0])

                    fitted_emissivity[1, j] = emissivity
                    emissivity_error[i, j] = emissivity_sem
                except:
                    continue

            else:
                def fit_func(wav, T, emissivity):
                    return emissivity * blackbody_rad(wav, T)

                popt, _ = curve_fit(fit_func, wav_fit, gray_body_fit, p0=[t_init, e_init])

                fitted_temp[i, j] = popt[0]
                fitted_emissivity[i, j] = popt[1]

    return fitted_temp, fitted_emissivity, temp_error, emissivity_error


ray.init(ignore_reinit_error=True, num_cpus=30)

@ray.remote
def process_key(key, survey_dict, save_directory):
    print(key)
    iof_image = np.load(survey_dict[key]['iof_data'])
    d_au = fetch_d_from_lbl(survey_dict[key]['lbl_file'])

    fitted_temp, fitted_emissivity, temp_error, emissivity_error = clark_model(iof_image, d=d_au, fitting_wav_range=[3, 4.1])

    data_to_save = {
        'fitted_temp': fitted_temp,
        'fitted_emissivity': fitted_emissivity,
        'temp_error': temp_error,
        'emissivity_error': emissivity_error
    }

    filepath = f"{save_directory}{key}_temp_emissivity.npy"
    np.save(filepath, data_to_save)
    print(f"Saved results for {key} to {filepath}")

tasks = [process_key.remote(key, survey_dict, save_directory) for key in survey_dict.keys()]

ray.get(tasks)

# Shutdown Ray
ray.shutdown()



