import os
import glob
import warnings
import numpy as np
from tqdm import tqdm
from natsort import natsorted
import numpy as np
import scipy
import scipy.signal as scisig

from datetime import datetime
to_date   = lambda string: datetime.strptime(string, '%Y-%m-%d')
S1_LAUNCH = to_date('2014-04-03')

# s2cloudless: see https://github.com/sentinel-hub/sentinel2-cloud-detector
from s2cloudless import S2PixelCloudDetector

import rasterio
from rasterio.merge import merge
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# from util.detect_cloudshadow import get_cloud_mask, get_shadow_mask


# utility functions used in the dataloaders of SEN12MS-CR and SEN12MS-CR-TS
def read_tif(path_IMG):
    tif = rasterio.open(path_IMG)
    return tif

def read_img(tif):
    return tif.read().astype(np.float32)

def rescale(img, oldMin, oldMax):
    oldRange = oldMax - oldMin
    img      = (img - oldMin) / oldRange
    return img

def process_MS(img, method):
    if method=='default':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img = rescale(img, [intensity_min, intensity_max])   # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method=='resnet':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img /= 2000                                        # project to [0,5], preserve global intensities (across patches)
    img = np.nan_to_num(img)
    return img

def process_SAR(img, method):
    if method=='default':
        dB_min, dB_max = -25, 0                            # define a reasonable range of SAR dB
        img = np.clip(img, dB_min, dB_max)                 # intensity clipping to a global unified SAR dB range
        img = rescale(img, [dB_min, dB_max])                 # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method=='resnet':
        # project SAR to [0, 2] range
        dB_min, dB_max = [-25.0, -32.5], [0, 0]
        img = np.concatenate([(2 * (np.clip(img[0], dB_min[0], dB_max[0]) - dB_min[0]) / (dB_max[0] - dB_min[0]))[None, ...],
                              (2 * (np.clip(img[1], dB_min[1], dB_max[1]) - dB_min[1]) / (dB_max[1] - dB_min[1]))[None, ...]], axis=0)
    img = np.nan_to_num(img)
    return img

def rescale(data, limits):
    return (data - limits[0]) / (limits[1] - limits[0])


def normalized_difference(channel1, channel2):
    subchan = channel1 - channel2
    sumchan = channel1 + channel2
    sumchan[sumchan == 0] = 0.001  # checking for 0 divisions
    return subchan / sumchan


def get_shadow_mask(data_image):
    data_image = data_image / 10000.

    (ch, r, c) = data_image.shape
    shadowmask = np.zeros((r, c)).astype('float32')

    BB     = data_image[1]
    BNIR   = data_image[7]
    BSWIR1 = data_image[11]

    CSI = (BNIR + BSWIR1) / 2.

    t3 = 3/4 # cloud-score index threshold
    T3 = np.min(CSI) + t3 * (np.mean(CSI) - np.min(CSI))

    t4 = 5 / 6  # water-body index threshold
    T4 = np.min(BB) + t4 * (np.mean(BB) - np.min(BB))

    shadow_tf = np.logical_and(CSI < T3, BB < T4)

    shadowmask[shadow_tf] = -1
    shadowmask = scisig.medfilt2d(shadowmask, 5)

    return shadowmask


def get_cloud_mask(data_image, cloud_threshold, binarize=False, use_moist_check=False):
    '''Adapted from https://github.com/samsammurphy/cloud-masking-sentinel2/blob/master/cloud-masking-sentinel2.ipynb'''

    data_image = data_image / 10000.
    (ch, r, c) = data_image.shape

    # Cloud until proven otherwise
    score = np.ones((r, c)).astype('float32')
    # Clouds are reasonably bright in the blue and aerosol/cirrus bands.
    score = np.minimum(score, rescale(data_image[1], [0.1, 0.5]))
    score = np.minimum(score, rescale(data_image[0], [0.1, 0.3]))
    score = np.minimum(score, rescale((data_image[0] + data_image[10]), [0.4, 0.9]))
    score = np.minimum(score, rescale((data_image[3] + data_image[2] + data_image[1]), [0.2, 0.8]))

    if use_moist_check:
        # Clouds are moist
        ndmi = normalized_difference(data_image[7], data_image[11])
        score = np.minimum(score, rescale(ndmi, [-0.1, 0.1]))

    # However, clouds are not snow.
    ndsi = normalized_difference(data_image[2], data_image[11])
    score = np.minimum(score, rescale(ndsi, [0.8, 0.6]))

    boxsize = 7
    box = np.ones((boxsize, boxsize)) / (boxsize ** 2)

    score = scipy.ndimage.morphology.grey_closing(score, size=(5, 5))
    score = scisig.convolve2d(score, box, mode='same')

    score = np.clip(score, 0.00001, 1.0)

    if binarize:
        score[score >= cloud_threshold] = 1
        score[score < cloud_threshold]  = 0

    return score

# IN: [13 x H x W] S2 image (of arbitrary resolution H,W), scalar cloud detection threshold
# OUT: cloud & shadow segmentation mask (of same resolution)
# the multispectral S2 images are expected to have their default ranges and not be value-standardized yet
# cloud_threshold: the higher the more conservative the masks (i.e. less pixels labeled clouds/shadows)
def get_cloud_cloudshadow_mask(data_image, cloud_threshold):
    cloud_mask = get_cloud_mask(data_image, cloud_threshold, binarize=True)
    shadow_mask = get_shadow_mask(data_image)

    # encode clouds and shadows as segmentation masks
    cloud_cloudshadow_mask = np.zeros_like(cloud_mask)
    cloud_cloudshadow_mask[shadow_mask < 0] = -1
    cloud_cloudshadow_mask[cloud_mask > 0]  = 1

    return cloud_cloudshadow_mask


def get_cloud_cloudshadow_mask(img, cloud_threshold=0.2):
    cloud_mask = get_cloud_mask(img, cloud_threshold, binarize=True)
    shadow_mask = get_shadow_mask(img)

    # encode clouds and shadows as segmentation masks
    cloud_cloudshadow_mask = np.zeros_like(cloud_mask)
    cloud_cloudshadow_mask[shadow_mask < 0] = -1
    cloud_cloudshadow_mask[cloud_mask > 0] = 1

    # label clouds and shadows
    cloud_cloudshadow_mask[cloud_cloudshadow_mask != 0] = 1
    return cloud_cloudshadow_mask


# recursively apply function to nested dictionary
def iterdict(dictionary, fct):
    for k,v in dictionary.items():        
        if isinstance(v, dict):
            dictionary[k] = iterdict(v, fct)
        else:      
            dictionary[k] = fct(v)      
    return dictionary

def get_cloud_map(img, detector, instance=None):

    # get cloud masks
    img = np.clip(img, 0, 10000)
    mask = np.ones((img.shape[-1], img.shape[-1]))
    # note: if your model may suffer from dark pixel artifacts,
    #       you may consider adjusting these filtering parameters
    if not (img.mean()<1e-5 and img.std() <1e-5):
        if detector == 'cloud_cloudshadow_mask':
            threshold = 0.2  # set to e.g. 0.2 or 0.4
            mask = get_cloud_cloudshadow_mask(img, threshold)
        elif detector== 's2cloudless_map':
            threshold = 0.5
            mask = instance.get_cloud_probability_maps(np.moveaxis(img/10000, 0, -1)[None, ...])[0, ...]
            mask[mask < threshold] = 0
            mask = gaussian_filter(mask, sigma=2)
        elif detector == 's2cloudless_mask':
            mask = instance.get_cloud_masks(np.moveaxis(img/10000, 0, -1)[None, ...])[0, ...]
        else:
            mask = np.ones((img.shape[-1], img.shape[-1]))
            warnings.warn(f'Method {detector} not yet implemented!')
    else:   warnings.warn(f'Encountered a blank sample, defaulting to cloudy mask.')
    return mask.astype(np.float32)


# function to fetch paired data, which may differ in modalities or dates
def get_pairedS1(patch_list, root_dir, mod=None, time=None):
    paired_list = []
    for patch in patch_list:
        seed, roi, modality, time_number, fname = patch.split('/')
        time = time_number if time is None else time # unless overwriting, ...
        mod  = modality if mod is None else mod      # keep the patch list's original time and modality
        n_patch       = fname.split('patch_')[-1].split('.tif')[0]
        paired_dir    = os.path.join(seed, roi, mod.upper(), str(time))
        candidates    = os.path.join(root_dir, paired_dir, f'{mod}_{seed}_{roi}_ImgNo_{time}_*_patch_{n_patch}.tif')
        paired_list.append(os.path.join(paired_dir, os.path.basename(glob.glob(candidates)[0])))
    return paired_list


""" SEN12MSCRTS data loader class, inherits from torch.utils.data.Dataset

    IN: 
    root:               str, path to your copy of the SEN12MS-CR-TS data set
    split:              str, in [all | train | val | test]
    region:             str, [all | africa | america | asiaEast | asiaWest | europa]
    cloud_masks:        str, type of cloud mask detector to run on optical data, in []
    sample_type:        str, [generic | cloudy_cloudfree]
    depricated --> vary_samples:       bool, whether to draw random samples across epochs or not, matters only if sample_type is 'cloud_cloudfree'
    sampler             str, [fixed | fixedsubset | random]
    n_input_samples:    int, number of input samples in time series
    rescale_method:     str, [default | resnet]
    min_cov:            float, in [0.0, 1.0]
    max_cov:            float, in [0.0, 1.0]
    import_data_path:   str, path to importing the suppl. file specifying what time points to load for input and output
    
    OUT:
    data_loader:        SEN12MSCRTS instance, implements an iterator that can be traversed via __getitem__(pdx),
                        which returns the pdx-th dictionary of patch-samples (whose structure depends on sample_type)
"""

class SEN12MSCRTS(Dataset):
    def __init__(self, split="all", region='all', cloud_masks='s2cloudless_mask', sample_type='cloudy_cloudfree', sampler='fixed', n_input_samples=3, rescale_method='default', min_cov=0.0, max_cov=1.0, import_data_path=None, custom_samples=None):
        
        self.root_dir = '/share/hariharan/cloud_removal/SEN12MSCRTS'   # set root directory which contains all ROI
        self.region   = region # region according to which the ROI are selected
        self.ROI      = {'ROIs1158': ['106'],
                         'ROIs1868': ['17', '36', '56', '73', '85', '100', '114', '119', '121', '126', '127', '139', '142', '143'],
                         'ROIs1970': ['20', '21', '35', '40', '57', '65', '71', '82', '83', '91', '112', '116', '119', '128', '132', '133', '135', '139', '142', '144', '149'],
                         'ROIs2017': ['8', '22', '25', '32', '49', '61', '63', '69', '75', '103', '108', '115', '116', '117', '130', '140', '146']}
        
        # define splits conform with SEN12MS-CR
        self.splits         = {}
        if self.region=='all':
            all_ROI             = [os.path.join(key, val) for key, vals in self.ROI.items() for val in vals]
            self.splits['test'] = [os.path.join('ROIs1868', '119'), os.path.join('ROIs1970', '139'), os.path.join('ROIs2017', '108'), os.path.join('ROIs2017', '63'), os.path.join('ROIs1158', '106'), os.path.join('ROIs1868', '73'), os.path.join('ROIs2017', '32'),
                                   os.path.join('ROIs1868', '100'), os.path.join('ROIs1970', '132'), os.path.join('ROIs2017', '103'), os.path.join('ROIs1868', '142'), os.path.join('ROIs1970', '20'), os.path.join('ROIs2017', '140')]  # official test split, across continents
            self.splits['val']  = [os.path.join('ROIs2017', '22'), os.path.join('ROIs1970', '65'), os.path.join('ROIs2017', '117'), os.path.join('ROIs1868', '127'), os.path.join('ROIs1868', '17')] # insert a validation split here
            self.splits['train']= [roi for roi in all_ROI if roi not in self.splits['val'] and roi not in self.splits['test']]  # all remaining ROI are used for training
        elif self.region=='africa':
            self.splits['test'] = [os.path.join('ROIs2017', '32'), os.path.join('ROIs2017', '140')]
            self.splits['val']  = [os.path.join('ROIs2017', '22')]
            self.splits['train']= [os.path.join('ROIs1970', '21'), os.path.join('ROIs1970', '35'), os.path.join('ROIs1970', '40'),
                                   os.path.join('ROIs2017', '8'), os.path.join('ROIs2017', '61'), os.path.join('ROIs2017', '75')]
        elif self.region=='america':
            self.splits['test'] = [os.path.join('ROIs1158', '106'), os.path.join('ROIs1970', '132')]
            self.splits['val']  = [os.path.join('ROIs1970', '65')]
            self.splits['train']= [os.path.join('ROIs1868', '36'), os.path.join('ROIs1868', '85'),
                                   os.path.join('ROIs1970', '82'), os.path.join('ROIs1970', '142'),
                                   os.path.join('ROIs2017', '49'), os.path.join('ROIs2017', '116')]
        elif self.region=='asiaEast':
            self.splits['test'] = [os.path.join('ROIs1868', '73'), os.path.join('ROIs1868', '119'), os.path.join('ROIs1970', '139')]
            self.splits['val']  = [os.path.join('ROIs2017', '117')]
            self.splits['train']= [os.path.join('ROIs1868', '114'), os.path.join('ROIs1868', '126'), os.path.join('ROIs1868', '143'), 
                                   os.path.join('ROIs1970', '116'), os.path.join('ROIs1970', '135'),
                                   os.path.join('ROIs2017', '25')]
        elif self.region=='asiaWest':
            self.splits['test'] = [os.path.join('ROIs1868', '100')]
            self.splits['val']  = [os.path.join('ROIs1868', '127')]
            self.splits['train']= [os.path.join('ROIs1970', '57'), os.path.join('ROIs1970', '83'), os.path.join('ROIs1970', '112'),
                                   os.path.join('ROIs2017', '69'), os.path.join('ROIs2017', '115'), os.path.join('ROIs2017', '130')]
        elif self.region=='europa':
            self.splits['test'] = [os.path.join('ROIs2017', '63'), os.path.join('ROIs2017', '103'), os.path.join('ROIs2017', '108'), os.path.join('ROIs1868', '142'), os.path.join('ROIs1970', '20')]
            self.splits['val']  = [os.path.join('ROIs1868', '17')]
            self.splits['train']= [os.path.join('ROIs1868', '56'), os.path.join('ROIs1868', '121'), os.path.join('ROIs1868', '139'),
                                   os.path.join('ROIs1970', '71'), os.path.join('ROIs1970', '91'), os.path.join('ROIs1970', '119'), os.path.join('ROIs1970', '128'), os.path.join('ROIs1970', '133'), os.path.join('ROIs1970', '144'), os.path.join('ROIs1970', '149'),
                                   os.path.join('ROIs2017', '146')]
        else: raise NotImplementedError

        self.splits["all"]  = self.splits["train"] + self.splits["test"] + self.splits["val"]
        self.split = split
        
        assert split in ['all', 'train', 'val', 'test'], "Input dataset must be either assigned as all, train, test, or val!"
        assert sample_type in ['generic', 'cloudy_cloudfree'], "Input data must be either generic or cloudy_cloudfree type!"
        assert cloud_masks in [None, 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'], "Unknown cloud mask type!"

        self.modalities     = ["S1", "S2"]
        self.time_points    = range(30)
        self.cloud_masks    = cloud_masks  # e.g. 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'
        self.sample_type    = sample_type if self.cloud_masks is not None else 'generic' # pick 'generic' or 'cloudy_cloudfree'
        self.sampling       = sampler # type of sampler
        self.vary_samples   = self.sampling =='random' if self.sample_type=='cloudy_cloudfree' else False # whether to draw different samples across epochs
        self.n_input_t      = n_input_samples  # specifies the number of samples, if only part of the time series is used as an input

        if self.vary_samples:
            if self.split in ['val', 'test']:
                warnings.warn(f'Loading {self.split} split, but sampled time points will differ each epoch!')
            else:
                warnings.warn(f'Randomly sampling targets, but remember to change seed if desiring different samples across models!')

        if self.vary_samples:
            self.t_windows = np.lib.stride_tricks.sliding_window_view(self.time_points, window_shape=self.n_input_t+1)

        if self.cloud_masks in ['s2cloudless_map', 's2cloudless_mask']:
            self.cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)
        else: self.cloud_detector = None

        self.import_data_path = import_data_path
        if self.import_data_path:
            # fetch time points as specified in the imported file, expects arguments are set accordingly
            if os.path.isdir(self.import_data_path):
                import_here   = os.path.join(self.import_data_path, f'generic_{self.n_input_t}_{self.split}_{self.region}_{self.cloud_masks}.npy')
            else:
                import_here   = self.import_data_path
            self.data_pairs   = np.load(import_here, allow_pickle=True).item()
            self.n_data_pairs = len(self.data_pairs)
            self.epoch_count  = 0 # count, for loading time points that vary across epochs
            print(f'\nImporting data pairings for split {self.split} from {import_here}.')
        else: print('\nData pairings are computed on the fly. Note. Pre-computing may speed up data loading')

        self.custom_samples   = custom_samples
        if isinstance (self.custom_samples, list):
            self.paths            = self.custom_samples
            self.import_data_path = None
        else: self.paths            = self.get_paths()
        self.n_samples        = len(self.paths)
        # raise a warning that no data has been found
        if not self.n_samples: self.throw_warn()

        self.method          = rescale_method
        self.min_cov, self.max_cov = min_cov, max_cov

    def throw_warn(self):
        warnings.warn("""No data samples found! Please use the following directory structure:
                        
        path/to/your/SEN12MSCRTS/directory:
        ├───ROIs1158
        ├───ROIs1868
        ├───ROIs1970
        │   ├───20
        │   ├───21
        │   │   ├───S1
        │   │   └───S2
        │   │       ├───0
        │   │       ├───1
        │   │       │   └─── ... *.tif files
        │   │       └───30
        │   ...
        └───ROIs2017
                        
        Note: the data is provided by ROI geo-spatially separated and sensor modalities individually.
        You can simply merge the downloaded & extracted archives' subdirectories via 'mv */* .' in the parent directory
        to obtain the required structure specified above, which the data loader expects.
        """)

    # indexes all patches contained in the current data split
    def get_paths(self):  # assuming for the same ROI+num, the patch numbers are the same
        print(f'\nProcessing paths for {self.split} split of region {self.region}')

        paths = []
        for inddex, (roi_dir, rois) in enumerate(self.ROI.items()):

            for roi in tqdm(rois):                
                
                roi_path = os.path.join(self.root_dir, roi_dir, roi)
                # skip non-existent ROI or ROI not part of the current data split
                if not os.path.isdir(roi_path) or os.path.join(roi_dir, roi) not in self.splits[self.split]: continue
                path_s1_t, path_s2_t = [], [],
                for tdx in self.time_points:
                    # working with directory under time stamp tdx
                    path_s1_complete = os.path.join(roi_path, self.modalities[0], str(tdx))
                    path_s2_complete = os.path.join(roi_path, self.modalities[1], str(tdx))

                    # same as complete paths, truncating root directory's path
                    path_s1 = os.path.join(roi_dir, roi, self.modalities[0], str(tdx))
                    path_s2 = os.path.join(roi_dir, roi, self.modalities[1], str(tdx))

                    # get list of files which contains all the patches at time tdx
                    s1_t = natsorted([os.path.join(path_s1, f) for f in os.listdir(path_s1_complete) if (os.path.isfile(os.path.join(path_s1_complete, f)) and ".tif" in f)])
                    s2_t = natsorted([os.path.join(path_s2, f) for f in os.listdir(path_s2_complete) if (os.path.isfile(os.path.join(path_s2_complete, f)) and ".tif" in f)])

                    # same number of patches
                    assert len(s1_t) == len(s2_t)

                    # sort via file names according to patch number and store
                    path_s1_t.append(s1_t)
                    path_s2_t.append(s2_t)

                # for each patch of the ROI, collect its time points and make this one sample
                for pdx in range(len(path_s1_t[0])):
                    sample = {"S1": [path_s1_t[tdx][pdx] for tdx in self.time_points],
                              "S2": [path_s2_t[tdx][pdx] for tdx in self.time_points]}
                    paths.append(sample)

        return paths


    def fixed_sampler(self, coverage, clear_tresh = 1e-3):
        # sample custom time points from the current patch space in the current split
        # sort observation indices according to cloud coverage, ascendingly
        coverage_idx = np.argsort(coverage)
        cloudless_idx = coverage_idx[0] # take the (earliest) least cloudy sample
        # take the first n_input_t samples with cloud coverage e.g. in [0.1, 0.5], ...
        inputs_idx = [pdx for pdx, perc in enumerate(coverage) if perc >= self.min_cov and perc <= self.max_cov][:self.n_input_t]
        if len(inputs_idx) < self.n_input_t:
            # ... if not exists then take the first n_input_t samples (except target patch)
            inputs_idx = [pdx for pdx in range(len(coverage)) if pdx!=cloudless_idx][:self.n_input_t]
            coverage_match   = False # flag input samples that didn't meet the required cloud coverage
        else: coverage_match = True  # assume the requested amount of cloud coverage is met
        # check whether the target meets the requested amount of clearness
        if coverage[cloudless_idx] > clear_tresh: coverage_match = False
        return inputs_idx, cloudless_idx, coverage_match

    def fixedsubset_sampler(self, coverage, earliest_idx=0, latext_idx=30, clear_tresh = 1e-3):
        # apply the fixed sampler on only a subsequence of the input sequence
        inputs_idx, cloudless_idx, coverage_match = self.fixed_sampler(self, coverage[earliest_idx:latext_idx], clear_tresh)
        # shift sampled indices by the offset of the subsequence
        inputs_idx, cloudless_idx = [idx + earliest_idx for idx in inputs_idx], cloudless_idx + earliest_idx
        # if the sampled indices do not meet the criteria, then default to sampling over the full time series
        if not coverage_match: inputs_idx, cloudless_idx, coverage_match = self.fixed_sampler(self, coverage, clear_tresh)
        return inputs_idx, cloudless_idx, coverage_match

    def random_sampler(self, coverage, clear_tresh = 1e-3):
        # sample a random target time point below 0.1% coverage (i.e. coverage<1e-3), or at min coverage
        is_clear = np.argwhere(np.array(coverage)<clear_tresh).flatten()
        try: cloudless_idx = is_clear[np.random.randint(0, len(is_clear))]
        except: cloudless_idx = np.array(coverage).argmin()
        # around this target time point, pick self.n_input_t input time points
        windows = [window for window in self.t_windows if cloudless_idx in window]
        # we pick the window with cloudless_idx centered such that input samples are temporally adjacent,
        # alternatively: pick a causal window (with cloudless_idx at the end) or randomly sample input dates
        inputs_idx = [input_t for input_t in windows[len(windows)//2] if input_t!=cloudless_idx]
        coverage_match = True # note: not checking whether any requested cloud coverage is met in this mode   
        return inputs_idx, cloudless_idx, coverage_match

    def sampler(self, s1, s2, masks, coverage, clear_tresh = 1e-3, earliest_idx=0, latext_idx=30):
        if self.sampling=='random':
            inputs_idx, cloudless_idx, coverage_match = self.random_sampler(coverage, clear_tresh)
        elif self.sampling=='fixedsubset':
            inputs_idx, cloudless_idx, coverage_match = self.fixedsubset_sampler(coverage, clear_tresh, earliest_idx=earliest_idx, latext_idx=latext_idx)
        else: # default to fixed sampler
            inputs_idx, cloudless_idx, coverage_match = self.fixed_sampler(coverage, clear_tresh)
        
        input_s1, input_s2, input_masks = np.array(s1)[inputs_idx], np.array(s2)[inputs_idx], np.array(masks)[inputs_idx]
        target_s1, target_s2, target_mask = np.array(s1)[cloudless_idx], np.array(s2)[cloudless_idx], np.array(masks)[cloudless_idx]

        data = {"input":  [input_s1, input_s2, input_masks, inputs_idx], 
                "target": [target_s1, target_s2, target_mask, cloudless_idx],
                "match":  coverage_match}
        return data

    def save_binary_matrix_as_npy(self, matrix, filename):
        np.save(filename, matrix)

    def read_npy_as_binary_matrix(self, filename):
        binary_matrix = np.load(filename)
        return binary_matrix


    # load images at a given patch pdx for given time points tdx
    def get_imgs(self, pdx, tdx=range(0,30)):
        # load the images and infer the masks
        s1_tif   = [read_tif(os.path.join(self.root_dir, img)) for img in np.array(self.paths[pdx]['S1'])[tdx]]
        s2_tif   = [read_tif(os.path.join(self.root_dir, img)) for img in np.array(self.paths[pdx]['S2'])[tdx]]
        coord    = [list(tif.bounds) for tif in s2_tif]
        s1       = [process_SAR(read_img(img), self.method) for img in s1_tif]
        s2       = [read_img(img) for img in s2_tif]  # note: pre-processing happens after cloud detection
        # masks    = None if not self.cloud_masks else [get_cloud_map(img, self.cloud_masks, self.cloud_detector) for img in s2]

        if not self.cloud_masks:
            masks = None

        else:
            masks = []
            for s2_img, s2_path in zip(s2, np.array(self.paths[pdx]['S2'])[tdx]):
                cloud_path = s2_path.split(".tif")[0] + "_cloud_mask.npy"
                # cloud_saved_path = s2_path.split(".tif")[0] + "_cloud_mask.npy"
                cloud_path = os.path.join(self.root_dir, cloud_path)
                
                if os.path.exists(cloud_path):
                    try:
                        mask = self.read_npy_as_binary_matrix(cloud_path)
                    except:
                        mask = get_cloud_map(s2_img, self.cloud_masks, self.cloud_detector).astype(bool)
                        self.save_binary_matrix_as_npy(mask, cloud_path)

                else:
                    mask = get_cloud_map(s2_img, self.cloud_masks, self.cloud_detector).astype(bool)
                    self.save_binary_matrix_as_npy(mask, cloud_path)
                    # logger.info(f"saving cloud mask: {cloud_path}")
                masks.append(mask)

        # get statistics and additional meta information
        coverage = [np.mean(mask) for mask in masks]
        s1_dates = [to_date(img.split('/')[-1].split('_')[5]) for img in np.array(self.paths[pdx]['S1'])[tdx]]
        s2_dates = [to_date(img.split('/')[-1].split('_')[5]) for img in np.array(self.paths[pdx]['S2'])[tdx]]
        s1_td    = [(date-S1_LAUNCH).days for date in s1_dates]
        s2_td    = [(date-S1_LAUNCH).days for date in s2_dates]

        return s1_tif, s2_tif, coord, s1, s2, masks, coverage, s1_dates, s2_dates, s1_td, s2_td

    # function to merge (a temporal list of spatial lists containing) raster patches into a single rasterized patch
    def mosaic_patches(self, paths):
        src_files_to_mosaic = []

        for tp in paths:
            tp_mosaic = []
            for sp in tp: # collect patches in space to mosaic over
                src = rasterio.open(os.path.join(self.root_dir, sp))
                tp_mosaic.append(src)
            mosaic, out_trans = merge(tp_mosaic)
            src_files_to_mosaic.append(mosaic.astype(np.float32))
        return src_files_to_mosaic #, mosaic_meta

    def getsample(self, pdx):
        return self.__getitem__(pdx)

    def __getitem__(self, pdx):  # get the time series of one patch

        # get all images of patch pdx for online selection of dates tdx
        #s1_tif, s2_tif, coord, s1, s2, masks, coverage, s1_dates, s2_dates, s1_td, s2_td = self.get_imgs(pdx)

        if self.sample_type == 'cloudy_cloudfree':

            s1_tif, s2_tif, coord, s1, s2, masks, coverage, s1_dates, s2_dates, s1_td, s2_td = self.get_imgs(pdx)
            data_samples =self.sampler(s1, s2, masks, coverage, clear_tresh = 1e-3)

            if not self.custom_samples:
                input_s1, input_s2, input_masks, inputs_idx         = data_samples['input']
                target_s1, target_s2, target_mask, cloudless_idx    = data_samples['target']
                coverage_match                                      = data_samples['match']
                
                # preprocess S2 data (after cloud masks have been computed)
                input_s2    = [process_MS(img, self.method) for img in input_s2]
                target_s2   = [process_MS(target_s2, self.method)]
                
                if not self.import_data_path:
                    in_s1_td, in_s2_td = [s1_td[idx] for idx in inputs_idx], [s2_td[idx] for idx in inputs_idx]
                    tg_s1_td, tg_s2_td = [s1_td[cloudless_idx]], [s2_td[cloudless_idx]]
                    in_coord, tg_coord = [coord[idx] for idx in inputs_idx], [coord[cloudless_idx]]

            sample = {'input': {'S1': list(input_s1),
                                'S2': input_s2,
                                'masks': list(input_masks),
                                'coverage': [np.mean(mask) for mask in input_masks],
                                'S1 TD': in_s1_td, #[s1_td[idx] for idx in inputs_idx],
                                'S2 TD': in_s2_td, #[s2_td[idx] for idx in inputs_idx],
                                'S1 path': [] if self.custom_samples else [os.path.join(self.root_dir, self.paths[pdx]['S1'][idx]) for idx in inputs_idx],
                                'S2 path': [] if self.custom_samples else [os.path.join(self.root_dir, self.paths[pdx]['S2'][idx]) for idx in inputs_idx],
                                'idx': [] if self.custom_samples else inputs_idx,
                                'coord': in_coord, #[coord[idx] for idx in inputs_idx],
                                },
                    'target': {'S1': [target_s1],
                                'S2': target_s2,
                                'masks': [target_mask],
                                'coverage': [np.mean(target_mask)],
                                'S1 TD': tg_s1_td, #[s1_td[cloudless_idx]],
                                'S2 TD': tg_s2_td, #[s2_td[cloudless_idx]],
                                'S1 path': [] if self.custom_samples else [os.path.join(self.root_dir, self.paths[pdx]['S1'][cloudless_idx])],
                                'S2 path': [] if self.custom_samples else [os.path.join(self.root_dir, self.paths[pdx]['S2'][cloudless_idx])],
                                'idx': [] if self.custom_samples else cloudless_idx,
                                'coord': tg_coord, #[coord[cloudless_idx]],
                                },
                    'coverage bin': coverage_match
                    }
        
        elif self.sample_type == 'generic':
            # did not implement custom sampling for options other than 'cloudy_cloudfree' yet
            if self.custom_samples: raise NotImplementedError

            s1_tif, s2_tif, coord, s1, s2, masks, coverage, s1_dates, s2_dates, s1_td, s2_td = self.get_imgs(pdx)

            sample = {'S1': s1,
                      'S2': [process_MS(img, self.method) for img in s2],
                      'masks': masks,
                      'coverage': coverage,
                      'S1 TD': s1_td,
                      'S2 TD': s2_td,
                      'S1 path': [os.path.join(self.root_dir, self.paths[pdx]['S1'][idx]) for idx in self.time_points],
                      'S2 path': [os.path.join(self.root_dir, self.paths[pdx]['S2'][idx]) for idx in self.time_points],
                      'coord': coord
                      }
        return sample
    

    def __len__(self):
        # length of generated list
        return self.n_samples

    def incr_epoch_count(self):
        # increment epoch count by 1
        self.epoch_count += 1
        
        


def get_SEN12MSCRTS(mode, batch_size, num_workers=2, num_frames=3):
    
    assert mode in ["all","train","val","test"]
    
    if mode == "train":
        loader = SEN12MSCRTS(split=mode, sample_type='cloudy_cloudfree', n_input_samples=num_frames)
        dataloader   = DataLoader(loader, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        
    elif mode == "val":
        loader = SEN12MSCRTS(split=mode, sample_type='cloudy_cloudfree', n_input_samples=num_frames)
        dataloader   = DataLoader(loader, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
    elif mode == "test":
        loader = SEN12MSCRTS(split=mode, sample_type='cloudy_cloudfree', n_input_samples=num_frames)
        dataloader   = DataLoader(loader, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # loader = SEN12MSCRTS(split="test", sample_type='cloudy_cloudfree', n_input_samples=num_frames)
    # dataloader   = DataLoader(loader, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader