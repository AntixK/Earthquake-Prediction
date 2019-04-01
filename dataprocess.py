import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from scipy import stats
from scipy.fftpack import dct
import pathlib
import os

class DataProcessor():
    def __init__(self, filename, path, windowsize = 36):
        self.original_filename = filename
        self.data_path = path
        self.chunksize = windowsize
        self.resolution = None
        self.data_chunks = None
        self.filename = None

        pathlib.Path('/processed_data').mkdir(exist_ok=True) 
        self.processed_data_path = os.getcwd() + "/processed_data/"
        #self.load_data()

    def load_data(self, data_path, chunksize = None):

        if self.filename is None:
            raise ValueError("Raw file not processed! Run resolve_data() first")
        if chunksize is None:
            chunksize = self.chunksize
        data_chunks = pd.read_csv(data_path+self.filename,
                                       chunksize=chunksize,low_memory=False)
        return data_chunks

    def get_iterator(self):
        if self.data_chunks is None:
            raise ValueError("Processed data not loaded!")
        return self.data_chunks

    def set_chunk_size(self, chunksize):
        self.chunksize = chunksize

    def get_next_frame(self):
        if self.data_chunks is None:
            raise ValueError("Processed data not loaded!")
        return self.data_chunks.get_chunk()

    def reset_dataloader(self):
        self.resolution = None
        self.data_chunks = None
        self.filename = None
        self.processed_data_path = None

    def data_summary(self):
        if self.filename is None:
            raise ValueError("Raw file not processed! Run resolve_data() first")
        data = self.load_data(self.data_path)
        print(data.describe())

    def shape(self):
        if self.filename is None:
            raise ValueError("Raw file not processed! Run resolve_data() first")
        data = self.load_data(self.data_path)
        return data.shape


    def plot_data_sample(self, N = 200_000_000, subsample = 1_000, transform_x = None, figsize=(25,6)):

        if self.filename is None:
            temp_chunk = pd.read_csv(self.data_path + self.original_filename,
                                 dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64}, nrows=N)
            x = temp_chunk.values[:, 0][::subsample]
            y = temp_chunk.values[:, 1][::subsample]
        else:
            temp_chunk = pd.read_csv(self.processed_data_path + self.filename)

            x = temp_chunk.values[:,0]
            y = temp_chunk.values[:,1]

        if transform_x is not None:
            x =  transform_x(x)

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
        l1 = ax1.plot(x, lw=2, label="Acoustic Signal")

        ax2 = ax1.twinx()
        l2 = ax2.plot(y, "r", lw=2, label="Time to Failure")

        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, fontsize=25)

        ax1.grid(True)
        ax1.set_xlabel("Index", fontsize=25)
        ax1.set_ylabel("Signal", fontsize=25)
        ax2.set_ylabel("Time (s)", fontsize=25)

        plt.show()

    def plot_resolution(self):
        self.plot_data_sample(N=10_000, subsample=1, figsize=(18,6))

    def _find_resolution(self):
        N = 10_000
        temp_chunk = pd.read_csv(self.data_path+self.original_filename, nrows=N)
        y = temp_chunk.values[:, 1]

        a = np.ediff1d(y) # Find consecutive element difference
        self.resolution = int(np.ediff1d(np.where(a < -0.0001)).mean()) # If the difference is more than threshold, that's the resolution

    def resolve_data(self, summary_stats = 'mean'):
        """
        The "time to failure" has a resolution of 4096 as shown in the plot_resolution function.
        So, this function compresses the dataset by approximating the batches of 4096 values by
        its sufficient statistics, since the time to failure remians the same.
        We call this resolving the data.
        
        """
        from pathlib import Path
        file = Path(self.processed_data_path + "processed_train.csv")
        if file.is_file():
            print("Pre-resolved data already available!")
            self.filename = "processed_train.csv"
        else:
            print("Warning : This process may take some time... Please wait...")
            if self.resolution is None:
                self._find_resolution()

            assert self.resolution == 4096

            data_chunks = pd.read_csv(self.data_path + self.original_filename,
                                      dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64},
                                      chunksize=self.resolution - 1, low_memory=False)

            x_s, y_s = [], []
            # Replace the signal with the summary statistic within the resolution frame
            print("Using mean/std")
            for chunk in tqdm(data_chunks):
                x = chunk.values[:, 0]
                y = chunk.values[:, 1]

                if summary_stats == 'z-score':
                    x_s.append(x.mean()/(x.std() + 1e-6))
                elif summary_stats == 'mean':
                    x_s.append(x.mean())
                else:
                    raise ValueError("Unknown summary statistic given to resolve the data!")
                y_s.append(y.mean())

            df = pd.DataFrame({'acoustic_data': x_s, 'time_to_failure':y_s})
            df.to_csv(self.processed_data_path + "resolved_train.csv", index=False)
            self.filename = "resolved_train.csv"
            print("Done! Resolved data saved at {}".format(self.processed_data_path + self.filename))

    def extract_features(self, features = None, N = 150_000):
        tr_chunks = pd.read_csv(self.data_path+'train.csv', chunksize = N, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
        num_segments = int(np.floor(629_145_480 / N))
        X_tr = pd.DataFrame(index=range(num_segments), dtype=np.float64)
        y_tr = pd.DataFrame(index=range(num_segments), dtype=np.float64,
                            columns=['time_to_failure'])
        print("Extrating features... Please wait...")
        for segment, seg in tqdm_notebook(enumerate(tr_chunks)):

            x_raw = seg['acoustic_data']
            x = x_raw.values
            y = seg['time_to_failure'].values[-1]
            
            y_tr.loc[segment, 'time_to_failure'] = y

            # Moment-based Features
            X_tr.loc[segment, 'ave'] = x.mean()
            X_tr.loc[segment, 'std'] = x.std()
            X_tr.loc[segment, 'kurt'] = stats.kurtosis(x)
            X_tr.loc[segment, 'skew'] = stats.skew(x)

            # Quantile Features
            X_tr.loc[segment, 'min'] = x.min()
            X_tr.loc[segment, 'q01'] = np.quantile(x,0.01)
            X_tr.loc[segment, 'q05'] = np.quantile(x,0.05)
            X_tr.loc[segment, 'q95'] = np.quantile(x,0.95)
            X_tr.loc[segment, 'q99'] = np.quantile(x,0.99)
            X_tr.loc[segment, 'abs_median'] = np.median(np.abs(x))
            X_tr.loc[segment, 'abs_q95'] = np.quantile(np.abs(x),0.95)
            X_tr.loc[segment, 'abs_q99'] = np.quantile(np.abs(x),0.99)
            X_tr.loc[segment, 'F_test'], X_tr.loc[segment, 'p_test'] = stats.f_oneway(x[:30000],x[30000:60000],x[60000:90000],x[90000:120000],x[120000:])
            X_tr.loc[segment, 'av_change_abs'] = np.mean(np.diff(x))
            
            # Frequency Domain Features
            a_dct = np.abs(dct(x))
            X_tr.loc[segment, 'mean-abs-DCT'] = a_dct.mean()
            X_tr.loc[segment, 'max-abs-DCT'] = a_dct.max()
            X_tr.loc[segment, 'min-abs-DCT'] = a_dct.min()    
            X_tr.loc[segment, 'q-DCT-0.05'] = np.quantile(a_dct,0.05)
            X_tr.loc[segment, 'q-DCT-0.95'] = np.quantile(a_dct,0.95)
            X_tr.loc[segment, 'q-DCT-0.25'] = np.quantile(a_dct,0.25)
            X_tr.loc[segment, 'q-DCT-0.75'] = np.quantile(a_dct,0.75)
            
            # Norm Features
            X_tr.loc[segment, 'max'] = x.max()            
            X_tr.loc[segment, 'max-abs'] = np.abs(x).max()
            X_tr.loc[segment, 'norm-2'] = np.linalg.norm(x,2)
            X_tr.loc[segment, 'norm-3'] = np.linalg.norm(x,3)
            
            for windows in [10,100, 1000]:
                x_roll_std = x_raw.rolling(windows).std().dropna().values
                x_roll_mean = x_raw.rolling(windows).mean().dropna().values
                
                X_tr.loc[segment, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
                X_tr.loc[segment, 'std_roll_std_' + str(windows)] = x_roll_std.std()
                X_tr.loc[segment, 'max_roll_std_' + str(windows)] = x_roll_std.max()
                X_tr.loc[segment, 'min_roll_std_' + str(windows)] = x_roll_std.min()
            
                X_tr.loc[segment, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.01)
                X_tr.loc[segment, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.05)
                X_tr.loc[segment, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.95)
                X_tr.loc[segment, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std,0.99)
                X_tr.loc[segment, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
                X_tr.loc[segment, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
                
                X_tr.loc[segment, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
                X_tr.loc[segment, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
                X_tr.loc[segment, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
                X_tr.loc[segment, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
                X_tr.loc[segment, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean,0.01)
                X_tr.loc[segment, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean,0.05)
                X_tr.loc[segment, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean,0.95)
                X_tr.loc[segment, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean,0.99)
                X_tr.loc[segment, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
                X_tr.loc[segment, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

        X_tr.to_csv(self.processed_data_path+"X_train_features.csv",index=False)
        y_tr.to_csv(self.processed_data_path+"y_train_features.csv",index=False)
        print("Done! Feature extracted data saved at {}".format(self.processed_data_path))

    def get_feature_data(self):
        X_tr = pd.read_csv(self.processed_data_path+"X_train_features.csv")
        y_tr = pd.read_csv(self.processed_data_path+"y_train_features.csv")
        return X_tr, y_tr

class TestDataLoader():
    pass





