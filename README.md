# Speech_Enhancement_DL
It will take noisy audio as input and remove all noises and produce clean audio.

# **Apurv**





**In Brief:-**

Step:1::Getting the dataset

Step:2::Changing the audio in Time-Serie

Step:3::Visualisation of audio signal and checking spectograms.

Step:3::Pre-processing the dataset

Step:4::Making input in accordance with the model input

Step:5::Training the model

Step:6::Checking how efficient the model is.

**Explation:-**

Step:1::

For clean voice: LibriSpeech

For noise voice: ESC-50 dataset

Step:2::

Using librosa library converted .wav/.flac into numpy array of 8k sampling rate.

Now the converted array size is (32000,) to (38000,0) depending on the duration of audio signal.

Step:3::

Used librosa.display for visualizing time Vs amplitude plot of voise and noise.

Librosa library has been used for further magnitude and phase plots.

Step:4::

Used this value for different constants to process the data:

sample\_rate=8000

min\_duration=1.0

![](RackMultipart20201109-4-1ws0119_html_145f73ec10e003bf.gif)frame\_length=8064

hop\_length\_frame=8064 Name in accordance of Librosa library

hop\_length\_frame\_noise=5000

nb\_samples=50

n\_fft=255

hop\_length\_fft=63

I have created an random noise array with shape (50,8064) then blends voice input.Performs same operation on noise data.

Then we have to change that blended audios into spectograms,as model will take magnitude/phase spectograms as input.

For this we will use short time fourier transform to change domain to phase.

Librosa again make it easy for us by librosa.stft module.

Audio file output after this was:-(50,8064 ),Then we use np.vstack to stack all audio for the training input.

Step:5:-

For a given noisy\_clean\_audio,we are predicting **noisy part using CNN based UNET pre-trained model.**

The input has been transformed from shape () into (),in accordance to fir in UNET model.

Step:6:-

As,the output is predicted using an pre-trained model with very few examples.So there is great room for increasing performance,but as with limited Google drive space and data I can&#39;t trained it on 20GB of training data.

But I just want to show how I will be approach and solve this

Voise(Time Vs Amplitude)

![Alt text](images/1.PNG?raw=true "Title")

Noise(Time Vs Amplitude)

![Alt text](images/2.PNG?raw=true "Title"

Blended Noisy\_voice(Time Vs Amplitude)

![Alt text](images/3.PNG?raw=true "Title"

. Final output

![Alt text](images/4.PNG?raw=true "Title"
