# Speech_Enhancement_Using_Deep_Learning

This projects aims to enhance the quality of audio signal by removing noise where,
Deep Learning maps the relationship between the noisy and clean speech signals.

## Packgaes Required

1.Python 
2.Keras 2.3.1
3.Lebrosa
4.Soundfile
5.Numpy
6.pandas
7.Jupyter notebook

## How to run

If you have to denoise an audio input then create an directory dir_for_prediction and load sound_enhancement.py.
Then run (prediction function) with required directory and name of audio.wav.

How I have approached the problem:


**In Brief:-**

Step:1::Getting the dataset

Step:2::Changing the audio in Time-Serie

Step:3::Visualisation of audio signal and checking spectograms.

Step:3::Pre-processing the dataset

Step:4::Making input in accordance with the model input

Step:5::Training the model

Step:6::Checking how efficient the model is.

**With Explations:-**

**Step:1::**

For clean voice: LibriSpeech

For noise voice: ESC-50 dataset

**Step:2::**

Using librosa library converted .wav/.flac into numpy array of 8k sampling rate.

Now the converted array size is (32000,) to (38000,0) depending on the duration of audio signal.

**Step:3::**

Used librosa.display for visualizing time Vs amplitude plot of voise and noise.

Librosa library has been used for further magnitude and phase plots.

**Step:4::**

Used this value for different constants to process the data:

**sample\_rate=8000**

**min\_duration=1.0**

**hop\_length\_frame=8064 Name in accordance of Librosa library**

**hop\_length\_frame\_noise=5000**

**nb\_samples=50**

**n\_fft=255**

**hop\_length\_fft=63**

I have created an random noise array with shape (50,8064) then blends voice input.Performs same operation on noise data.

Then we have to change that blended audios into spectograms,as model will take magnitude/phase spectograms as input.

For this we will use short time fourier transform to change domain to phase.

Librosa again make it easy for us by librosa.stft module.

Audio file output after this was:-(50,8064 ),Then we use np.vstack to stack all audio for the training input.

**Step:5::**

For a given noisy\_clean\_audio,we are predicting **noisy part using CNN based UNET pre-trained model.**

The input has been transformed from shape () into (),in accordance to fir in UNET model.

**Step:6::**

As,the output is predicted using an pre-trained model with very few examples.So there is great room for increasing performance,but as with limited Google drive space and data I can&#39;t trained it on 20GB of training data.

But I just want to show how I will be approach and solve this

Voise(Time Vs Amplitude)

![alt text](/images/1.png)

Noise(Time Vs Amplitude)

![alt text](/images/2.png)

Blended Noisy\_voice(Time Vs Amplitude)

![alt text](/images/3.png)

Final output

![alt text](/images/1.png)

## Application
Well speech denoising has many application but my aim was to use it to enhance the audio quality in an call recordings.
Usually phone calls have wide variety of noises and voice quality is also poor Building this module will help in understanding phone conversation
well.

## Contributing
This was my side project to learn and explore speech_processing using deep learning.
