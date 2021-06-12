# SSC Probability Workshop 2021

## Schedule:
### (all times eastern)
    1:00 - 1:30 Introductions
    1:30 - 2:00 Form groups
    2:00 - 2:30 Group work
    2:30 - 3:00 Break
    3:00 - 4:00 Group work
    4:00 - 5:00 Informal presentations

## Abstract:
There are two basic ingredients for particle tracking: (1) microscopy videos of nanometer to micrometer sized "particles" suspended in a fluid and (2) a stochastic model of particle motion. Given these two ingredients, we can use machine learning methods to gain insight into micron-scale systems. Particle tracking has many applications in physics, chemistry, and biology. We will be focusing primarily on the latter. Some examples of "particles" are synthetic beads, genetically expressed fluorescent proteins, biopolymers, viruses, and bacteria. The motion of small particles in a fluid is a stochastic process. The classical example is Brownian motion, which was originally discovered through observation of pollen suspended in water. Once microscopy videos are obtained, the position of each particle is tracked through time. The result is a set of position-time series tracks. The tracks are typically used to infer properties of the fluid. The first example discovered was through observation of Brownian motion. The simplest stochastic model of Brownian motion involves a single parameter, the diffusivity. The Stokes-Einstein relation is a formula that relates the diffusivity and particle size to the fluid viscosity and temperature. In particle tracking microrheology, particle motion is used to estimate the viscosity and elastic properties of a non Newtonian visco-elastic fluid. In biology, many new applications for particle tracking are beginning to emerge, thanks to advances in microscopy, machine learning, and neural networks. A few examples are characterizing active bacterial motion of Salmonella in mucus and measuring macromolecular crowding in the cytoplasm of living cells.

## Requirements: 
This workshop assumes a basic understanding of probability and stochastic processes. Some amount of programming will be involved in all of the projects (students with complementary skills will be encouraged to form teams). We will primarily be using Python (but R, Julia, or C++ might be ok too). Students will need to bring a laptop or tablet equipped with a keyboard. The only required software is the Google Chrome internet browser with a Gmail or other Google account logged in. 

## Recommended software:
You are free to use whatever you want (R, Python, Matlab, etc). I have two recommendations (I think the first is the best option).
  1. Download and install Anaconda, which will allow you to run Jupyter notebooks locally on your machine. https://www.anaconda.com/products/individual#Downloads
  2. Download this repo into your Google Drive folder and use Google Colab. https://research.google.com/colaboratory/

For option 2 (Google Colab) there are several steps, unfortunately, before you can work with the local files. You must first "mount" your google drive. Then change your directory from within the Colab notebook to the repository folder that has the code and data files. This can be done as follows.

  1. On the left edge of the page, there is a small folder icon. If you click on that it opens up a Files browser (you might have to wait a few seconds for it to load). At the top of the new browser is another folder icon, this time with the Drive logo. If you click on this, it will create a new cell in your notebook with some code you can execute to mount your Drive. The code looks like this

    ```from google.colab import drive```
    ```drive.mount('/content/drive')```

  2. Execute the new cell. It will launch a new tab that you can use to authorize access to your Drive folder. After you agree to authorize, you have to copy an auth code. Once it is copied, go back to your Colab notebook. Paste the auth code in the space provided in the Colab notebook (in the output of the cell you executed to mount your Drive).

  3. You need to change your directory (within the notebook) to where you copied the Github repo in your Drive folder. You can do this with the following command (in a new cell)

    ```%cd /content/drive/<path to your repo folder>```

  where `<path to your repo folder>` is the path. If you navigate to your folder in the Files browser in the Colab notebook, you can right click on the folder icon to get an option to copy the path. Then you can simply past it into place in the above command.
