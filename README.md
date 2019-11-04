A feed-forward neural network based software for treatment effect and propensity score estimation (in case of observational data). It is supporting both regression and classification treatment variable case.

### Installation

With Anaconda
Open the terminal that has anaconda installed and create a new environment:
```sh
conda create --name=name_of_the_environment python=3.6.8
```
Activate new environment
```sh
conda activate name_of_the_environment
```
Clone the repository onto your computer
```sh
git clone https://github.com/...
```
cd to the downloaded repository
```sh
cd causal_nets
```
Install the software by typing:
```sh
pip install -r requirements_cpu.txt
python3 setup_cpu.py install
```
or 
```sh
pip install -r requirements_gpu.txt
python3 setup_gpu.py install
```
The only difference is that one version will install tensorflow with only CPU support and the other will install tensorflow with GPU support. If you have a GPU available on your computer, it's highly encoureged to install tensorflow-gpu version. However, to actually enable GPU support for tensorflow, appropriate versions of CUDA and cudann toolkit have to be installed as well. For tensorflow-gpu 1.14 version CUDA 10 and cudnn 7.3. See more at:

exit the folder causal_nets
```sh
cd ..
```
Test installation by importing the package:
```Python
import causal_nets
```
You can exit the environment by typing:
```sh
conda deactivate
```
and enter it again:
```sh
conda activate name_of_the_environment
```

### Example

The below code is a short example showcasing the usage of causal_nets software.

```Python
import causal_nets

# Generate data.

```

### References

Farrell, M.H., Liang, T. and Misra, S., 2018:
'Deep neural networks for estimation and inference: Application to causal effects and other semiparametric estimands',
[<a href="https://arxiv.org/pdf/1809.09953.pdf">arxiv</a>]
