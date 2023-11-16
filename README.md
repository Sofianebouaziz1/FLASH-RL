# FLASH-RL

## Description

<img src="https://github.com/Sofianebouaziz1/FLASH-RL/blob/main/figures/Global_solution.PNG" width="50%" align="right"/>
<p style="text-align: justify"> <strong> FLASH-RL</strong> (Federated Learning Addressing System and Static Heterogeneity using Reinforcement Learning) is a novel and effective strategy for client selection in Federated Learning (FL) using Reinforcement Learning (RL). It addresses the challenges of system and static heterogeneity by considering the computational capabilities of clients, such as processing power and network connectivity, along with their data characteristics. <strong>FLASH-RL</strong> introduces a reputation-based utility function to evaluate client contributions based on their current and past performance. Additionally, an adapted algorithm is proposed to expedite the DDQL learning process.
</p>

[**Features**](#Features)
| [**Tutorial**](https://github.com/Sofianebouaziz1/FLASH-RL/blob/main/tutorial.ipynb)
| [**Structure**](#Code-structure)
| [**References**](#How-to-cite)

## Features

FLASH-RL package offers the following features:

* A FL system built from scratch, enabling the simulation of a server and several clients.
* A client selection in FL based on RL and more specifically on an adapted Double Deep Q Learning (DDQL) algorithm. This project marks the first release of a source code for this problematic.
* Multiple data division scenarios created for the MobiAct private dataset.
* Simulation of a heterogeneous environment in terms of the edge hardware equipment.
* Use of a reputation-based utility function to compute the reward attributed to each client.
* An adapted DDQL algorithm that allows multiple actions to be selected.


## Paper
FLASH-RL's paper has been accepted in the 41st IEEE International Conference on Computer Design (ICCD 2023). Please refer to the arXiv version [here] (https://arxiv.org/abs/2311.06917) for the full paper.

## Authors 

FLASH-RL has been developed by Sofiane Bouaziz, Hadjer Benmeziane, Youcef Imine, Leila Hamdad, Smail Niar and Hamza Ouarnoughi.

You can contact us by opening a new issue in the repository.

## Requirements
FLASH-RL has been implemented and tested with the following versions: 
- Python (v3.11.3).
- Pytorch (v2.0.0).
- Scikit-Learn (v1.2.2).
- Scipy (v1.10.1).
- FedLab (v1.3.0).
- NumPy (v1.24.3).


## Code structure


```
FLASH-RL/
 ├── RL/ --- Scripts for the RL module implementantaton.
 |    ├── DQL.py  --- Contains the adapted DDQL implementation.
 |    └── MLP.py --- The neural network structure used for the DDQL agent.
 | 
 ├── clientFL/ --- Defining the FL client class. 
 ├── data_division/ --- Creating and storing different non-iid data divisions.
 |    ├── MobiAct/MobiAct_divisions.py  --- Script for creating the MobiAct divisions.
 ├── data_manipulation/ --- Enabling the creation of structured non-iid data divisions among the clients for CIFAR-10 and MNIST. 
 ├── data_preprocessing/ --- Contains a script that pre-processes MobiAct data.
 ├── models/ --- Contains the different neural networks used for each dataset.
 └──serverFL/
      ├── Server_FAVOR.py  --- Contains the FAVOR implementation.
      ├── Server_FLASHRL.py  --- Contains the **FLASH-RL** implementation.
      └── Server_FedProx.py --- Contains the FedProx and FedAVG implementation.
```

## Experimental results
### General results
The following table summarizes the results we obtained by comparing FLASH-RL with FedAVG and FAVOR, based on accuracy (%) and latency (s).


<div style="text-align:center;">
  <img src="https://github.com/Sofianebouaziz1/FLASH-RL/blob/main/figures/overall_results.PNG" width="100%"/>
</div>

This results highlights the effectiveness of our method in striking a desirable balance between maximizing accuracy and minimizing end-to-end latency.

### Use case
The following figure shows the progression of the F1 score for the global model and end-to-end latency for each MobiAct division.

<div style="text-align:center;">
  <img src="https://github.com/Sofianebouaziz1/FLASH-RL/blob/main/figures/use_case.PNG" width="100%"/>
</div>

The Figure highlights FLASH-RL’s ability to find a compromise between maximizing the F1-score of the overall model and minimizing end-to-end latency


## How to cite?
In case you are using the FLASH-RL for your research, please consider citing our work::

```BibTex
@misc{bouaziz2023flashrl,
      title={FLASH-RL: Federated Learning Addressing System and Static Heterogeneity using Reinforcement Learning}, 
      author={Sofiane Bouaziz and Hadjer Benmeziane and Youcef Imine and Leila Hamdad and Smail Niar and Hamza Ouarnoughi},
      year={2023},
      eprint={2311.06917},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
