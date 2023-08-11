# Deep Reinforcement Learning Based Anti-Jamming

This project is an implementation of the paper "Deep Reinforcement Learning Based Anti-Jamming Using Clear Channel Assessment Information in a Cognitive Radio Environment" (https://ieeexplore.ieee.org/document/9993858/). It uses a Double Deep Q-Network (DDQN) to train an agent for optimal channel selection in a wireless communication environment to mitigate the effects of jamming. The agent is trained and tested in an ns-3 Gym environment, where it interacts with different types of jammers.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.7 or higher
- TensorFlow 2.0 or higher
- Gym
- Matplotlib
- Numpy
- ns3 3.30
- ns3-gym 1.0.2

Please refer to the `requirements.txt` file for the specific versions of libraries used.

### Installing

1. Clone the repository to your local machine.
2. Install the required packages using pip:

```bash
pip install -r requirements.txt
```
3. Run the main script:

```bash
python antiJamming_v2.py
```

### Usage
The main script trains the DDQN agent for a specified number of episodes. The agent's performance is evaluated and plotted after training. The agent, rewards, throughput, and channel switching times are saved for further analysis.


### License
This project is licensed under the MIT License - see the LICENSE.md file for details

### Acknowledgements

This project is made possible by the following:

- **ns-3 Gym Environment**: The project utilizes the ns-3 Gym environment for training and testing the agent. More details about ns-3 Gym can be found [here](https://apps.nsnam.org/app/ns3-gym/).

- **Research Paper**: The implementation is based on the research paper titled "Deep Reinforcement Learning Based Anti-Jamming Using Clear Channel Assessment Information in a Cognitive Radio Environment". The paper provides the theoretical foundation for the project and can be accessed [here](https://ieeexplore.ieee.org/abstract/document/9993858/).
