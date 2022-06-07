# Nonlinear endogenoues learning model
This repository contains a Pyomo based constrained nonlinear optimisation program. The program optimises the capacity expansion of a one-node system and implements the concept of experience curves to calculate technology costs endogenously in the optimisation. 


Run file "import_data.py" to import and create data files for the model

The model is implemented in experiment 1 to 5, each script can be independently run after importing data.

Exp1.py: Varying all learning rates

Exp2.py: Varying PV learning rate

Exp3.py: Implementing a floor cost

Exp4.py: Implementing carbon capture technologies

Exp5.py: Varying discount rates

LR_test.py: Validation of learning implementation

CO2_constraint.py: Validation of CO2 constraint implementation


## License

[MIT](https://choosealicense.com/licenses/mit/)
