# Spatial-RSCR-Simulator
This repository include code for RSCR mechanism simulation, animation and database generation. The generated database can be downloaded at https://www.kaggle.com/datasets/purwarlab/rscr-mechanisms.

## Spatial RSCR Mechanism
Spatial RSCR mechanism is a 1-DOF, close-look mechanism. It can generate various spatial path, and has potential application on rehabilitation mechanisms, flapping-wing robot, etc.

![](https://github.com/Xueting-Deng/Spatial-RSCR-Simulator/blob/main/2024-08-14%2021-42-54.gif)

<img width="520" alt="1730162737602" src="https://github.com/user-attachments/assets/20487c6e-f1f8-4d82-a97b-ca4a80011004">

<img width="520" alt="1730165036766" src="https://github.com/user-attachments/assets/b4e42f4b-408c-4968-82b0-9979048b843d">


## The code
Functions.py - This file includes all functions needed for other scripts. The core function for simulation is solve_equations. This function uses scipy.root for solving the constraint equations, which are derived by the geometry constraints RSCR mechanism has. Our journal paper is just accepted but not published yet, please see the text book Kinematic and Dynamic Simulation of Multibody Systems The Real-Time Challenge for reference on how to derive the constraint equations. (https://link.springer.com/book/10.1007/978-1-4612-2600-0) Or contact me :D

RSCR-Animation - This file selects a specific RSCR mechanism to do the animation. The animation package used here is VPython. Feel free to change tha parameters to animate other RSCR mechanims.

RSCR-Database-555 - This file can generate a database with RSCR joints located inside a 5*5*5 mesh grid.

Data-Normalization - This file normalizes the path from database. It will cover the path to B-Spline first, and then translate, rotate, reflect and scale the path.


 

