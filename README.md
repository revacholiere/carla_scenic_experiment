after running Carla simulator, run the following example command

scenic YOUR_SCENIC_FILENAME.scenic -S -m scenic.simulators.carla.model --2d --time 200 --seed 100 --count 2

time, seed and count parameters are optional and you can change them as you would like. (--count determines how many individual simulations will be generated and run, no argument means it will run scenarios infinitely)
