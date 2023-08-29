for noise in 0.001; do #  0.01 0.1
#  for param in GP_base_double GP_base_base # GP_0.01 envs
#  for param in random_base_double random_base_base random_bandit_double random_bandit_base
#  for param in sparse_base_double sparse_base_base sparse_bandit_double sparse_bandit_base random_base_double random_base_base random_bandit_double random_bandit_base
# random envs
  for param in random_base_double random_base_base random_bandit_double random_bandit_base
  do
    for i in {1..10}
    do
#      if [ $param = GP_base_double ]
      if [ $param = random_base_double ]
      then
        python main_no_visu.py --env $i --param $param --noise_sigma $noise
      else
        python main_no_visu.py --env $i --param $param --noise_sigma $noise --generate False
      fi
    done
  done
#
#  for param in GP_bandit_double GP_bandit_base
#  do
#    for i in {100..110}
#    do
#      if [ $param = GP_bandit_double ]
#      then
#        python main_no_visu.py --env $i --param $param --noise_sigma $noise
#      else
#        python main_no_visu.py --env $i --param $param --generate False --noise_sigma $noise
#      fi
#    done
#  done
done