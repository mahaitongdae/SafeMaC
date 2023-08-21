for param in smcc_MacOpt_GP_random_double smcc_MacOpt_GP_random_bandit smcc_MacOpt_GP_random_bandit_double smcc_MacOpt_GP_random
#for param in smcc_MacOpt_GP_double smcc_MacOpt_GP_bandit smcc_MacOpt_GP_bandit_double smcc_MacOpt_GP
do
  for i in {1..10}
  do
    python main_no_visu.py --env $i --param $param
  done
done