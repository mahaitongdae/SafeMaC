#for param in smcc_MacOpt_GP_random_double smcc_MacOpt_GP_random #
#for param in smcc_MacOpt_GP_double smcc_MacOpt_GP
#do
#  for i in {1..10}
#  do
#    python main_no_visu.py --env $i --param $param
#  done
#done

#for param in smcc_MacOpt_GP_random_bandit_double smcc_MacOpt_GP_random_bandit
#for param in smcc_MacOpt_GP_bandit_double smcc_MacOpt_GP_bandit
for param in sparse_base_double sparse_bandit_double sparse_base_base sparse_bandit_base
do
  for i in {1..10}
  do
    if [ $param = sparse_base_double ]
    then
      python main_no_visu.py --env $i --param $param
    else
      python main_no_visu.py --env $i --param $param --generate False
    fi
  done
done