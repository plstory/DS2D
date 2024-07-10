from src.utils import FloorplansAndPromptEvaluation, Evaluate
# plans = FloorplansAndPromptEvaluation('/network/scratch/l/luozhiha/results/results_v7_greedy/customer_dropout_longprompt/0/only_room_area/')
# plans.evaluate()
# plans.summarize()
# print(plans.results)
# print(plans.summaries)
# plans.cleanup(keep_results=True, keep_summaries=True)

overall_evaluation = Evaluate('/network/scratch/l/luozhiha/results/rplan/',
                              metrics='all',
                              experiment_list='all',
                              # experiment_list=['5_new_dropout','6_new_dropout','7_new_dropout','8_new_dropout',],
                              if_separate_num_room_results=False)
overall_evaluation.evaluate()
# overall_evaluation.summarize()
# overall_evaluation.plot_results()
# overall_evaluation.save_summaries(save_path='/network/scratch/x/xuolga/Results/results_all.pkl')
# import pdb; pdb.set_trace()