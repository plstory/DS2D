import sys
eval_path = sys.argv[1]

from src.utils import FloorplansAndPromptEvaluation, Evaluate

overall_evaluation = Evaluate(eval_path,
                              metrics='all',
                              experiment_list='all',
                              if_separate_num_room_results=False)
overall_evaluation.evaluate()