import os, typing
from .util import list_folders
from collections import defaultdict
from .eval_sample import FloorplansAndPromptEvaluation, MetricSummaries, MetricResults
from ..metrics import Metrics
from functools import partial

class Evaluate:
    def __init__(self, root_dir='/network/scratch/l/luozhiha/results/', metrics='all', experiment_list='all', if_separate_num_room_results=False):
        self.root_dir = root_dir
        self.metrics = Metrics(metrics)
        self.model_tags = self.get_model_tags(experiment_list)
        sample_lookup = {model_tag: self.get_sample_ids(model_tag) for model_tag in self.model_tags}
        self.model_strats_lookup = self.get_model_strats_lookup(sample_lookup)
        self.RESULTS = defaultdict(lambda: defaultdict(MetricResults))
        self.SUMMARIES = defaultdict(lambda: defaultdict(MetricSummaries))
        self.SD_SUMMARIES = defaultdict(lambda: defaultdict(MetricSummaries))
        self.if_separate_num_room_results = if_separate_num_room_results
        if if_separate_num_room_results:
            self.RESULTS_separated_by_num_room = defaultdict(lambda: defaultdict(lambda: defaultdict(MetricResults)))
            self.SUMMARIES_separated_by_num_room = defaultdict(lambda: defaultdict(lambda: defaultdict(MetricSummaries)))
    
    def evaluate(self):
        from .plot import get_latex_from_df, get_df_from_summary_v2
        import pandas as pd
        summary_df = None
        for model_tag, strat_lookup in self.model_strats_lookup.items():
            strat_df = None
            path = os.path.join(self.root_dir, model_tag)
            for strat, sample_ids in strat_lookup.items():
                for sample_id in sample_ids:
                    plans = FloorplansAndPromptEvaluation(os.path.join(self.root_dir, model_tag, sample_id, strat), metrics=self.metrics)
                    plans.evaluate()
                    self.RESULTS[model_tag][strat] += plans.results
                    if self.if_separate_num_room_results:
                        self.RESULTS_separated_by_num_room[plans.get_num_rooms_from_prompt()][model_tag][strat] += plans.results
                    # plans.cleanup(keep_results=True, keep_summaries=True)
                    print(f"Evaluated {model_tag} {strat} {sample_id}")
                    # break # skipping evalutions for code testing; comment out this line to evaluate all samples

                self.SUMMARIES[model_tag][strat], self.SD_SUMMARIES[model_tag][strat] = self.RESULTS[model_tag][strat].summarize()
                self.save_summary(self.SUMMARIES[model_tag][strat], os.path.join(path, f"{strat}_avg_summary.pkl"))
                self.save_summary(self.SD_SUMMARIES[model_tag][strat], os.path.join(path, f"{strat}_sd_summary.pkl"))

                df = get_df_from_summary_v2(self.SUMMARIES[model_tag][strat],
                                            categories=self.SUMMARIES[model_tag][strat].__dict__.keys(),
                                            strat_name=model_tag+" "+strat+" avg")
                df_sd = get_df_from_summary_v2(self.SD_SUMMARIES[model_tag][strat],
                                               categories=self.SD_SUMMARIES[model_tag][strat].__dict__.keys(),
                                               strat_name=model_tag+" "+strat+" sd", if_std=True)

                strat_df = pd.concat([df, df_sd], axis=0) if strat_df is None else pd.concat([strat_df, df, df_sd], axis=0)
            with open(os.path.join(path, "evaluation_results.txt"), 'w') as f:
                f.write(get_latex_from_df(strat_df))
            strat_df.to_pickle(os.path.join(path, "evaluation_df.pkl"))
            summary_df = strat_df if summary_df is None else pd.concat([summary_df, strat_df], axis=0)
        with open(os.path.join(self.root_dir, "evaluation_results.txt"), 'w') as f:
            f.write(get_latex_from_df(summary_df))
        summary_df.to_pickle(os.path.join(self.root_dir, "evaluation_df.pkl"))

    
    def get_model_tags(self, experiment_list):
        model_tags = []
        experiments = list_folders(self.root_dir) if experiment_list == 'all' else experiment_list
        for out_dir in experiments:
            for model_tag in list_folders(os.path.join(self.root_dir, out_dir)):
                model_tags.append(os.path.join(out_dir, model_tag))
        return model_tags
    
    def get_sample_ids(self, model_tag):
        sample_ids = []
        for sample_id in list_folders(os.path.join(self.root_dir, model_tag)):
            sample_ids.append(sample_id)
        return sample_ids
    
    def get_model_strats_lookup(self, sample_lookup):
        ret = defaultdict(lambda: defaultdict(list))
        for model_tag, sample_ids in sample_lookup.items():
            for sample_id in sample_ids:           
                strat_list = self.get_sample_strats(model_tag, sample_id)
                for strat in strat_list:
                    ret[model_tag][strat].append(sample_id)
        return ret
        
    
    def get_sample_strats(self, model_tag, sample_id):
        sample_strats = []
        for sample_strat in list_folders(os.path.join(self.root_dir, model_tag, sample_id)):
            sample_strats.append(sample_strat)
        return sample_strats
    
    def save_summary(self, summary, save_path):
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(summary, f)
    
    def write_summary_to_latex_file(self, summary, strat_name, save_path):
        from .plot import get_latex_from_df, get_df_from_summary
        df = get_df_from_summary(summary, categories=summary.__dict__.keys(), strat_name=strat_name)
        with open(save_path, 'w') as f:
            f.write(get_latex_from_df(df))

    # def summarize(self):
    #     self.SUMMARIES = self.summarize_over_samples(self.RESULTS)
    #     if self.if_separate_num_room_results:
    #         for num_rooms, model_lookup in self.RESULTS_separated_by_num_room.items():
    #             buff = self.summarize_over_samples(model_lookup)
    #             # reorganize dictionary key order
    #             for model_tag, strat_lookup in buff.items():
    #                 for strat, summaries in strat_lookup.items():
    #                     self.SUMMARIES_separated_by_num_room[model_tag][strat][num_rooms] = summaries

    
    # def summarize_over_samples(self, results):
    #     ret = defaultdict(lambda: defaultdict(MetricSummaries))
    #     for model_tag, strat_lookup in results.items():
    #         for strat, summaries in strat_lookup.items():
    #             overall_summaries = MetricSummaries()
    #             for metric_name in self.metrics.metric_lookup.keys():
    #                 buff = None
    #                 valid_count = 0
    #                 for summary in summaries:
    #                     metric_score = getattr(summary, metric_name)
    #                     if not metric_score is None:
    #                         valid_count += 1
    #                         if isinstance(metric_score, tuple):
    #                             buff = (buff[0] + metric_score[0], buff[1] + metric_score[1]) if not buff is None else metric_score
    #                         else:
    #                             buff = buff+metric_score if not buff is None else metric_score
    #                 if isinstance(buff, tuple):
    #                     buff = buff[0]/valid_count, buff[1]/valid_count
    #                 else:
    #                     buff = buff / valid_count if not buff is None else None
    #                 setattr(overall_summaries, metric_name, buff)
    #             ret[model_tag][strat] = overall_summaries
    #     return ret
    
    # def save_summaries(self, save_path):
    #     import pickle
    #     def convert_to_dict(obj):
    #         if isinstance(obj, defaultdict):
    #             return {k: convert_to_dict(v) for k, v in obj.items()}
    #         else:
    #             return obj.__dict__
    #     with open(save_path, 'wb') as f:
    #         pickle.dump(convert_to_dict(self.SUMMARIES), f)
    #     if self.if_separate_num_room_results:
    #         with open(save_path[:-4]+"_rooms.pkl", 'wb') as f:
    #             pickle.dump(convert_to_dict(self.SUMMARIES_separated_by_num_room), f)
    
    # def load_summary(self, load_path):
    #     import pickle
    #     with open(load_path, 'rb') as f:
    #         self.SUMMARIES = pickle.load(f)
    #     if self.if_separate_num_room_results:
    #         with open(load_path[:-4]+"_rooms.pkl", 'rb') as f:
    #             self.SUMMARIES_separated_by_num_room = pickle.load(f)
    
    # def get_null_summary(self):
    #     null_summary = MetricSummaries()
    #     import typing
    #     # setup avg all
    #     for metric_name in self.metrics.metric_lookup.keys():
    #         metric_type = null_summary.__dataclass_fields__[metric_name].type
    #         if metric_type == typing.Tuple[float, float]:
    #             setattr(null_summary, metric_name, (0, 0))
    #         else:
    #             setattr(null_summary, metric_name, 0)
    #     return null_summary

    # def average_all_results(self, summary):
    #     avg_all = self.get_null_summary()
        
    #     metric_counter = defaultdict(int)
    #     for model_tag, strat_lookup in summary.items():
    #         for strat, _summary in strat_lookup.items():
    #             for metric_name in self.metrics.metric_lookup.keys():
    #                 if not isinstance(_summary, dict):
    #                     metric_score = getattr(_summary, metric_name)
    #                 else:
    #                     metric_score = _summary[metric_name]
    #                 if not metric_score is None:
    #                     metric_counter[metric_name] += 1
    #                     if isinstance(metric_score, tuple):
    #                         setattr(avg_all, metric_name, (getattr(avg_all, metric_name)[0] + metric_score[0], getattr(avg_all, metric_name)[1] + metric_score[1]))
    #                     else:
    #                         setattr(avg_all, metric_name, getattr(avg_all, metric_name) + metric_score)
    #     # compute the averages
    #     plotable_categories = []
    #     for metric_name in self.metrics.metric_lookup.keys():
    #         metric_score = getattr(avg_all, metric_name)
    #         if isinstance(metric_score, tuple):
    #             if metric_counter[metric_name] == 0:
    #                 setattr(avg_all, metric_name, None)
    #             else:
    #                 setattr(avg_all, metric_name, (metric_score[0]/metric_counter[metric_name], metric_score[1]/metric_counter[metric_name]))
    #                 plotable_categories.append(metric_name)
    #         else:
    #             if metric_counter[metric_name] == 0:
    #                 setattr(avg_all, metric_name, None)
    #             else:
    #                 setattr(avg_all, metric_name, metric_score/metric_counter[metric_name])
    #                 plotable_categories.append(metric_name)
    #     return avg_all, plotable_categories
    
    # def average_all_results_separated_by_num_rooms(self):
    #     # reorganize dictionary key order
    #     from copy import deepcopy
    #     buff = defaultdict(lambda: defaultdict(lambda: defaultdict(MetricSummaries))) #deepcopy(self.SUMMARIES_separated_by_num_room)
    #     for model_tag, strat_lookup in self.SUMMARIES_separated_by_num_room.items():
    #         for strat, summaries_by_num_rooms in strat_lookup.items():
    #             for num_rooms in summaries_by_num_rooms.keys():
    #                 buff[num_rooms][model_tag][strat] = summaries_by_num_rooms[num_rooms]
        
    #     avg_all  = {}
    #     for num_rooms, model_lookup in buff.items():
    #         avg_all[num_rooms] =  self.average_all_results(buff[num_rooms])
    #     return avg_all

    
    # def write_summary_to_latex_file(self, save_path, **kwargs):
    #     from .plot import get_latex_from_df, get_df_from_summary
    #     avg_all, plottable_categories = self.average_all_results(self.SUMMARIES)
    #     df = get_df_from_summary(avg_all, categories=plottable_categories, strat_name="Average")
    #     with open(save_path, 'w') as f:
    #         f.write(get_latex_from_df(df))
    
    # def write_summary_separated_by_num_rooms_to_latex_file(self, save_path, **kwargs):
    #     from .plot import get_latex_from_df, get_df_from_summary_separated_by_num_rooms_v2
    #     avg_all = self.average_all_results_separated_by_num_rooms()
    #     df = get_df_from_summary_separated_by_num_rooms_v2(avg_all, strat_name="Average")
    #     with open(save_path, 'w') as f:
    #         f.write(get_latex_from_df(df))

    # def plot_results(self):
    #     from .plot import get_df_from_summary, plot_radar_from_df, plot_categories_sanity_check, \
    #                       get_df_from_summary_separated_by_num_rooms_v2, plot_3d_from_df
    #     import pandas as pd
    #     for model_tag, strat_lookup in self.SUMMARIES.items():
    #         plot_stuff = []
    #         plotable_categories = plot_categories_sanity_check(strat_lookup, categories=list(self.metrics.metric_lookup.keys()))
    #         for strat, summary in strat_lookup.items():
    #             df_buff = get_df_from_summary(summary, categories=plotable_categories, strat_name=strat)
    #             if not df_buff is None: # if the experiment is missing, the df will be None
    #                 plot_stuff.append(df_buff)

    #         df = pd.concat(plot_stuff, axis=0)
    #         plot_radar_from_df(df, f"plots/{model_tag.replace('/', '_')}.png", title=model_tag)
    #     if self.if_separate_num_room_results:
    #         avg_all = self.average_all_results_separated_by_num_rooms()
    #         df = get_df_from_summary_separated_by_num_rooms_v2(avg_all, strat_name="Average")
    #         plot_3d_from_df(df, f"plots/averages_by_room.png", title="Average")
    
    # def clear_summaries_and_results(self):
    #     try:
    #         del self.RESULTS, self.SUMMARIES
    #     except:
    #         pass
    #     if self.if_separate_num_room_results:
    #         try:
    #             del self.RESULTS_separated_by_num_room, self.SUMMARIES_separated_by_num_room
    #         except:
    #             pass
            
                        
                