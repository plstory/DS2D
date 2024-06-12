from .process_dataset import Floorplan, FloorplansAndPrompt
from ..metrics import Metrics
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings

@dataclass
class DataStatistics:
    total_area: float=None
    polygon_area: Tuple[float, float]=None
    room_count: int=None
    polygon_overlap_count: int=None
    # polygon_containment_count: int=None
    room_height: float=None
    room_width: float=None
    num_room_prompt: int=None
    room_area_prompt: float=None
    room_type_prompt: int=None
    room_id_type_match_prompt: int=None
    room_height_prompt: float=None
    room_width_prompt: float=None
    json_file: int=None
    json_strict_file: int=None
    total_file: int=None


@dataclass
class MetricSummaries:
    total_area_self_consistency: float=None
    polygon_area_self_consistency: Tuple[float, float]=None
    room_count_self_consistency: float=None
    room_id_self_consistency: float=None
    polygon_overlap_count_self_consistency: int=None
    # polygon_containment_count_self_consistency: int=None
    room_height_self_consistency: float=None
    room_width_self_consistency: float=None
    num_room_prompt_consistency: Tuple[int, int]=None
    room_id_prompt_consistency: Tuple[float, float]=None
    room_area_prompt_consistency: float=None
    polygon_area_sum_vs_total_area_prompt_consistency: float=None
    room_type_prompt_consistency: Tuple[float, float]=None
    room_id_type_match_prompt_consistency: float=None
    room_height_prompt_consistency: float=None
    room_width_prompt_consistency: float=None
    json_file_consistency: float=None
    json_strict_file_consistency: float=None

    def __str__(self):
        ret = ""
        for metric_name, result in self.__dict__.items():
            ret += f"Metric: {metric_name}"
            ret += f"\n\t{result}"
            ret += "\n"
        return ret

@dataclass
class MetricResults:
    total_area_self_consistency: List[float]=None
    polygon_area_self_consistency: List[Tuple[float, float]]=None
    room_count_self_consistency: List[bool]=None
    room_id_self_consistency: List[bool]=None
    polygon_overlap_count_self_consistency: List[int]=None
    # polygon_containment_count_self_consistency: List[int]=None
    room_height_self_consistency: List[float]=None
    room_width_self_consistency: List[float]=None
    num_room_prompt_consistency: List[Tuple[int, int]]=None
    room_id_prompt_consistency: List[Tuple[float, float]]=None
    room_area_prompt_consistency: List[float]=None
    polygon_area_sum_vs_total_area_prompt_consistency: List[float]=None
    room_type_prompt_consistency: List[Tuple[float, float]]=None
    room_id_type_match_prompt_consistency: List[float]=None
    room_height_prompt_consistency: List[float]=None
    room_width_prompt_consistency: List[float]=None
    json_file_consistency: List[bool]=None
    json_strict_file_consistency: List[bool]=None

    def _isempty(self):
        for metric_name, result in self.__dict__.items():
            if result is not None and len(result) > 0:
                return False
        return True

    def __str__(self):
        ret = ""
        for metric_name, result in self.__dict__.items():
            ret += f"Metric: {metric_name}"
            if isinstance(result, list):
                for i, r in enumerate(result):
                    ret += f"\n\tFloorplan {i}: {r}"
            else:
                ret += f"\n\t{result}"
            ret += "\n"
        return ret
    
    def __add__(self, other):
        new_results = MetricResults()
        if self._isempty():
            return other
        if other._isempty():
            return self
        for metric_name in self.__dict__.keys():
            setattr(new_results, metric_name, getattr(self, metric_name) + getattr(other, metric_name))
        return new_results
    
    def summarize(self):
        summaries, sd_summaries = MetricSummaries(), MetricSummaries()
        for metric_name in self.__dict__.keys():
            metric_score = getattr(self, metric_name)
            metric_score = [x for x in metric_score if x is not None]
            numel = len(metric_score)
            if numel == 0:
                continue
            elif isinstance(metric_score[0], Tuple):
                
                setattr(summaries, metric_name, tuple(sum(x) / numel for x in zip(*metric_score)))
            else:
                setattr(summaries, metric_name, sum(metric_score) / numel)

        for metric_name in self.__dict__.keys():
            if metric_name in ['json_file_consistency', 'json_strict_file_consistency', 'room_count_self_consistency', 'room_id_self_consistency']:
                continue
            metric_score = getattr(self, metric_name)
            metric_score = [x for x in metric_score if x is not None]
            numel = len(metric_score)
            if numel == 0:
                continue
            elif isinstance(metric_score[0], Tuple):
                # numel -= 1 # unbiased estimator
                temp1 = (sum((x - summaries.__dict__[metric_name][0])**2 for x, y in metric_score) / numel)**0.5
                temp2 = (sum((y - summaries.__dict__[metric_name][1])**2 for x, y in metric_score) / numel)**0.5
                if metric_name == 'polygon_area_self_consistency':
                    setattr(sd_summaries,
                            metric_name,
                            (temp1, None))
                else:
                    setattr(sd_summaries,
                            metric_name,
                            (temp1, temp2))
            else:
                # numel -= 1 # unbiased estimator
                setattr(sd_summaries,
                        metric_name,
                        (sum((x - summaries.__dict__[metric_name])**2 for x in metric_score) / numel)**0.5)

        return summaries, sd_summaries

class FloorplansAndPromptEvaluation(FloorplansAndPrompt):
    
    def __init__(self, path, json_strs=None, metrics='all'):
        super().__init__(path, json_strs)
        self.metrics = metrics if isinstance(metrics, Metrics) else Metrics(metrics)
        self.results = MetricResults()
        self.summaries = MetricSummaries()
        self.if_evlauated = False
        try:
            self.floorplans.pop('ground_truth')
        except KeyError:
            pass
    
    def cleanup(self, keep_results=False, keep_summaries=True):
        # cleanup everything except evaluation summaries
        del self.floorplans, self.metrics
        if not keep_results:
            del self.results
        if not keep_summaries:
            del self.summaries

    def get_num_rooms_from_prompt(self):
        return self.floorplans["prompt"].get_room_count()
    
    def summarize(self):
        warnings.warn("This method is deprecated. Use summarize from MetricResults instead.", DeprecationWarning)
        assert self.if_evlauated, "Floorplans have not been evaluated yet"
        for metric_name in self.metrics.metric_lookup.keys():
            metric_score = getattr(self.results, metric_name)
            metric_score = [x for x in metric_score if x is not None]
            numel = len(metric_score)
            if numel == 0:
                continue
            elif isinstance(metric_score[0], Tuple):
                setattr(self.summaries, metric_name, tuple(sum(x) / numel for x in zip(*metric_score)))
            else:
                setattr(self.summaries, metric_name, sum(metric_score) / numel)
    
    def evaluate(self):
        if self.if_evlauated:
            return self.results
        for metric_name in self.metrics.metric_lookup.keys():
            setattr(self.results, metric_name, self._evaluate(metric_name))
        self.if_evlauated = True
        return self.results
    
    def _evaluate(self, metric_name):
        try:
            metric = self.metrics.metric_lookup[metric_name]
        except KeyError:
            raise NotImplementedError(f"Metric {metric_name} not implemented")
        if 'file_consistency' in metric_name:
            per_floorplan_results = []
            for k, floorplan in self.floorplans.items():
                if k == "prompt" or k == "ground_truth":
                    continue
                per_floorplan_results.append(metric(floorplan))
            return per_floorplan_results
        elif 'self_consistency' in metric_name:
            per_floorplan_results = []
            for k, floorplan in self.floorplans.items():
                if k == "prompt" or k == "ground_truth":
                    continue
                if floorplan.validate_normal:
                    per_floorplan_results.append(metric(floorplan))
                else:
                    per_floorplan_results.append(None)
            return per_floorplan_results
        elif 'prompt_consistency' in metric_name:
            per_floorplan_results = []
            for k, floorplan in self.floorplans.items():
                if k == "prompt" or k == "ground_truth":
                    continue
                if floorplan.validate_normal:
                    per_floorplan_results.append(metric(floorplan, self.floorplans["prompt"]))
                else:
                    per_floorplan_results.append(None)
            return per_floorplan_results