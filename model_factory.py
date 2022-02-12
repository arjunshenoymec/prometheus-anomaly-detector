import model
import model_lstm
import model_fourier
import logging

_LOGGER = logging.getLogger(__name__)

class ModelPredictorFactory:
    def __init__(self):
        self.model_map = {'prophet': model.MetricPredictor, 'lstm':
                model_lstm.MetricPredictor, 'fourier':
                model_fourier.MetricPredictor}
        self.default_predictor = model_lstm.MetricPredictor
    def __getitem__(self, model_name):
        if model_name in self.model_map.keys():
            return self.model_map[model_name]
        else:
            _LOGGER.error(f"model {model_name} not found in model factory
            proceeding with default")
            return self.default_predictor
