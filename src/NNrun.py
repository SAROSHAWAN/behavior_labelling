from src.neuralNet.zeroshot import process_teacher_batch
from src.processData.sceneGenerator import scene_batch_gen
from src.neuralNet.encoding import process_observation_batch
from enum import Enum, auto

class models(Enum):
    ENCODE = auto()
    ZSHOT = auto()
    #GRU = auto()
    ALL = auto()

#note: 3rd param wants a list, even when pass isngle enum pls put in list []
def data_pipeline_helper(doc_container, registry, funct={models.ALL}): #default do all, both bart and sbert atm

    model_mapping = {
        models.ENCODE: process_observation_batch,
        models.ZSHOT: process_teacher_batch,
        #models.GRU: GRU_func,
    }
    if models.ALL in funct:
        active_models = [models.ENCODE, models.ZSHOT] #MOD WHEN MORE MODEL GET ADD
    else:
        active_models = funct
    # This will hold: { "funct name": { "CharacterName": [ {result_dict}, ... ] } }
    results = {model.name: {name: [] for name in registry.keys()} for model in active_models}

    # str, list
    batch_gen = scene_batch_gen(doc_container, registry)
    for target_name, scene_batch in batch_gen:
        for funct in active_models:
            processor = model_mapping[funct]
            processor(scene_batch, results[funct.name][target_name])
        
    return results
