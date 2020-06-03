import pickle
import xgboost

model_loc = "../RL_data/full_model.pickle.dat"
loaded_model = pickle.load(open(model_loc, "rb")) # So far failing here

from xgboost import Booster

booster = Booster()
# booster.save_model(..?)
### OR MAYBE
# loaded_model.booster.saveModel(path)