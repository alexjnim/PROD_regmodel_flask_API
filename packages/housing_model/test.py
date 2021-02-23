from housingmodel.processing.data_management import load_dataset
from housingmodel.config import config
from housingmodel.processing.validation import validate_inputs
import numpy as np

data = load_dataset(file_name=config.TRAINING_DATA_FILE)

validated_data = validate_inputs(input_data=data)

print(len(validated_data))

l = [
    (1, "longitude", "hello"),
    (34, "ocean_proximity", 45),
    (12, "households", np.nan),
    (4, "total_bedrooms", ""),
]

for swap in l:
    data.loc[swap[0], swap[1]] = swap[2]

validated_data = validate_inputs(input_data=data)
print(len(validated_data))
