"""doc
"""

import mlflow
import tensorflow as tf

from utils.utils import get_dataset, get_tokenizer, set_seed, tracking

# TODO SET SEED
set_seed()
# TODO LOSS, OPTIM
LOSS = 'temp'
OPTIM = 'temp'
tensorb = tf.keras.callbacks.TensorBoard()
callback_list = ''
experiment_id = tracking('TEST RUN')

# ! START HERE
train_data = get_dataset(file_path='data',shuffle= True)
valid_data = get_dataset(file_path='data')
tokenizer, vocab_size = get_tokenizer(dataset=train_data)

# TODO Model is been imported
model_name = '1DCNN'
main_model = tf.keras.Sequential([tf.keras.layers.Dense(10),
                                  tf.keras.layers.Dense(1)])

main_model.compile(loss=LOSS, optimizer=OPTIM, metrics=["f1_score"])

mlflow.tensorflow.autolog(log_datasets=False,)
with mlflow.start_run(run_name=model_name, experiment_id=experiment_id,):
    # TODO Setup distributed training
    # # TODO Setup train_time
