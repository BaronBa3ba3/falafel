import tensorflow as tf

import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback

run = neptune.init_run(
    project="falafel/falafel",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZjE2MmI4OC01ZWZmLTQwMzktYTkxZi0yZDM2MmZhNTBmMzMifQ==",
)  # your credentials

params = {"lr": 0.005, "momentum": 0.4, "epochs": 10, "batch_size": 64}
run["parameters"] = params

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax),
    ]
)

optimizer = tf.keras.optimizers.SGD(
    learning_rate=params["lr"],
    momentum=params["momentum"],
)

model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

neptune_cbk = NeptuneCallback(run=run, base_namespace="training")

model.fit(
    x_train,
    y_train,
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    callbacks=[neptune_cbk],
)

eval_metrics = model.evaluate(x_test, y_test, verbose=0)
for j, metric in enumerate(eval_metrics):
    run["eval/{}".format(model.metrics_names[j])] = metric

run.stop()