## Requirements

To install requirements:

```setup
!pip install -r requirements.txt
```

## Training

To train the model, run this command:

```train
!python trainer.py
```

or more advaned train:

```
!python trainer.py --hyperparameter_name=value
```

for example:

```
!python trainer.py --batch_size=16 --bert_model=indolem/indobertweet-uncased
```

You can see the other hyperparameter on util/hyperparameter.py
